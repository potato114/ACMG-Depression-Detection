#!/usr/bin/env python3
"""Cut DAIC-WOZ participant speech using timestamps in the TSV transcripts.

Merge nearby consecutive Participant rows, then pack selected ranges into chunks
targeting 4-7 seconds. Long rows are not split; short neighbors may join them.
Write matching WAV/TXT pairs under {out_root}/{pid}/{label}{pid}_s0001.*.
Labels use train/dev PHQ8_Score and full-test PHQ_Score: <10 normal, else Depression.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PID_AUDIO_RE = re.compile(r"^(?P<pid>\d+)_AUDIO\.wav$")


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _load_scores(daic_root: Path) -> Dict[str, int]:
    scores: Dict[str, int] = {}

    for filename, score_col in (
        ("train_split_Depression_AVEC2017.csv", "PHQ8_Score"),
        ("dev_split_Depression_AVEC2017.csv", "PHQ8_Score"),
        ("full_test_split.csv", "PHQ_Score"),
    ):
        path = daic_root / filename
        if not path.exists():
            continue
        with path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("Participant_ID", "")).strip()
                if not pid:
                    continue
                try:
                    scores[pid] = int(float(str(row.get(score_col, "")).strip()))
                except ValueError:
                    continue
    return scores


def _iter_available_pids(unzip_dir: Path) -> List[str]:
    pids: List[str] = []
    for p in unzip_dir.glob("*_AUDIO.wav"):
        m = PID_AUDIO_RE.match(p.name)
        if m and p.is_file():
            pids.append(m.group("pid"))
    return sorted(pids, key=int)


def _write_audio(
    audio_path: Path,
    out_path: Path,
    ranges: List[Tuple[float, float]],
    overwrite: bool,
) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if len(ranges) == 1:
        start, stop = ranges[0]
        cmd += ["-ss", f"{start:.3f}", "-to", f"{stop:.3f}", "-i", str(audio_path)]
    else:
        # Join only selected ranges, dropping gaps and intervening Ellie speech.
        filters = [
            f"[0:a]atrim=start={start:.3f}:end={stop:.3f},"
            f"asetpts=PTS-STARTPTS[a{i}]"
            for i, (start, stop) in enumerate(ranges)
        ]
        labels = "".join(f"[a{i}]" for i in range(len(ranges)))
        filters.append(labels + f"concat=n={len(ranges)}:v=0:a=1[outa]")
        cmd += ["-i", str(audio_path), "-filter_complex", ";".join(filters), "-map", "[outa]"]
    cmd += ["-c:a", "pcm_s16le", "-y" if overwrite else "-n", str(out_path)]
    subprocess.run(cmd, check=True)


@dataclass(frozen=True)
class SegmentRow:
    start_time: float
    stop_time: float
    text: str

    @property
    def duration(self) -> float:
        return self.stop_time - self.start_time


def _load_participant_segments(transcript_path: Path, *, merge_gap: float) -> List[SegmentRow]:
    # Merge only *consecutive* Participant rows (i.e., do not merge across Ellie turns),
    # and only when the pause between them is small.
    merge_gap = max(0.0, float(merge_gap))

    segs: List[SegmentRow] = []
    cur: Optional[SegmentRow] = None
    with transcript_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            speaker = str(row.get("speaker", "")).strip()
            if speaker != "Participant":
                if cur is not None:
                    segs.append(cur)
                    cur = None
                continue
            st = str(row.get("start_time", "")).strip()
            et = str(row.get("stop_time", "")).strip()
            val = str(row.get("value", "")).strip()
            try:
                start = float(st)
                stop = float(et)
            except ValueError:
                continue
            if not (stop > start) or not val:
                continue
            if cur is None:
                cur = SegmentRow(start_time=start, stop_time=stop, text=val)
                continue

            gap = start - cur.stop_time
            if gap <= merge_gap:
                cur = SegmentRow(
                    start_time=cur.start_time,
                    stop_time=max(cur.stop_time, stop),
                    text=f"{cur.text} {val}",
                )
            else:
                segs.append(cur)
                cur = SegmentRow(start_time=start, stop_time=stop, text=val)

    if cur is not None:
        segs.append(cur)
    return segs


def _pack_segments_to_chunks(
    segs: List[SegmentRow],
    *,
    chunk_min: float,
    chunk_max: float,
) -> List[List[SegmentRow]]:
    chunk_min = max(0.0, float(chunk_min))
    chunk_max = float(chunk_max)
    if chunk_max <= 0:
        return [[s] for s in segs]

    # Stop near the midpoint when possible, then reduce short tail chunks.
    target = (chunk_min + chunk_max) / 2.0 if chunk_min > 0 else chunk_max
    chunks: List[List[SegmentRow]] = []
    cur: List[SegmentRow] = []
    cur_dur = 0.0

    def can_merge(previous: List[SegmentRow], following: List[SegmentRow]) -> bool:
        previous_dur = sum(s.duration for s in previous)
        # Already oversized chunks may absorb neighbors; never split a long row.
        return previous_dur > chunk_max or (
            previous_dur + sum(s.duration for s in following) <= chunk_max
        )

    for seg in segs:
        d = seg.duration
        if d <= 0:
            continue

        if d > chunk_max:
            if cur and cur_dur >= chunk_min:
                chunks.append(cur)
                cur = []
            chunks.append(cur + [seg])
            cur = []
            cur_dur = 0.0
            continue

        if cur and (
            cur_dur + d > chunk_max
            or (cur_dur >= chunk_min and cur_dur >= target)
        ):
            chunks.append(cur)
            cur = []
            cur_dur = 0.0
        cur.append(seg)
        cur_dur += d

    if cur:
        if chunks and can_merge(chunks[-1], cur):
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)

    packed: List[List[SegmentRow]] = []
    for chunk in chunks:
        if packed and sum(s.duration for s in chunk) < chunk_min and can_merge(packed[-1], chunk):
            packed[-1].extend(chunk)
        else:
            packed.append(chunk)
    return packed


def _process_one_pid(
    pid: str,
    *,
    unzip_dir: Path,
    out_root: Path,
    label: str,
    overwrite: bool,
    clean: bool,
    min_dur: float,
    merge_gap: float,
    chunk_min: float,
    chunk_max: float,
    dry_run: bool,
) -> Tuple[str, int, int]:
    audio_path = unzip_dir / f"{pid}_AUDIO.wav"
    transcript_path = unzip_dir / f"{pid}_TRANSCRIPT.csv"
    if not audio_path.exists() or not transcript_path.exists():
        return (pid, 0, 0)

    dur = _wav_duration_seconds(audio_path)
    segs = _load_participant_segments(transcript_path, merge_gap=merge_gap)
    # Filter tiny atomic segments before packing (default keeps all).
    if min_dur > 0:
        segs = [s for s in segs if s.duration >= min_dur]

    chunks = _pack_segments_to_chunks(segs, chunk_min=chunk_min, chunk_max=chunk_max)

    out_dir = out_root / pid
    if not dry_run:
        if clean and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0

    for idx, chunk in enumerate(chunks, start=1):
        # Normalize and clamp ranges per atomic segment.
        ranges: List[Tuple[float, float]] = []
        texts: List[str] = []
        for seg in chunk:
            start = max(0.0, float(seg.start_time))
            stop = min(dur, float(seg.stop_time)) if dur > 0 else float(seg.stop_time)
            if not (stop > start):
                continue
            ranges.append((start, stop))
            texts.append(seg.text.strip())
        if not ranges:
            skipped += 1
            continue

        base = f"{label}{pid}_s{idx:04d}"
        wav_out = out_dir / f"{base}.wav"
        txt_out = out_dir / f"{base}.txt"

        if dry_run:
            ok += 1
            continue

        # Write text first; if audio generation fails, it's still useful for debugging.
        if overwrite or (not txt_out.exists()):
            txt_out.write_text(" ".join([t for t in texts if t]).strip() + "\n", encoding="utf-8")
        if not overwrite and wav_out.exists():
            ok += 1
            continue
        try:
            _write_audio(audio_path, wav_out, ranges, overwrite=overwrite)
        except subprocess.CalledProcessError:
            skipped += 1
            # Best-effort cleanup of a potentially broken partial file.
            try:
                if wav_out.exists():
                    wav_out.unlink()
            except OSError:
                pass
            continue

        ok += 1

    return (pid, ok, skipped)


def _process_participants(worker, tasks: List[Tuple[str, str]], jobs: int):
    if jobs == 1:
        for pid, label in tasks:
            yield worker(pid, label=label)
        return

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(worker, pid, label=label) for pid, label in tasks]
        for future in as_completed(futures):
            yield future.result()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--daic-root",
        default=str(Path(__file__).resolve().parent),
        help="DAIC-WOZ root directory containing split CSVs (default: this script's directory).",
    )
    ap.add_argument(
        "--unzip-dir",
        default=None,
        help="Directory containing *_AUDIO.wav and *_TRANSCRIPT.csv (default: {daic_root}/unzip).",
    )
    ap.add_argument(
        "--out-root",
        default=None,
        help="Output root (default: {daic_root}/segments_participant).",
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Score threshold: score < threshold => normal else Depression (default: 10).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing segment files.")
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing per-subject output folders before writing (use when regenerating).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only count segments; do not write files.")
    ap.add_argument(
        "--min-dur",
        type=float,
        default=0.0,
        help="Skip segments shorter than this (seconds). Default keeps all segments.",
    )
    ap.add_argument(
        "--merge-gap",
        type=float,
        default=1.5,
        help="Merge consecutive Participant rows if the gap (next.start - prev.stop) <= this seconds (default: 1.5).",
    )
    ap.add_argument("--chunk-min", type=float, default=4.0, help="Target min chunk duration in seconds (default: 4).")
    ap.add_argument("--chunk-max", type=float, default=7.0, help="Target max chunk duration in seconds (default: 7).")
    ap.add_argument("--jobs", type=int, default=1, help="Parallelism across participants (default: 1).")
    ap.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional list of Participant_IDs to process (default: all with audio + score).",
    )

    args = ap.parse_args(argv)

    if args.chunk_max <= 0:
        print("ERROR: --chunk-max must be > 0", file=sys.stderr)
        return 2
    if args.chunk_min < 0:
        print("ERROR: --chunk-min must be >= 0", file=sys.stderr)
        return 2
    if args.chunk_max < args.chunk_min:
        print("ERROR: --chunk-max must be >= --chunk-min", file=sys.stderr)
        return 2

    daic_root = Path(args.daic_root).resolve()
    unzip_dir = Path(args.unzip_dir).resolve() if args.unzip_dir else (daic_root / "unzip")
    out_root = Path(args.out_root).resolve() if args.out_root else (daic_root / "segments_participant")

    if not unzip_dir.exists():
        print(f"ERROR: unzip dir not found: {unzip_dir}", file=sys.stderr)
        return 2

    scores = _load_scores(daic_root)
    if not scores:
        print(f"ERROR: no scores loaded from split files under: {daic_root}", file=sys.stderr)
        return 2

    if args.ids:
        target_pids = [pid.strip() for pid in args.ids if pid.strip()]
    else:
        target_pids = _iter_available_pids(unzip_dir)
    tasks = [
        (pid, "normal" if scores[pid] < args.threshold else "Depression")
        for pid in target_pids if pid in scores
    ]

    if not tasks:
        print("Nothing to do (no matching Participant_IDs with audio+score).", file=sys.stderr)
        return 0

    worker = partial(
        _process_one_pid,
        unzip_dir=unzip_dir,
        out_root=out_root,
        overwrite=args.overwrite,
        clean=args.clean,
        min_dur=args.min_dur,
        merge_gap=args.merge_gap,
        chunk_min=args.chunk_min,
        chunk_max=args.chunk_max,
        dry_run=args.dry_run,
    )
    total_ok = 0
    total_skipped = 0
    results = _process_participants(worker, tasks, max(1, args.jobs))
    for i, (pid, ok, skipped) in enumerate(results, start=1):
        total_ok += ok
        total_skipped += skipped
        print(f"[{i}/{len(tasks)}] {pid}: ok={ok} skipped={skipped}", flush=True)

    print(f"Done. participants={len(tasks)} segments_ok={total_ok} segments_skipped={total_skipped}")
    print(f"Output: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
