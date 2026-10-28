# Preprocessing and Feature Extraction

Run the commands below from the repository root. The legacy extraction scripts
use local checkpoint paths, data paths, output paths, and device settings near
the bottom of each file; update those settings for your environment.

## Contents

| Script | Purpose |
| --- | --- |
| `segment_participant_audio.py` | Cut and pack DAIC-WOZ participant speech into matching WAV/TXT pairs. |
| `Depression_get_all_segments.py` | Legacy PDCD2025 WAV/TextGrid segmentation; requires the external `functions_XCQ` helper. |
| `Depression_get_dataset_csv.py` | Create PDCD2025 fold CSVs from segment filenames and a fold spreadsheet. |
| `extract_bert.py` | Extract BERT text features. |
| `extract_roberta.py` | Extract RoBERTa text features. |
| `extract_hubert.py` | Extract HuBERT audio features at layers 12 and 24. |
| `extract_wav2vec2.py` | Extract Wav2Vec 2.0 audio features at layers 12 and 24. |
| `extract_whisper.py` | Extract Whisper encoder features. |
| `wavlm/extract_wavlm.py` | Extract WavLM audio features using the bundled model implementation. |

## Dependencies

DAIC-WOZ segmentation requires Python and `ffmpeg` on `PATH`; it uses only
Python's standard library. For the legacy model extractors and spreadsheet
preprocessing, install the optional dependencies:

```bash
python -m pip install -r requirements-feature-extraction.txt
```

The extractors use different model libraries. Use an environment compatible
with the chosen checkpoint, especially for the legacy Fairseq-based models.
Pretrained weights are obtained separately and placed at the paths configured
in each script.

## DAIC-WOZ Segmentation

Pass `--daic-root` explicitly: the script's default is its own directory, while
the dataset is normally stored elsewhere. Expected input layout:

```text
/path/to/DAIC-WOZ/
|-- train_split_Depression_AVEC2017.csv
|-- dev_split_Depression_AVEC2017.csv
|-- full_test_split.csv
`-- unzip/
    |-- 401_AUDIO.wav
    |-- 401_TRANSCRIPT.csv
    `-- ...
```

The transcript files have a `.csv` extension but tab-separated content with
`start_time`, `stop_time`, `speaker`, and `value` columns. The script selects
`Participant` rows. Defaults merge consecutive participant rows separated by
at most 1.5 seconds, then pack selected ranges toward 4-7 seconds. Long rows
are not split, and short tails may remain or join a longer chunk. Separate
selected ranges are concatenated without the intervening speech or gaps.

Preview one participant without writing or deleting outputs:

```bash
python extract_feature/segment_participant_audio.py \
  --daic-root /path/to/DAIC-WOZ \
  --ids 401 \
  --dry-run
```

Process the available participants:

```bash
python extract_feature/segment_participant_audio.py \
  --daic-root /path/to/DAIC-WOZ \
  --jobs 4
```

Outputs default to `/path/to/DAIC-WOZ/segments_participant/<participant_id>/`.
Each chunk has matching `.wav` and `.txt` files, for example
`normal401_s0001.wav` and `normal401_s0001.txt`. The default score threshold is
10: lower scores use `normal`, and scores at or above it use `Depression`.
Train/dev scores come from `PHQ8_Score`; full-test scores use `PHQ_Score`.
The standard test-split CSV has no score and is not used to assign labels.
Use `--out-root` to select another destination and `--help` for all options.

## Legacy Extraction Workflow

The scripts imported from the acoustic-textual baseline repository retain
their original behavior and local configuration. After configuring data,
checkpoints, and devices, invoke the selected extractor, for example:

```bash
python extract_feature/extract_bert.py
python extract_feature/extract_hubert.py
python extract_feature/wavlm/extract_wavlm.py
```

Their default input directory is `data/dataset/resample_segments`; outputs go
under `data/feature/Audio/` or `data/feature/Text/`. Feature files store
`segment_id`, `subject_id`, and `feature`. Update the ACMG YAML configuration
to match the output directories and the encoder's feature dimension.

Legacy scripts parse PDCD2025 filenames ending in an emotion suffix such as
`_neu.wav` or `_neg.txt`. The DAIC-WOZ segmentation script instead creates
names like `normal401_s0001.wav`. Adapt the filename parsing before applying
a legacy extractor to DAIC-WOZ; these are separate preparation paths.

`Depression_get_all_segments.py` imports the dataset-specific `functions_XCQ`
module, which is not provided in the source repository. Supply that helper
before using the legacy TextGrid segmentation workflow. The fold CSV script
also requires the corresponding fold spreadsheet. These legacy scripts do
not create DAIC-WOZ train/dev split CSVs automatically.

## Provenance and Notices

The 11 legacy files were copied from
[Emotional-Inconsistency-Depression](https://github.com/potato114/Emotional-Inconsistency-Depression/tree/2f5494503d70f96617994f8ffaf47ca71cd7d417/extract_feature)
at commit `2f5494503d70f96617994f8ffaf47ca71cd7d417`, with original file contents
and copyright notices preserved. Their import does not replace those notices
with a new license. The bundled WavLM implementation retains its
[MIT license](wavlm/LICENSE).

`segment_participant_audio.py` is the DAIC-WOZ preparation script added for
ACMG. Dataset files, derived features, and pretrained weights are obtained
separately; the directory contains code and documentation.
