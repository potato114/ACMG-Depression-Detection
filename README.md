<div align="center">

# Learning to Attend to Depression-Related Patterns

### An Adaptive Cross-Modal Gating Network for Depression Detection

Official implementation of **ACMG**, accepted by **INTERSPEECH 2026**

<p>
  <a href="https://arxiv.org/abs/2604.10181"><img src="https://img.shields.io/badge/arXiv-2604.10181-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/INTERSPEECH-2026-1f6feb.svg" alt="INTERSPEECH 2026">
  <img src="https://img.shields.io/badge/Python-3.10-3776ab.svg" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-required-ee4c2c.svg" alt="PyTorch">
</p>

[Overview](#overview) | [Results](#results) | [Datasets](#datasets) | [Installation](#installation) | [Quick Start](#quick-start) | [Citation](#citation)

</div>

---

## News

- **April 2026:** The ACMG paper is available on [arXiv](https://arxiv.org/abs/2604.10181).
- **2026:** ACMG was accepted by **INTERSPEECH 2026**.

## Overview

Depression-related evidence is sparse and unevenly distributed across an
interview. Acoustic and textual sequences also contain irrelevant or
conflicting information. ACMG uses the global context of each modality to
adaptively reweight the frame- or token-level features of the other modality,
allowing the model to focus on clinically meaningful patterns before
multimodal fusion and classification.

### Highlights

- **Adaptive feature selection:** emphasizes informative acoustic frames and
  textual tokens instead of treating every segment equally.
- **Cross-modal conditioning:** uses global audio context to guide text gating
  and global text context to guide audio gating.
- **Two evaluation settings:** supports three-class PDCD2025 experiments and
  binary DAIC-WOZ experiments.
- **Interpretable behavior:** learned gates highlight low-energy speech regions
  and depression-related language patterns.

<p align="center">
  <img src="figures/final_framework.png" width="95%" alt="Overview of the ACMG architecture">
</p>

The repository includes two models built on the same Transformer backbone:

| Model | Description |
| --- | --- |
| `BaselineAT` | Independently encodes audio and text, then concatenates their pooled representations. |
| `ACMG` | Adds adaptive cross-modal gates before pooling and classification. |

Inputs use the shape `[batch, time, input_dim]`. Padding masks use
`[batch, time]`, where `True` denotes padding.

## Results

### PDCD2025

Subject-level results using the official five-fold cross-validation protocol:

| Method | Accuracy | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Transformer (Qwen) | 79.78 | 79.48 | 79.37 | 79.59 |
| + ACMG with unimodal gating | 80.88 | 80.58 | 80.31 | 80.85 |
| **+ ACMG with cross-modal gating** | **81.25** | **80.77** | **80.61** | **80.93** |

### DAIC-WOZ

Results on the official development partition using audio and text:

| Method | F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| Transformer (Qwen) | 64.38 | 64.77 | 64.13 |
| **+ ACMG with cross-modal gating** | **69.39** | **71.79** | **68.48** |

## Datasets

### PDCD2025

PDCD2025 contains 22.9 hours of online telephone counselling recordings from
272 participants aged 12 to 45, including healthy controls and participants
with clinically diagnosed depression. The task has three classes: healthy,
mild depression, and moderate depression.

> **Privacy notice:** PDCD2025 contains sensitive clinical speech and
> participant information. Participant consent and privacy restrictions do not
> permit redistribution, so neither the dataset nor its derived features can
> be released in this repository.

### DAIC-WOZ

[DAIC-WOZ](https://dcapswoz.ict.usc.edu/) contains 189 English clinical
interviews annotated with PHQ-8 depression scores. Following the dataset
protocol, the experiments use participant speech only and perform binary
classification with a PHQ-8 threshold of 10. Access must be requested from the
dataset owner; the data and derived features are not included here.

## Installation

```bash
git clone https://github.com/potato114/ACMG-Depression-Detection.git
cd ACMG-Depression-Detection

conda create -n acmg python=3.10 -y
conda activate acmg
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA environment if
the default package selected by `pip` is not appropriate.

## Data Preparation

Each split CSV must contain the following columns:

```text
segment_id,label,speech_label,txt_label,state
```

For PDCD2025, `state` is `train` or `test`. For DAIC-WOZ, it is `train` or
`dev`. Audio and text feature directories contain one `<segment_id>.pt` file
per segment. Each file stores a dictionary with `segment_id`, `subject_id`, and
`feature`; the paper configuration uses features with shape `[time, 1024]`.

Place the prepared files in the following layout, or update the paths in the
corresponding YAML file:

```text
data/
|-- splits/
|   |-- fold_1.csv
|   |-- ...
|   |-- fold_5.csv
|   `-- daic_woz_fold_1.csv
`-- features/
    |-- audio/
    |-- text/
    `-- daic_woz/
        |-- audio/
        `-- text/
```

## Quick Start

### PDCD2025

Baseline:

```bash
python baseline.py \
  --config config/config.yaml \
  --fold 1 \
  --model-type A_T \
  --device cuda:0
```

ACMG:

```bash
python baseline.py \
  --config config/config.yaml \
  --fold 1 \
  --model-type GateModel_A_T_bytime \
  --device cuda:0
```

Repeat the commands with `--fold 1` through `--fold 5` for five-fold
cross-validation.

### DAIC-WOZ

Baseline:

```bash
python baseline_DAIC_WOZ.py \
  --config config/config_DAIC.yaml \
  --model-type A_T \
  --device cuda:0
```

ACMG:

```bash
python baseline_DAIC_WOZ.py \
  --config config/config_DAIC.yaml \
  --model-type GateModel_A_T_bytime \
  --device cuda:0
```

Use `--device cpu` when CUDA is unavailable. Training outputs are written to
`outputs/`.

## Gate Visualization

The learned gates assign larger weights to depression-related acoustic regions
and textual cues while suppressing less informative features.

<p align="center">
  <img src="figures/find.png" width="95%" alt="Visualization of acoustic and textual gating weights">
</p>

The acoustic gates exhibit a negative correlation with acoustic energy, while
the textual gates emphasize negative sentiment and self-referential language.

## Verification

```bash
pytest -q
```

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{yu2026learning,
  title   = {Learning to Attend to Depression-Related Patterns: An Adaptive Cross-Modal Gating Network for Depression Detection},
  author  = {Yu, Hangbin and Yang, Yudong and Su, Rongfeng and Yan, Nan and Wang, Lan},
  journal = {arXiv preprint arXiv:2604.10181},
  year    = {2026}
}
```

## License and Responsible Release

An open-source license has not yet been applied. The copyright holder and
institutional authorization must be confirmed before licensing because some
source files currently carry an `xcq&AISML` copyright notice. Dataset access is
independent of the future code license; no code license grants permission to
use or redistribute PDCD2025 or DAIC-WOZ.

<details>
<summary><strong>Reproducibility and release notes</strong></summary>

- This repository excludes clinical data, participant identifiers, derived
  features, predictions, logs, and checkpoints.
- The feature-extraction pipeline is not included because the exact pretrained
  model revisions and preprocessing settings still need to be consolidated.
- For compatibility with the original experiments, the retained PDCD2025 flow
  evaluates the test split after every epoch and uses subject-level F1 to
  select the saved checkpoint. New experiments should use a separate
  validation split before reporting final test results. DAIC-WOZ uses its
  development split for checkpoint selection.

</details>
