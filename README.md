# Learning to Attend to Depression-Related Patterns: An Adaptive Cross-Modal Gating Network for Depression Detection

Official implementation of **ACMG**, accepted by **INTERSPEECH 2026**.

**Paper:** [arXiv:2604.10181](https://arxiv.org/abs/2604.10181)

**Code:** [ACMG-Depression-Detection](https://github.com/potato114/ACMG-Depression-Detection)

Depression-related evidence is often sparse and unevenly distributed across an
interview. Acoustic and textual features can also contain irrelevant or
conflicting information. ACMG addresses this problem with an adaptive
cross-modal gating mechanism that uses the global context of one modality to
reweight the frame- or token-level features of the other modality. This helps
the model focus on depression-related acoustic regions and textual cues before
multimodal fusion and classification.

## Model Overview

![Overview of the ACMG architecture](figures/final_framework.png)

This repository provides two audio-text models with the same Transformer-based
backbone:

- `BaselineAT`: independently encodes acoustic and textual sequences, then
  concatenates the pooled representations for classification.
- `ACMG`: adds adaptive cross-modal gates before pooling and classification.

Both inputs use the shape `[batch, time, input_dim]`. Padding masks use
`[batch, time]`, where `True` denotes padding.

## Datasets

### PDCD2025

PDCD2025 contains 22.9 hours of online telephone counselling recordings from
272 participants aged 12 to 45, including healthy controls and participants
with clinically diagnosed depression. The task has three classes: healthy,
mild depression, and moderate depression. The experiments use the official
five-fold cross-validation protocol.

PDCD2025 contains sensitive clinical speech and participant information.
Participant consent and privacy restrictions do not permit redistribution, so
the dataset and its derived features cannot be released in this repository.

### DAIC-WOZ

[DAIC-WOZ](https://dcapswoz.ict.usc.edu/) contains 189 English clinical
interviews annotated with PHQ-8 depression scores. Following the dataset
protocol, this repository uses participant speech only and performs binary
classification with a PHQ-8 threshold of 10. Access must be requested from the
dataset owner; DAIC-WOZ and its derived features are not included here.

## Environment

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
corresponding YAML configuration:

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

Train the audio-text baseline:

```bash
python baseline.py \
  --config config/config.yaml \
  --fold 1 \
  --model-type A_T \
  --device cuda:0
```

Train ACMG:

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

Train the audio-text baseline:

```bash
python baseline_DAIC_WOZ.py \
  --config config/config_DAIC.yaml \
  --model-type A_T \
  --device cuda:0
```

Train ACMG:

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

![Visualization of acoustic and textual gating weights](figures/find.png)

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

## Release Notes

This repository excludes clinical data, participant identifiers, derived
features, predictions, logs, and checkpoints. The feature-extraction pipeline
is not included because the exact pretrained-model revisions and preprocessing
settings still need to be consolidated.

For behavioral compatibility with the original experiments, the retained
PDCD2025 training flow evaluates the test split after every epoch and uses its
subject-level F1 score to select the saved checkpoint. New experiments should
use a separate validation split before reporting final test results. DAIC-WOZ
uses its development split for checkpoint selection.

No open-source license is included yet. Confirm ownership and institutional
permission before selecting a license or redistributing the code.
