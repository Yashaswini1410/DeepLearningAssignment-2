# Operation Cyber-Histology-Multi-Class Clinical Triage Pipeline

MAI/IDL SS26 - Assignment-2

**Team:** Chithaloori Yashwanth Goud (10001147) · Yashaswini Suresh (10001027)

---
## Overview

This repository contains the reconstructed and extended deep-learning pipeline for
multi-class medical image classification, rebuilt after the recovered draft codebase was
found to be non-functional.

The pipeline trains and evaluates three CNN architectures AlexNet, VGG16, and
ResNet18 across four diagnostic image datasets: cells, chest, lesions,
and orgs. It has GreenResNet, a heavily downscaled architecture
designed to preserve baseline accuracy at a fraction of the computational cost.

All models are configurable and executable from a single central configuration file
(`config.json`).

### Work in this repository

| Task | Scope | branch |
|---|---|---|
| Task 1 | Pipeline reconstruction, bug remediation, benchmarking infrastructure | `main` |
| Task 2 | Green Initiative architectural downscaling and efficiency verification | `main` |
| Task 3 | Transfer learning on the `organs` dataset | `development-task3` |

---

## Repository Structure

```
DeepLearningAssignment-2/
├── data.py                 # Dataset loading, shuffled train/val split, DataLoaders
├── models.py               # AlexNet, VGG16, ResNet18, GreenResNet
├── fit.py                  # Trainer class: training loop, validation, best-model checkpointing
├── train.py                # Single-run training driven by config.json
├── benchmark.py            # Task 1
├── benchmark_green.py      # Task 2
├── config.json             # Central configuration for all scripts
├── README.md               # Read me file
├── AUDIT_LOG.md            # log of every bug found and fixed
├── REPORT.md               # Consolidated benchmark report and recommendations
└── data/                   # Dataset files (not tracked in git)
    ├── cells.pt
    ├── chest.pt
    ├── lesions.pt
    └── orgs.pt
```

The `development-task3` branch contains two additional scripts for the transfer-learning
task:

```
├── pretrain.py             # Pretrains ResNet18 on `orgs`, saves orgs_pretrained.pt
└── transfer.py             # Adapts the pretrained weights to the scarce `organs` set
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `data.py` | Loads a dataset `.pt` file, shuffles the samples before splitting, carves out a validation split, and returns train / validation / test `DataLoader`s. |
| `models.py` | All model definitions. `VGGBlock` and `ResBlock` are the reusable building blocks; `GreenResNet` is the efficiency-optimized model for Task 2. |
| `fit.py` | The `Trainer` class encapsulates the training loop, per-epoch validation, and retention of the best-validation-epoch weights. |
| `train.py` | Trains one model on one dataset, as specified by `DATA` and `MODEL` in the config. |
| `benchmark.py` | Loops over every dataset and model, trains each, and reports accuracy, precision, recall, and macro F1 on the test set. |
| `benchmark_green.py` | Same matrix, extended with parameter count, total training runtime, per-sample inference latency, and peak training / inference memory. |
| `pretrain.py` | *(branch `development-task3`)* Trains ResNet18 on the large `orgs` dataset and saves the learned weights as `orgs_pretrained.pt`  the transferred knowledge. |
| `transfer.py` | *(branch `development-task3`)* Loads those weights and evaluates four adaptation regimes on the scarce `organs` dataset: zero-shot, scratch, fine-tune, and frozen-backbone. |

---

## Prerequisites

- Python 3.9 or newer
- PyTorch
- scikit-learn
- NumPy

A CUDA-capable GPU is optional. All scripts detect the device automatically and fall back
to CPU if no GPU is available.

### Installation

```bash
# clone the repository
git clone https://github.com/Yashaswini1410/DeepLearningAssignment-2.git
cd DeepLearningAssignment-2

# install dependencies
pip install torch scikit-learn numpy
```

---

## Data Setup

The dataset files are not tracked in this repository. Download them and place them in a
local folder matching the `DATA_PATH` value in `config.json` (default: `data/`).

```
data/
├── cells.pt
├── chest.pt
├── lesions.pt
└── orgs.pt
```

Each `.pt` file is a dictionary containing the tensors `train_images`, `train_labels`,
`test_images`, and `test_labels`.

Task 3 additionally requires `organs.pt` in the same folder.

---

## Usage

All commands are run from the repository root.

### Train a single model

Trains the model named in `MODEL` on the dataset named in `DATA`, printing per-epoch
training and validation loss and accuracy. The weights from the best validation epoch are
retained at the end of the run:

```bash
python train.py
```

### Run the training and test pipeline (Task 1)

Trains and tests every model in `MODELS` on every dataset in `DATASETS`. For each pair it
runs the full training routine, then evaluates the best weights **once on the held-out test
set**, reporting accuracy, precision, recall, and macro F1:

```bash
python benchmark.py
```

This is the entry point for the complete train-and-test pipeline. `train.py` performs
training only; all test-set evaluation happens in the benchmark scripts.

### Run the efficiency benchmark (Task 2)

Produces the same evaluation matrix, extended with parameter count, total training runtime,
per-sample inference latency, and peak training and inference memory:

```bash
python benchmark_green.py
```

### Run transfer learning (Task 3)

Task 3 lives on the `development-task3` branch and additionally requires `organs.pt` in the
data folder:

```bash
git checkout development-task3

# 1. pretrain ResNet18 on the large `orgs` dataset -> orgs_pretrained.pt
python pretrain.py

# 2. adapt to the scarce `organs` dataset and evaluate all four regimes
python transfer.py
```

`transfer.py` compares four adaptation regimes on the `organs` test set zero-shot
(pretrained weights, no training), scratch (random weights, no transfer), fine-tune
(pretrained weights, all layers, small learning rate), and frozen (pretrained backbone
frozen, fresh classifier head only) reporting accuracy, precision, recall, and macro F1
for each.

---


## Documentation

| Document | Contents |
|---|---|
| `AUDIT_LOG.md` | Every bug, corruption, and anti-pattern found in the recovered code: file, manifestation, root cause, correction, and the commit hash containing the fix. |
| `REPORT.md` | Consolidated benchmark report ,full metrics for every dataset/model pair, hyperparameter experiments, architectural recommendations, and the Green Initiative efficiency analysis. |

---

## Branches

| Branch | Contents |
|---|---|
| `main` | Task 1 (pipeline reconstruction) and Task 2 (Green Initiative) |
| `development-task3` | Task 3 transfer learning on the `organs` dataset (`pretrain.py`, `transfer.py`) |
