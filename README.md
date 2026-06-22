# Data-oriented Self-supervised Learning for Cancer of Unknown Primary (CUP) Diagnosis

## Overview

Cancer of Unknown Primary (CUP) is a metastatic tumor histologically confirmed as cancer, but whose primary site cannot be identified. Despite advances in diagnostic technology, automated CUP diagnosis remains clinically challenging due to limited labeled data, tumor heterogeneity, and the lack of standardized biomarkers.

This repository provides the code for our CUP-specific self-supervised learning framework, which integrates:

- **DNA methylation arrays** converted into images and processed via a Vision Transformer (ViT) backbone
- **microRNA (miRNA) expression data** transformed using centered nonlinear transformation to preserve relative expression patterns
- **Histopathology images** (label-limited) combined with miRNA data to learn complementary morphological representations

By fusing molecular and histopathological features, the model learns subtle, diffuse, and context-dependent morphological cues characteristic of metastatic lesions in CUP.

---

## Method

 ![Framework Overview](Journal.png)

Key design choices:

- Self-supervised pre-training tailored for CUP's data-scarce setting
- Training objectives designed to capture heterogeneous and ambiguous appearances of metastatic lesions
- Multimodal integration of molecular and morphological features for improved generalization

---

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.8+. See `requirements.txt` for full dependency list.

---

## Data Preparation

This work uses a real-world CUP benchmark dataset. Please refer to the paper for details on data sources and preprocessing steps.

Once data is downloaded and preprocessed, organize it as follows:

```
data/
├── methylation/        # Methylation array images
├── mirna/              # miRNA expression files (.csv or .tsv)
└── histopathology/     # Histopathology image patches
```

---

## Training

```bash
python train.py --config configs/cup_ssl.yaml
```

Key configuration options (see `configs/cup_ssl.yaml`):

| Argument | Description |
|---|---|
| `--data_root` | Path to dataset root |
| `--modality` | `methylation`, `mirna`, `hist`, or `all` |
| `--epochs` | Number of training epochs |
| `--batch_size` | Batch size |
| `--lr` | Learning rate |

---

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --data_root data/
```

---

## Results

Our framework outperforms comparable state-of-the-art baselines on the real-world CUP benchmark. See the paper for full quantitative results and analysis.
