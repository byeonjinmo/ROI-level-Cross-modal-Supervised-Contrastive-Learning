# ROI-level Cross-modal Supervised Contrastive Learning for Multimodal Depression Risk Prediction

Official PyTorch implementation for the paper submitted to *Medical Image Analysis*.

> **RCSCL** aligns structural and functional brain representations at the ROI level through class-conditional cross-modal contrastive learning, enabling more discriminative multimodal depression risk prediction.

## Overview

We propose a two-stage multimodal framework that integrates **T1-weighted MRI** and **resting-state fMRI** for depression high-risk classification.
ROI-level tokens are extracted under a common **Schaefer 200-parcel atlas** from both modalities, aligned in a shared embedding space via **supervised contrastive learning**, and fused through **self-attention** for final classification.

### Architecture

```
                          Stage A: Contrastive Pretraining
                    ┌─────────────────────────────────────────┐
                    │                                         │
  T1 MRI ──► MedicalNet ──► Layer2 ──► ROI Pooling ──► Attention ──► Projection ──► z_s
  (91x109x91) ResNet-18     Feature     (Schaefer200)   Pooling       Head           │
                             Map         (B,200,128)    (B,128)      (B,128)         │
                                                                                  InfoNCE
                                                                                   Loss
  rs-fMRI ──► GAT ──────────────────► Node Embeddings ──► Attention ──► Projection ──► z_f
  (graph)     (2-layer,                (B,200,256)        Pooling       Head           │
               4-head)                                   (B,256)      (B,128)         │
                    │                                         │
                    └─────────────────────────────────────────┘

                          Stage B: Classification
                    ┌─────────────────────────────────────────┐
                    │                                         │
  T1 global feat ──►│──► LayerNorm ──► Project ──┐            │
  (512-d)           │                            ├──► Self-Attention ──► Classifier ──► Risk
  fMRI global feat ─│──► LayerNorm ──► Project ──┘     Fusion              (MLP)       Score
  (256-d)           │                                         │
                    └─────────────────────────────────────────┘
```

### Two-Stage Training

| Stage | Objective | Description |
|-------|-----------|-------------|
| **A** | Contrastive pretraining | Align T1 ROI tokens and fMRI node embeddings via class-conditional InfoNCE loss |
| **B** | Classification | Fine-tune with focal/BCE loss + optional contrastive regularization |

### Fusion Strategies

| Method | Flag | Description |
|--------|------|-------------|
| Self-attention | `--fusion-type attn` | Multi-head self-attention over modality tokens **(default)** |
| Concatenation | `--fusion-type concat` | Direct concatenation of projected features |
| Gated | `--fusion-type gated` | Learnable per-sample modality weighting |
| Cross-attention | `--fusion-type cross_attn` | Bidirectional cross-attention (GNN <-> T1) |
| Optimal transport | `--fusion-type ot` | Sinkhorn OT for soft ROI correspondence |

## Installation

```bash
git clone https://github.com/byeonjinmo/RCSCL.git
cd RCSCL
pip install -r requirements.txt
```

### Requirements
- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Geometric >= 2.3
- nibabel, torchio, scikit-learn, scipy

## Data Preparation

### Directory Structure
```
data/
  label.csv                         # Columns: Subject_ID, Depression_HighRisk
  T1_MNI/                           # T1 MRI volumes (MNI-registered, .nii.gz)
    BM0001_T1_brain_MNI_flirt.nii.gz
    ...
  outputs/                          # rs-fMRI graph features per subject
    BM0001/
      atlas_only_schaefer200/
        NODE_local_activity.csv     # ReHo, ALFF_z, fALFF
        NODE_time_stats.csv         # mean, std, skew, kurt
        ADJ_abs_dens10.csv          # Adjacency matrix (top 10% threshold)
    ...
  atlas/
    Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii
pretrain/
  resnet_18_23dataset.pth           # MedicalNet pretrained weights
```

### Node Features (7-dim per ROI)

Each of the 200 ROIs has a 7-dimensional feature vector constructed from preprocessed rs-fMRI:

| Feature | Source | Description |
|---------|--------|-------------|
| ReHo | `NODE_local_activity.csv` | Regional homogeneity (local synchronization) |
| ALFF | `NODE_local_activity.csv` | Amplitude of low-frequency fluctuations |
| fALFF | `NODE_local_activity.csv` | Fractional ALFF |
| mean | `NODE_time_stats.csv` | Mean of ROI time series |
| std | `NODE_time_stats.csv` | Standard deviation |
| skewness | `NODE_time_stats.csv` | Skewness |
| kurtosis | `NODE_time_stats.csv` | Kurtosis |

## Training

### Proposed method (recommended)

```bash
python train.py \
    --pretrain-contrastive \
    --supervised-contrastive \
    --contrastive-mode hybrid \
    --pretrain-epochs 50 \
    --epochs 150 \
    --lr-t1 2e-5 \
    --lr-gnn 5e-5 \
    --lr-fusion 1e-4 \
    --contrastive-tau 0.05 \
    --fusion-dropout 0.4 \
    --patience 30 \
    --save-dir ./results
```

<details>
<summary><b>What each flag does</b></summary>

| Flag | Value | Role |
|------|-------|------|
| `--pretrain-contrastive` | - | Enable Stage A contrastive pretraining |
| `--supervised-contrastive` | - | Use class labels to form positive/negative pairs |
| `--contrastive-mode hybrid` | `hybrid` | Combine self-match and same-class alignment |
| `--pretrain-epochs` | `50` | Number of Stage A epochs |
| `--epochs` | `150` | Number of Stage B classification epochs |
| `--lr-t1` | `2e-5` | Learning rate for MedicalNet ResNet-18 encoder |
| `--lr-gnn` | `5e-5` | Learning rate for GAT encoder |
| `--lr-fusion` | `1e-4` | Learning rate for fusion module and classifier |
| `--contrastive-tau` | `0.05` | InfoNCE temperature (lower = sharper) |
| `--fusion-dropout` | `0.4` | Dropout rate in fusion layers |
| `--patience` | `30` | Early stopping patience (epochs) |
| `--save-dir` | `./results` | Output directory for models and metrics |

</details>

### Single modality baselines

```bash
# T1 MRI only (MedicalNet ResNet-18)
python train_single_modality.py --modality t1 --save-dir results_t1

# rs-fMRI only (GAT)
python train_single_modality.py --modality fmri --save-dir results_fmri
```

### External validation (SRPBS)

```bash
python evaluate_external.py \
    --model-dir ./results \
    --external-root ./data/external \
    --train-normalizer ./results/feat_norm.npz \
    --use-contrastive \
    --save-dir ./results_external
```

### Ablation studies

```bash
# Compare fusion strategies
python scripts/run_fusion_comparison.py --exp quick

# Compare contrastive learning modes
python scripts/run_contrastive_comparison.py

# Full ablation (fusion + contrastive + class weighting)
python scripts/run_ablation_study.py --exp all
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 4 | Batch size (small for 3D volumes) |
| `--lr-t1` | 5e-5 | T1 encoder learning rate |
| `--lr-gnn` | 1e-4 | GNN encoder learning rate |
| `--lr-fusion` | 5e-4 | Fusion / classifier learning rate |
| `--contrastive-tau` | 0.07 | InfoNCE temperature |
| `--pretrain-epochs` | 20 | Stage A contrastive pretraining epochs |
| `--fusion-type` | `attn` | Fusion strategy (`concat`, `gated`, `attn`, `cross_attn`, `ot`) |
| `--fusion-dropout` | 0.3 | Dropout in fusion layers |
| `--patience` | 20 | Early stopping patience |
| `--diagonal-weight` | 2.0 | Self-match weight in hybrid contrastive mode |

## Project Structure

```
RCSCL/
  models/
    resnet3d.py            # MedicalNet 3D ResNet-18 with dilated convolutions
    gnn.py                 # GNNBackbone (GCN / GraphSAGE / GAT)
    multimodal_fusion.py   # MultimodalFusion with 5 fusion strategies
    contrastive.py         # InfoNCE loss, AttentionPooling, ContrastiveModule
    roi_pooling.py         # Schaefer200 atlas-based 3D ROI pooling
    ot_fusion.py           # Sinkhorn optimal transport fusion
  utils.py                 # Shared utilities (metrics, data loading, normalization)
  train.py                 # Main training script (5-fold CV, two-stage)
  train_single_modality.py # Single modality baselines (T1 only / fMRI only)
  evaluate_external.py     # External validation on SRPBS dataset
  scripts/
    run_fusion_comparison.py
    run_contrastive_comparison.py
    run_ablation_study.py
```

## Citation

```bibtex
@article{byeon2025rcscl,
  title={ROI-level Cross-modal Supervised Contrastive Learning for Multimodal Depression Risk Prediction with rs-fMRI and T1 MRI},
  author={Byeon, Jinmo and Lee, Hakjin and Jang, MyeongGyun and Yoon, Sujung and Lee, Hwamin},
  journal={Medical Image Analysis},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
