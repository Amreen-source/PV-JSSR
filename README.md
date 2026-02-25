# PV-JSSR: Joint Super-Resolution and Segmentation with Cross-Task Interaction for Photovoltaic Panel Mapping

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of **PV-JSSR**, a unified framework that jointly performs 8× super-resolution and semantic segmentation of photovoltaic panels from multi-resolution satellite imagery.

> **PV-JSSR: Joint Super-Resolution and Segmentation with Cross-Task Interaction for Photovoltaic Panel Mapping from Multi-Resolution Imagery**  
> Amreen Batool
> Department of Electronic Engineering, Jeju National University

---

## Highlights

- **Joint SR + Segmentation**: Simultaneously produces 8× super-resolved imagery (0.8m → 0.1m GSD) and semantic segmentation masks in a single forward pass
- **Cross-Task Interaction Module (CTIM)**: Bidirectional feature exchange between SR and segmentation branches — SR→Seg for boundary precision, Seg→SR for reconstruction quality
- **Multi-Resolution Consistency Module (MRCM)**: Enforces cross-scale feature coherence across three resolution levels (0.1m, 0.3m, 0.8m GSD)
- **Resolution-Aware GSD Embedding**: Learnable conditioning that adapts feature extraction based on input resolution
- **State-of-the-art results**: IoU 0.865, Dice 0.919, PSNR 16.78 dB on the Jiangsu Province PV dataset

## Architecture Overview

```
Input (128×128, 0.8m GSD)
        │
        ▼
┌─────────────────────┐
│  Shared Encoder      │◄── GSD Embedding (sinusoidal PE + MLP)
│  (Swin Transformer)  │
│  F1, F2, F3, F4      │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│   SR   │ │Segmentation│
│Decoder │ │  Decoder   │
│(SGDM)  │ │  (U-Net)   │
└───┬────┘ └─────┬──────┘
    │             │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │    CTIM     │  ← Bidirectional feature exchange
    │ SR↔Seg      │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │    MRCM     │  ← Multi-resolution consistency
    └─────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
SR Output    Seg Masks
(512×512)    (512², 256², 128²)
```

## Results

### Comparison with State-of-the-Art

| Method | IoU↑ | Dice↑ | PSNR↑ | SSIM↑ |
|--------|------|-------|-------|-------|
| U-Net | 0.720 | 0.837 | 12.50 | 0.65 |
| DeepLabV3+ | 0.780 | 0.876 | 13.80 | 0.72 |
| SwinIR+Seg | 0.800 | 0.889 | 15.20 | 0.78 |
| SRSEG | 0.820 | 0.901 | 15.90 | 0.81 |
| **PV-JSSR (Ours)** | **0.865** | **0.919** | **16.78** | **0.83** |

### Component Ablation

| Configuration | IoU↑ | PSNR↑ | ΔIoU |
|--------------|------|-------|------|
| Baseline | 0.780 | 14.20 | – |
| + GSD Embedding | 0.800 | 14.80 | +2.0% |
| + CTIM | 0.830 | 15.60 | +3.0% |
| + MRCM | 0.850 | 16.20 | +2.0% |
| + Boundary Loss | 0.865 | 16.78 | +1.5% |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7

```bash
# Clone the repository
git clone https://github.com/jejunu-wind-ai/PV-JSSR.git
cd PV-JSSR

# Create conda environment
conda create -n pvjssr python=3.10 -y
conda activate pvjssr

# Install dependencies
pip install -r requirements.txt
```

## Dataset

We use the **Multi-Resolution PV Panel Dataset** from Jiangsu Province, China ([Jiang et al., 2021](https://doi.org/10.5194/essd-13-5493-2021)).

### Dataset Structure

```
data/
├── PV01/                    # Ground-mounted solar farms (922 samples)
│   ├── 0.1m/
│   │   ├── images/          # High-resolution RGB images (512×512)
│   │   └── masks/           # Binary segmentation masks
│   ├── 0.3m/
│   │   ├── images/          # Medium-resolution RGB images (256×256)
│   │   └── masks/
│   └── 0.8m/
│       ├── images/          # Low-resolution RGB images (128×128)
│       └── masks/
├── PV03/                    # Mixed urban-rural installations (2,308 samples)
│   └── ...                  # Same structure as PV01
├── PV08/                    # Rooftop urban installations (486 samples)
│   └── ...
└── splits/
    ├── train.txt            # 2,601 samples (70%)
    ├── val.txt              # 557 samples (15%)
    └── test.txt             # 558 samples (15%)
```

### Download

1. Download the raw dataset from: [https://doi.org/10.5194/essd-13-5493-2021](https://doi.org/10.5194/essd-13-5493-2021)
2. Run the preprocessing script:

```bash
python scripts/prepare_dataset.py --raw_dir /path/to/raw_data --output_dir ./data
```

### Dataset Statistics

| Region | Type | Samples | Train | Val | Test |
|--------|------|---------|-------|-----|------|
| PV01 | Ground-mounted | 922 | 645 | 139 | 138 |
| PV03 | Mixed urban-rural | 2,308 | 1,616 | 346 | 346 |
| PV08 | Rooftop urban | 486 | 340 | 72 | 74 |
| **Total** | | **3,716** | **2,601** | **557** | **558** |

## Usage

### Training

```bash
# Full training with default config (3-stage curriculum)
python train.py --config configs/default.yaml --data_dir ./data --gpu 0

# Resume from checkpoint
python train.py --config configs/default.yaml --resume checkpoints/stage2_epoch150.pth
```

The training proceeds in three curriculum stages:
1. **Stage 1** (epochs 1–50): Encoder pretraining with independent decoders
2. **Stage 2** (epochs 51–150): Joint training with CTIM activation
3. **Stage 3** (epochs 151–180): Full fine-tuning with MRCM

### Evaluation

```bash
# Evaluate on test set
python evaluate.py --config configs/default.yaml \
                   --checkpoint checkpoints/pvjssr_best.pth \
                   --data_dir ./data \
                   --save_vis

# Per-region evaluation
python evaluate.py --config configs/default.yaml \
                   --checkpoint checkpoints/pvjssr_best.pth \
                   --per_region
```

### Inference on Custom Images

```bash
# Single image inference
python inference.py --checkpoint checkpoints/pvjssr_best.pth \
                    --input path/to/image.tif \
                    --gsd 0.8 \
                    --output results/

# Batch inference
python inference.py --checkpoint checkpoints/pvjssr_best.pth \
                    --input_dir path/to/images/ \
                    --gsd 0.8 \
                    --output results/
```

## Pre-trained Models

| Model | Input GSD | IoU | PSNR | Download |
|-------|-----------|-----|------|----------|
| PV-JSSR (full) | 0.8m | 0.865 | 16.78 | [Link](#) |
| PV-JSSR (w/o MRCM) | 0.8m | 0.830 | 15.60 | [Link](#) |

## Project Structure

```
PV-JSSR/
├── configs/
│   └── default.yaml              # Training configuration
├── data/
│   ├── __init__.py
│   └── pv_dataset.py             # Dataset and data loading
├── models/
│   ├── __init__.py
│   ├── pv_jssr.py                # Main PV-JSSR model
│   ├── encoder.py                # Resolution-Aware Shared Encoder
│   ├── sr_decoder.py             # SR Decoder with PV-Semantic Guidance
│   ├── seg_decoder.py            # Segmentation Decoder with SR Injection
│   ├── ctim.py                   # Cross-Task Interaction Module
│   ├── mrcm.py                   # Multi-Resolution Consistency Module
│   └── gsd_embedding.py          # Learnable GSD Embedding
├── losses/
│   ├── __init__.py
│   └── losses.py                 # All loss functions
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # IoU, Dice, PSNR, SSIM computation
│   └── visualization.py          # Result visualization utilities
├── scripts/
│   └── prepare_dataset.py        # Dataset preparation script
├── train.py                      # Training entry point
├── evaluate.py                   # Evaluation script
├── inference.py                  # Inference on new images
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{batool2025pvjssr,
  title={PV-JSSR: Joint Super-Resolution and Segmentation with Cross-Task Interaction for Photovoltaic Panel Mapping from Multi-Resolution Imagery},
  author={Batool, Amreen and Kim, Yong-Won and Byun, Yung-Cheol},
  journal={},
  year={2025}
}
```
