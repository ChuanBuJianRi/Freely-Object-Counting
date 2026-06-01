# FreeCounting

Official code and experiments for ********* paper on free (training-free, prior-free) object counting.

## Overview

This project investigates class-agnostic object counting without requiring training data or class-specific priors. The repository contains reimplementations of recent counting methods along with comprehensive experiments on standard benchmarks.

## Repository Structure

```
├── ws_yiyang/
│   ├── OCCAM/                        # OCCAM reimplementation (training-free counting)
│   ├── OCCAM_experiments_series/     # Evaluation results on FSC-147
│   ├── omnicount/                    # OmniCount: SAM-based counting pipeline
│   ├── legacy/                       # Baseline models (Mask R-CNN, ViT, DINOv2, etc.)
│   └── official_data/                # Dataset configurations (FSC-147, OmniCount-191)
└── README.md
```

## Key Components

- **OCCAM**: A training-free, prior-free, multi-class object counting method based on SAM2 dense sampling + feature clustering. Supports single-seed (OCCAM-S) and multi-seed (OCCAM-M) configurations.
- **OmniCount**: SAM-based counting pipeline with support for single-class, multi-class, box-prompted, and aerial image counting.
- **Legacy Models**: Baseline counting models including Mask R-CNN, ViT-B/16, DINOv2-B/14, and Open Counter for comparison.

## FSC-147 Benchmark Results (OCCAM-S)

| Split | MAE  | RMSE | NAE  |
|-------|------|------|------|
| Val   | 43.65| 100.63| 1.16 |
| Test  | 45.47| 139.43| 1.20 |

## Quick Start

### OCCAM

```bash
cd ws_yiyang/OCCAM
pip install -r requirements.txt
python scripts/run_occam.py
```

### OmniCount

```bash
cd ws_yiyang/omnicount
pip install -r requirements.txt
bash scripts/download_checkpoints.sh
```

## Citation

```bibtex
@inproceedings{freeCounting2027,
  title={FreeCounting},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2027},
  series={AAAI '27}
}
```
