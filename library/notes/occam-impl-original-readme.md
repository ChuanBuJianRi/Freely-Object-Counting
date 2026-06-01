# OCCAM Counter

This repository currently contains the OCCAM project page. The official method
code has not been released here yet, so the `occam/` package is a lightweight
reimplementation from the paper:

> OCCAM: Class-Agnostic, Training-Free, Prior-Free and Multi-Class Object Counting

## Implemented Pipeline

The reconstruction follows the paper description:

1. Densely sample one seed point every 10 pixels.
2. Prompt SAM2 with each point and collect multimask outputs.
3. Post-process masks with connected components, size filtering, and IoU deduplication.
4. Crop each candidate object, preserve aspect ratio, pad to a square canvas.
5. Extract 2048-dimensional ImageNet ResNet-50 features.
6. Cluster features with a thresholded FINCH-style first-neighbor algorithm.

Two reported configurations are exposed:

- `single`: OCCAM-S, 224x224 crops, thresholds `12.0, 9.0, 7.75`
- `multi`: OCCAM-M, 500x500 crops, thresholds `5.0, 4.0, 3.0`

## Setup

Install the common dependencies:

```bash
pip install -r requirements.txt
```

Install SAM2 separately from the official repository:

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

Download a SAM 2.1 checkpoint and note the matching config name/path.

## Run

```bash
python scripts/run_occam.py \
  --image path/to/image.jpg \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint path/to/sam2.1_hiera_large.pt \
  --mode single \
  --output occam_result.jpg
```

The script prints JSON with one count per discovered cluster:

```json
{
  "num_clusters": 2,
  "counts": [17, 4],
  "total_count": 21
}
```

## Notes

This is not the authors' official implementation. The paper omits several
engineering details, so this version should be treated as a faithful starting
point for experimentation rather than an exact reproduction of the reported
numbers.