# occam/ — OCCAM baseline reimplementation

Faithful reconstruction of the OCCAM pipeline (Spanakis et al., arXiv 2601.13871),
ported here from `FreeCounting/ws_yiyang/OCCAM/occam/`. This package is the
baseline that all GOC ablations replace / augment.

## pipeline

1. SAM2 AMG produces dense, NMS-deduplicated mask candidates.
2. `apply_mask_policy` filters them (8 policies P0..P7; see `masks.py`).
3. `ResNetFeatureExtractor` crops each mask, pads to 224×224 (single) or
   500×500 (multi), extracts 2048-d ImageNet ResNet-50 features.
4. Clustering — either:
   - `thresholded_finch` (paper baseline, distance threshold schedule), or
   - `sng_cluster` (this project's contribution; see
     `library/paper/SNG-method.md`).
5. Counting head: `pred_strategy="total"` (sum over clusters) or `"max"`
   (largest cluster only — the FSC-147 default since each image queries one
   class).

## entrypoints

- single image    :: `python codes/scripts/run_occam.py --image ... --mode single`
- omnicount slice :: `python codes/scripts/eval_omnicount.py --coco-json ... --image-dir ...`
- FSC-147 full    :: `python codes/eval/eval_fsc147_full.py --mode single --splits val test`

See `codes/index.md` for full operations and `results/index.md` for which
configurations have already been swept.

## configurations exposed

- `single` :: OCCAM-S, 224×224 crops, FINCH thresholds (12.0, 9.0, 7.75)
- `multi`  :: OCCAM-M, 500×500 crops, FINCH thresholds (5.0, 4.0, 3.0)
- `cluster_method=finch|sng` with `(sng_epsilon, sng_delta)` for SNG
- `mask_policy=p0..p7` for mask post-filtering ablations
- `pred_strategy=total|max`
