# 2026-05-17-1138-eval-occam-fsc147-multi

task: eval
dataset: FSC-147 val (fraction=1/3, seed=42)
model: OCCAM-M (multi mode), SAM2 ViT-L AMG, ResNet-50 features, thresholded FINCH
source: ported from FreeCounting/ws_yiyang/OCCAM_experiments_series/occam_multi
status: done

## primary metrics

| split | n   | MAE   | RMSE  | NAE   | avg_time |
| ----- | --- | ----- | ----- | ----- | -------- |
| val   | 425 | 41.98 | 95.26 | 1.153 | 5.54 s   |

## config

- mode :: multi (500×500 crops)
- mask_policy :: p0
- min_mask_area_ratio / max_mask_area_ratio :: 0.0005 / 0.5
- cluster_method :: finch (thresholds 5.0, 4.0, 3.0)
- pred_strategy :: total

## artifacts

- results/metrics.json :: full metrics
- results/per_image_val.json :: per-image GT/pred dump
- results/summary.txt :: human-readable summary
- (the nested `results/` subdirectory mirrors the upstream FreeCounting layout 1:1)

## notes

OCCAM-M baseline reproduction; serves as reference for the
2026-05-19 mask-area-multi and 2026-05-21 mask-policy-multi ablations.
Multi-mode is faster (5.5 s/img vs 14 s/img for single) and slightly stronger
on val 1/3 subset.
