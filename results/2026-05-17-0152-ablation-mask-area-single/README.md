# 2026-05-17-0152-ablation-mask-area-single

task: ablation
dataset: FSC-147 val (fraction=1/3, seed=42)
model: OCCAM-S + thresholded FINCH (cluster_method=finch, pred=total)
axis: mask area ratio (min/max) under policy p0
source: ported from FreeCounting/ws_yiyang/OCCAM_experiments_series/ablation_mask_area
status: done

## sub-runs (sweep over `min_mask_area_ratio` / `max_mask_area_ratio`)

| sub_run     | min       | max  | MAE       |
| ----------- | --------- | ---- | --------- |
| A0_baseline | 0.0005    | 0.5  | 43.34     |
| A1_min0001  | 0.0001    | 0.5  | 45.69     |
| A2_min001   | 0.001     | 0.5  | 43.11     |
| A3_min005   | 0.005     | 0.5  | 50.88     |
| A4_min01    | 0.01      | 0.5  | 57.52     |
| A5_max025   | 0.0005    | 0.25 | 43.27     |
| **A6_max010** | **0.0005** | **0.10** | **42.94** ← best |
| A7_tight    | 0.001     | 0.10 | 43.04     |

## conclusion

- Tightening the **upper** bound to 0.10 (A6) helps modestly (-0.4 MAE).
- Raising the **lower** bound past 0.001 hurts dramatically because real
  small-object masks get filtered out (A3/A4: +7 to +14 MAE).
- A6 (`min=5e-4, max=0.10`) is adopted as the **upstream "A6 best"** in all
  later clustering ablations.

## artifacts

- run_ablation.sh :: original sweep launcher
- results/A*/metrics.json :: full per-config metrics
- results/A*/per_image_val.json :: per-image dump
- results/A*/run.log :: launcher log per config
