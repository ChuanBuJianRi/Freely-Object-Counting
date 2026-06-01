# 2026-05-19-1015-ablation-mask-area-multi

task: ablation
dataset: FSC-147 val (fraction=1/3, seed=42)
model: OCCAM-M (multi mode) + FINCH (pred=total)
axis: mask area ratio (min/max) under policy p0
source: ported from FreeCounting/ws_yiyang/OCCAM_experiments_series/ablation_mask_area_multi
status: done

## sub-runs

| sub_run     | min       | max  | MAE       |
| ----------- | --------- | ---- | --------- |
| M0_baseline | 0.0005    | 0.5  | 41.98     |
| M1_min0001  | 0.0001    | 0.5  | 44.25     |
| M2_min001   | 0.001     | 0.5  | 43.13     |
| M3_min005   | 0.005     | 0.5  | 50.88     |
| M4_min01    | 0.01      | 0.5  | 57.52     |
| M5_max025   | 0.0005    | 0.25 | 41.91     |
| **M6_max010** | **0.0005** | **0.10** | **41.58** ← best |
| M7_tight    | 0.001     | 0.10 | 43.06     |

## conclusion

Same trend as single-mode: M6 (min=5e-4, max=0.10) is the multi-mode
"A6 best" carried into the 2026-05-21 mask-policy ablation.

## artifacts

- run_ablation.sh
- results/M*/{metrics.json, per_image_val.json, run.log}
- run_ablation_resume.log :: resume log after partial failure
