# 2026-05-21-1328-ablation-mask-policy-multi

task: ablation
dataset: FSC-147 val (fraction=1/3, seed=42)
model: OCCAM-M + FINCH (pred=max)
axis: mask post-filtering policy P0..P7 on top of M6 area window (min=5e-4, max=0.10)
source: ported from FreeCounting/ws_yiyang/OCCAM_experiments_series/ablation_mask_policy_multi
status: done

## sub-runs

| sub_run            | policy | extra params                | MAE       |
| ------------------ | ------ | --------------------------- | --------- |
| MP0_baseline       | p0     | area_window only            | 31.86     |
| MP1_score090       | p1     | score_thresh=0.90           | 41.57     |
| MP2_topk100        | p2     | topk=100                    | 40.08     |
| MP3_iqr15          | p3     | iqr_k=1.5 (adaptive)        | 30.99     |
| MP4_otsu           | p4     | log-area Otsu               | 42.12     |
| MP5_score090_area  | p5     | score 0.90 + area window    | 41.59     |
| MP6_score_nms      | p6     | score-priority NMS          | 30.80     |
| **MP7_no_filter**  | **p7** | (no post-filter, only AMG)  | **30.71** ← best |

## key findings

1. **Doing less is better**: P7 (no project-side filtering, AMG output only)
   gives the lowest MAE on multi-mode FSC-147. AMG's internal NMS + stability
   filter is already strong enough that downstream filtering removes signal.
2. P6 (score-priority NMS) and P3 (adaptive IQR) are nearly tied with P7,
   suggesting any **scale-free** filter behaves well; absolute thresholds
   (P1/P5) hurt.
3. MP0 baseline (area window) is competitive (31.86), but the absolute
   improvement over P7 is small (-1.15 MAE) and likely brittle to image-area
   distribution shift.

## comparison to upstream baselines

- OCCAM-M baseline (P0 + min=5e-4, max=0.5, FINCH, pred=total): MAE 41.98
- M6 (P0 + min=5e-4, max=0.10, FINCH, pred=total):              MAE 41.58
- This run (P7 + same area, FINCH, pred=**max**):               MAE 30.71

The pred_strategy switch (`total → max`) accounts for most of the gap; mask
policy on top of it is a secondary +/− 1 MAE effect.

## artifacts

- run_sweep.sh / _smoke_test.sh
- results/MP*/{metrics.json, per_image_val.json, run.log}
- run_sweep.log
