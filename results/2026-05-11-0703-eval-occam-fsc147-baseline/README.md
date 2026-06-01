# 2026-05-11-0703-eval-occam-fsc147-baseline

task: eval
dataset: FSC-147 (val + test, fraction=1/3, seed=42)
model: OCCAM-S (single mode), SAM2 ViT-L AMG, ResNet-50 features, thresholded FINCH
source: ported from FreeCounting/ws_yiyang/OCCAM_experiments_series/origin_simulation
status: done

## primary metrics

| split | n   | MAE   | RMSE   | NAE   | avg_time |
| ----- | --- | ----- | ------ | ----- | -------- |
| val   | 423 | 43.65 | 100.63 | 1.157 | 14.09 s  |
| test  | 396 | 45.47 | 139.43 | 1.198 |  9.60 s  |

## config

- mode :: single (224×224 crops)
- sam2_config :: configs/sam2.1/sam2.1_hiera_l.yaml (sam2.1_hiera_large.pt)
- mask_policy :: p0 (paper baseline area window)
- min_mask_area_ratio / max_mask_area_ratio :: 0.0005 / 0.5
- cluster_method :: finch (thresholds 12.0, 9.0, 7.75)
- pred_strategy :: total
- by_gt_range bucket :: 201+ images dominate error (val MAE 276.96, test 164.52)

## artifacts

- metrics.json :: full metrics including by_gt_range buckets
- per_image_val.json / per_image_test.json :: per-image GT/pred dump
- summary.txt :: human-readable summary
- unusual_imgs/ :: per-bucket failure case images

## notes

This is the **OCCAM single-mode baseline reproduction**. All later ablations
use this as their reference point. FINCH on the FSC-147 val 1/3 subset gives
MAE ≈ 43.65 here, MAE ≈ 32.10 once mask-policy A6 (max_area=0.10) is applied
upstream (see 2026-05-20-...-clustering-sng / 2026-05-21-...-mask-policy-multi).
