---
name: counting-eval
description: Standardized evaluation pipeline for object-counting models. Use this skill whenever the user gives a checkpoint/prediction file and a dataset and wants to compute counting metrics — ALWAYS use this when MAE / RMSE / NAE on FSC-147, CARPK, ShanghaiTech (Part A/B), or any object-counting benchmark is mentioned, even if the user only says "evaluate", "score", "test on", or "run metrics on" a counting model. Produces a unified output (metrics.json + per-category table + GT-vs-Pred scatter plot) that matches the project's `results/<run_id>/` layout.
---

# counting-eval

Standardized, reproducible evaluation for class-agnostic / open-vocabulary object counting.

## When to invoke
- user provides a checkpoint, prediction file, or model module + a counting dataset, and wants metrics.
- user mentions FSC-147, CARPK, ShanghaiTech (Part_A / Part_B), TRANCOS, or any counting benchmark.
- user says "evaluate", "score", "compute MAE/RMSE/NAE", "test on counting", "run eval".

## Inputs (require explicitly; ask the user if missing)
- `predictions` :: a JSON / CSV / pickle mapping `image_id -> predicted_count` (float). If only a checkpoint is given, the agent must first run inference and persist this file.
- `ground_truth` :: dataset-specific GT count per image (load via the dataset's standard split file).
- `dataset_name` :: one of `fsc147`, `carpk`, `shtech_a`, `shtech_b`, `trancos`, `custom`.
- `split` :: `val` | `test` | `all`.
- `run_id` :: matches `results/<run_id>/` (per `results/index.md`).
- optional :: `per_category_field` (for FSC-147, the class label per image) for per-category breakdown.

## Required metrics
- `MAE`  = mean(|pred - gt|)
- `RMSE` = sqrt(mean((pred - gt)^2))
- `NAE`  = mean(|pred - gt| / max(gt, 1))   # natural absolute error
- per-category MAE/RMSE when `per_category_field` is provided.
- always report `n_samples`, `gt_total`, `pred_total`, `bias = mean(pred - gt)`.

## Required outputs (write to `results/<run_id>/eval/`)
1. `metrics.json` :: flat object, e.g.
   ```json
   {"dataset":"fsc147","split":"val","n":1286,"MAE":18.21,"RMSE":34.5,"NAE":0.41,"bias":-2.1}
   ```
2. `per_category.csv` :: columns `category,n,MAE,RMSE,NAE` (skip if no per-category field).
3. `scatter.pdf` and `scatter.png` :: GT vs Pred scatter, log-log when range > 100×, identity line dashed, axis labels `GT count` / `Pred count`, DPI ≥ 300.
4. `report.md` :: 1-page summary with the metrics table, top-5 worst predictions, per-category bar chart embedded.
5. update `results/index.md` :: set `primary_metric=MAE=<value>`, `status=done`.

## Algorithm
1. load predictions and GT, align on `image_id`; assert no missing keys (fail loudly if mismatch).
2. compute the four scalar metrics above.
3. if per-category field present, group and compute per-category metrics; sort categories by MAE descending.
4. plot scatter (matplotlib, no seaborn defaults; serif font; equal-aspect; identity line `y=x`).
5. write all outputs; never overwrite an existing `eval/` folder — append `eval-v2/`, `eval-v3/`, ... if rerunning.
6. on completion, append a row to `results/index.md` per `UPDATE_RULES`, and log a `history/` entry of type `add` with `operations_delta` describing the new eval artifact.

## Style constraints
- numerical precision :: 2 decimals in tables, 4 decimals in `metrics.json`.
- plotting :: serif font, color-blind safe (`#377eb8`, `#e41a1c`); no chart-junk; legend top-left.
- absolutely no silent fallbacks — missing GT or NaN must raise.

## Failure modes to guard
- mismatched image IDs between preds and GT.
- predictions in different scale than GT (e.g. density-map sums vs. integer counts) — assert and ask user.
- categories with `n<5` :: still report but flag as low-support.
