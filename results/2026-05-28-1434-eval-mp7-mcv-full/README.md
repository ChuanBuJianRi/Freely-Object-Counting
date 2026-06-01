# 2026-05-28-1434-eval-mp7-mcv-full

task: eval
dataset: FSC-147 val (fraction=1/3, seed=42, n=425)
model: OCCAM-M + min=5e-4/max=0.10 + mask_policy p7 + FINCH + **pred=mode_cluster_vote (MCV) v1, no guard**
status: done
related_history: history/2026-05-27-1406-add-mcv-pred-head.md
followup_run: results/2026-05-28-1538-eval-mp7-mcv-guard/  (MCV v2 with anchor-size guard)

## headline result :: NEGATIVE (MCV v1)

Overall MAE 30.71 → **32.00** (+1.29 vs MP7 + max baseline). RMSE
93.90 → **89.96** (−3.94). MCV v1 trades off: improves the heavy-count
buckets it was designed for, but regresses the small-count buckets it
was claimed not to touch.

| bucket   | n   | MP7 + max | **MP7 + MCV v1** | Δ        | designed effect           |
|----------|----:|----------:|----------------:|---------:|---------------------------|
| 1-10     |  46 |   6.50    | **16.67**       |  +10.17  | F1 mode contamination     |
| 11-50    | 251 |   7.39    | **14.92**       |   +7.53  | F1 mode contamination     |
| 51-200   | 105 |  35.24    | **25.08**       |  −10.16  | designed gain ✓           |
| 201+     |  23 | 312.96    | **280.65**      |  −32.31  | designed gain ✓           |
| **all**  | 425 |  **30.71**|  **32.00**      |   +1.29  | net regression on val 1/3 |

(2 images failed with CUDA OOM and are excluded from MAE; n shown as 425.)

## diagnosis (post-mortem on `per_image_val.json::trace`)

Among the 137 images in 11-50 where MCV is *worse* than max, the mean
over-count is **+21.7 counts** with mode-set size 3-6 and anchor cluster
size median ≈ 21 (90th percentile ≈ 42). This is exactly the F1 failure
mode predicted in `library/notes/MCV-method.md` §6: when the largest
non-singleton cluster is itself a small same-scale background cluster
(e.g. an evenly-spaced pattern), MAD-based mode membership promotes
neighbouring same-scale background clusters into the mode-set, and the
sum over-counts.

A guard-threshold sweep on the saved trace shows that gating on **anchor
cluster size ≥ A** removes the regression. Plateau A ∈ [30, 40] gives
overall MAE ≈ 28.9-29.5; the conservative pick **A=30** simulates to
**29.91**, beating both v1 (32.00) and the max baseline (30.71). This
finding directly motivates the followup run with `--mcv-min-anchor-size 30`.

## what's saved here

- `metrics.json` — final numbers (incl. `thermal: peak=63°C, no cooldown events`)
- `summary.txt` — pretty-printed bucket table
- `per_image_val.json` — every image's `pred / gt / ae / trace`. The
  `trace.cluster_sizes / cluster_log_area / anchor_index /
  mode_member_indices / sigma_log` fields are the input to the offline
  guard sweep above and to any future prediction-head experiment that
  wants to replay results without re-running SAM2/ResNet.
- `run.log` — full stdout (incl. 2 OOM failures: 1936.jpg, 1956.jpg).
- `config.yaml`, `run.sh` — reproducible config.

## reproduction

```bash
bash results/2026-05-28-1434-eval-mp7-mcv-full/run.sh
```
