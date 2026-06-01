# 2026-05-27-1529-eval-mp7-mcv-smoke

task: smoke
dataset: FSC-147 val (fraction=1/3, seed=42, --limit 20 → n=19 valid)
model: OCCAM-M + AMG no-filter (P7) + min=5e-4/max=0.10 + FINCH + **pred=mode_cluster_vote**
purpose: end-to-end smoke test of the new MCV prediction head against the same
         AMG → ResNet-50 → FINCH upstream as MP7. Verifies (a) MCV runs on real
         FSC-147 images, (b) MCV ≠ trivially equal to max, (c) the per_image
         JSON now stores cluster traces correctly.
status: done
related_history: history/2026-05-27-1406-add-mcv-pred-head.md
related_note: library/notes/MCV-method.md

## headline (this 19-image subset only)

|              | MP7 (max) | MP7 + MCV | delta            |
| ------------ | --------- | --------- | ---------------- |
| MAE          | 46.89     | **36.00** | **−10.89 (−23%)** |
| MSE          | 17897     | **11577** | **−6320 (−35%)** |
| RMSE         | 133.8     | 107.6     | −26.2            |
| time / image | 11.4 s    | 5.7 s     | (smoke had warm cache) |

(MP7 numbers above are recomputed on the same 19 images by reading
`results/2026-05-21-1328-ablation-mask-policy-multi/results/MP7_no_filter/per_image_val.json`,
not the full-set 30.71 published headline.)

## per-image breakdown (sorted by GT)

| image      | GT  | MP7 pred | MP7 AE | MCV pred | MCV AE | delta_AE |
| ---------- | --- | -------- | ------ | -------- | ------ | -------- |
| 1955.jpg   |  13 |  13      |  0     |  13      |  0     |   0      |
| 1938.jpg   |   9 |   7      |  2     |  20      | 11     |  +9      |
| 1947.jpg   |  12 |  10      |  2     |  40      | 28     | +26 ⚠    |
| 1897.jpg   |  12 |   8      |  4     |  15      |  3     |  −1      |
| 1928.jpg   |  13 |   9      |  4     |  33      | 20     | +16 ⚠    |
| 1901.jpg   |  17 |   9      |  8     |  22      |  5     |  −3      |
| 1918.jpg   |  18 |  13      |  5     |  34      | 16     | +11 ⚠    |
| 1916.jpg   |  20 |  21      |  1     |  46      | 26     | +25 ⚠    |
| 1907.jpg   |  28 |  15      | 13     |  29      |  1     | −12 ✓    |
| 1946.jpg   |  40 |  33      |  7     |  35      |  5     |  −2      |
| 1912.jpg   |  46 |  19      | 27     |  52      |  6     | −21 ✓    |
| 1913.jpg   |  50 |  44      |  6     |  77      | 27     | +21 ⚠    |
| 1931.jpg   |  54 |  34      | 20     |  62      |  8     | −12 ✓    |
| 1927.jpg   |  68 |  32      | 36     |  39      | 29     |  −7      |
| 1939.jpg   |  69 |  19      | 50     |  69      |  0     | −50 ✓✓   |
| 1906.jpg   |  70 |  44      | 26     |  71      |  1     | −25 ✓✓   |
| 1920.jpg   | 138 |  86      | 52     | 113      | 25     | −27 ✓    |
| 1934.jpg   | 171 | 116      | 55     | 162      |  9     | −46 ✓✓   |
| 1915.jpg   | 684 | 111      | 573    | 220      | 464    | −109 ✓✓  |

✓ = MCV better, ⚠ = MCV worse, ✓✓ = MCV >> better.

## findings

1. **MCV is a major net win on this subset**: MAE −23%, MSE −35%. The big
   wins come exactly where the design predicted: GT in [50, 700], where
   `pred=max` was missing 30–80% of the count because the query class was
   spread across 2–4 same-scale clusters. MCV fuses them.
2. **The 684-image (1915.jpg)** went from PRED=111 to PRED=220, recovering
   109 objects in one image. Trace: 9 clusters with sizes
   `[1,1,1,1,111,104,5,5, ...]`, anchor=cluster 4 (size 111),
   sigma_log=0.087, mode-set has 3 members. Still under-counts by 464 (the
   image has 684 ducks; AMG simply does not produce enough proposals).
   This is where DAMS (B in the roadmap) would help.
3. **MCV regressions are concentrated in GT ∈ [12, 20]** (1947/1928/1918/1916).
   These are images where MCV's MAD fence is too wide because the cluster
   set has a long tail of similar-scale background fragments. Symptoms:
   `sigma_log` >= 0.7 in the bad cases; in the wins it is typically
   ≤ 0.4. **Action:** consider clamping the fence (e.g. `min(k * sigma, 0.4)`)
   in a follow-up; do NOT do this in the first full run so the baseline
   MCV behaviour is recorded as-is.
4. **Trace integrity confirmed**: `cluster_sizes`, `cluster_log_area`,
   `anchor_index`, `mode_member_indices`, `sigma_log` all present and
   consistent in `per_image_val.json` for all 19 images. Future
   prediction-head experiments can replay offline.

## artifacts

- run.sh
- run.log / run.out
- per_image_val.json (with `trace` field)
- metrics.json (incl. `thermal` block from GpuGuard)
- summary.txt

## next step

Launch the **full** val (fraction=1/3, seed=42, n≈425) MP7+MCV run in
`results/2026-05-27-1529-eval-mp7-mcv/` so the headline numbers are
directly comparable to MP7 (MAE 30.71 / MSE 8817 on the same 425 images).
