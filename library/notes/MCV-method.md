### MCV: Mode-Cluster-Vote prediction head for OCCAM

PURPOSE: documents the Mode-Cluster-Vote (MCV) prediction head proposed as a drop-in replacement for OCCAM's `pred_strategy in {total, max}`, designed to (a) keep the pipeline strictly training-free / parameter-light and (b) reduce MSE on heavy-count images (the FSC-147 201+ bucket where current OCCAM-MP7 has MAE 312.96 / RMSE 387.10).

## 1. motivation: why MP7 fails on dense images

The current best baseline in `results/` is

- run :: 2026-05-21-1328-ablation-mask-policy-multi
- config :: OCCAM-M + AMG no-filter (P7) + FINCH + pred=max
- FSC-147 val (fraction=1/3, seed=42, n=425):
  - MAE  = 30.71
  - MSE  = 8816.73
  - RMSE = 93.90
  - by GT bucket:
    - 1-10   :: n=46  MAE= 6.50  RMSE= 14.94
    - 11-50  :: n=251 MAE= 7.39  RMSE= 12.04
    - 51-200 :: n=105 MAE=35.24  RMSE= 49.18
    - 201+   :: n=23  MAE=312.96 RMSE=387.10

23 images (5.4% of the eval set) carry essentially all of the RMSE budget. The structural reason is the prediction head:

- `pred=total` sums every cluster size, so background fragments and partial-object masks are all counted, blowing up the count on dense scenes.
- `pred=max` takes the largest cluster size, which is fine when the query class fits into one cluster (1-50 buckets), but on dense scenes the query class is fragmented into multiple same-sized clusters by both (i) AMG over-segmentation at small scale and (ii) FINCH/SNG splitting feature-space sub-modes. Taking only the largest cluster then under-counts severely.

`max` is therefore correct on images with one dominant cluster but is the wrong reduction on images where the query class is spread over several roughly equal-sized clusters of the same physical scale.

## 2. design goals

- training-free: no fine-tuning, no learned parameters, no validation-set tuning beyond what already exists in the pipeline.
- parameter-light: introduce zero new tunable hyperparameters; reuse the project's existing `mask_iqr_k = 1.5` (already used by mask-policy P3) when a robust spread multiplier is needed.
- backward compatible: `pred=total` and `pred=max` remain exact special cases; `mode_cluster_vote` is a new opt-in `pred_strategy`.
- fail-safe: on images where MCV cannot find a clear mode, fall back to `max` (never worse than current).

## 3. algorithm

Inputs from `OccamCounter.count(image) -> OccamResult`:

- `clusters: list[Cluster]`, each with `indices: tuple[int, ...]` pointing into `masks`.
- `masks: list[CandidateMask]` with bounding boxes; image height H, width W known from the input.

Procedure:

1. **discard singleton clusters** (size 1). These are almost always isolated background fragments; including them in the mode estimation pollutes the signal. (If only singletons exist, fall through to `max` directly.)

2. **per-cluster representative size**. For each remaining cluster `C`, compute the median of its members' bbox area-ratios:

       r(C) = median_{m in C} ((x1-x0) * (y1-y0)) / (H * W)

   Use bbox area (not mask area) because (a) it is what AMG / OCCAM already exposes and (b) it is more stable to mask-fragmentation noise.

3. **log-area embedding**. Map each cluster to `u(C) = log10(r(C))`. The log space is the right place to compare object sizes: ratio-of-areas, not difference-of-areas, is what makes two objects "the same size".

4. **anchor at the largest cluster**. Define the mode anchor as

       u* = u(C_argmax),  where  C_argmax = arg max_C |C|

   This is the cluster `pred=max` would return. The anchor is fully determined by the upstream pipeline; MCV only decides which OTHER clusters belong to the same physical-size mode as `C_argmax`.

5. **robust spread**. Compute the median absolute deviation of all cluster log-areas around `u*`:

       sigma* = median_{C} |u(C) - u*|

   MAD is the standard robust scale estimator; using `u*` (not the median of all `u(C)`) as the centre is intentional because we are testing membership relative to the anchor, not relative to the population centre.

6. **mode membership**. A cluster is in the mode iff

       |u(C) - u*| <= k * sigma*,  where  k = 1.5 (reused from `mask_iqr_k`)

   `k=1.5` is the IQR fence constant Mask-Policy P3 already uses; we reuse it verbatim so that MCV does not introduce a new hyperparameter. If `sigma* == 0` (only one non-singleton cluster), the membership set degenerates to {C_argmax} and MCV equals `max`.

7. **predict**:

       y_hat = sum_{C in mode-set} |C|

8. **fallback rule**: if step 1 leaves zero non-singleton clusters, use the original `pred=max`. This guarantees MCV never under-performs `max` on the small-count regime where every image has at most one valid cluster.

## 4. invariants and special cases

- **single-cluster images** (most 1-10 bucket): only one non-singleton cluster exists, mode-set = {C_argmax}, MCV equals `max`. **No regression possible on the easy bucket.**
- **two-mode images** (e.g. one row of geese in foreground + one row in background, both belonging to the query class but at different physical scales): MAD will reject the smaller-scale cluster from the mode anchored at the larger one. This is the conservative choice; a future variant can predict both modes and sum them, but only after confirming no regression on val.
- **noise-only images** (everything is singletons): fall back to `max`, which here equals 1 or 0 -- same as the current pipeline.
- **all clusters at one scale, scale matches GT**: mode-set = all non-singletons, sum equals what `total` would produce minus the singletons. This is the case MCV is designed to fix.

## 5. expected effect on each bucket (FSC-147 val 1/3)

| bucket  | n   | current MAE (MP7) | MCV expected behaviour                                   | predicted MAE      |
|---------|-----|-------------------|-----------------------------------------------------------|--------------------|
| 1-10    |  46 |   6.50            | mode-set = {C_argmax}, identical to `max`                 | ~6.5 (no change)   |
| 11-50   | 251 |   7.39            | usually one dominant cluster + small singletons -> ~max   | ~7.4 (no change)   |
| 51-200  | 105 |  35.24            | query class often split into 2-3 same-scale clusters; MCV recovers them | substantial drop, target ~20 |
| 201+    |  23 | 312.96            | query class split into 5-N same-scale clusters; MCV recovers most | large drop, target <=200 |

If 201+ MAE drops to 200 with everything else unchanged, the overall MAE moves from 30.71 to roughly 24.6 and MSE from 8817 to roughly 4500. These numbers are the success target for the run that follows this note.

## 6. failure modes (to monitor in the experiment)

- **F1 -- mode contamination**: a same-scale background cluster (e.g. evenly-spaced ground tiles) gets included in the mode and inflates the count. Mitigation: the anchor is the LARGEST cluster, which biases the spread estimator towards "the cluster the query class lives in"; a contaminating cluster has to be both same-size AND comparable in count. Diagnostic: per-image AE on small-count buckets; if 1-50 MAE goes up, this is happening.
- **F2 -- query class is multi-modal**: e.g. a few large fish in foreground + a school of small fish behind. MCV (anchored at the largest cluster) will pick whichever scale dominates and miss the other. This will show as residual under-count on a small subset of 51-200 images.
- **F3 -- sigma* explosion** when there are exactly 2 clusters on different scales: MAD = |u_small - u*|, k * sigma* = 1.5 * MAD covers the small one, MCV becomes `total`. This is by-design but worth verifying on per-image traces.
- **F4 -- evaluation noise**: we are still evaluating on a 1/3 sample (n=425). Differences <= 1 MAE are within seed noise; the experiment must show a clear-margin improvement (at least 2 MAE) to count as evidence.

## 7. relation to SNG and to FINCH

- MCV is **prediction-side**, not clustering-side. It does not change `cluster_method`, mask filtering, or feature extraction.
- It composes orthogonally with both `cluster_method=finch` and `cluster_method=sng`. The first MCV experiment uses FINCH (the current best clustering on FSC-147) so that the only changing variable is the prediction head.
- If MCV works, the natural follow-up is to compose with SNG-adaptive-delta (`library/notes/SNG-method.md` 7.1) so the whole pipeline is parameter-light: 1 SNG epsilon + 1 SNG alpha + 0 prediction-head parameters.

## 8. cross-folder contracts triggered by this note

- new code unit :: `codes/occam/predict.py` (or extension of `codes/occam/pipeline.py`) implementing `predict_count(result, strategy, *, k=1.5)` -- update `codes/index.md`.
- new history entry :: `history/<YYYY-MM-DD-HHMM>-mcv-prediction-head.md`, append row to `history.md`.
- new run :: `results/<YYYY-MM-DD-HHMM>-eval-mp7-mcv/` mirroring the MP7 config but with `pred_strategy=mode_cluster_vote`; append row to `results/index.md`.

## 9. reproduction command (to be valid after the code change lands)

```bash
python codes/eval/eval_fsc147_full.py \
  --mode multi --splits val --fraction 0.333 --seed 42 \
  --min-mask-area 0.0005 --max-mask-area 0.10 \
  --mask-policy p7 \
  --cluster-method finch \
  --pred-strategy mode_cluster_vote \
  --output-dir results/<run_id>/ \
  --data-dir <FSC147_root>
```

## 10. empirical validation (added 2026-05-28)

The §5 expected effect was tested end-to-end on FSC-147 val (1/3,
seed=42, n=425). Ground truth is more nuanced than §5 predicted.

### 10.1 v1 — MCV as originally specified (§3, no guard)

run :: `results/2026-05-28-1434-eval-mp7-mcv-full`

| bucket   | n   | MP7+max (prior best) | MP7+MCV v1 | Δ (MCV − max)        |
|----------|----:|---------------------:|-----------:|---------------------:|
| 1-10     |  46 |  6.50                | **16.67**  | **+10.17 (regress)** |
| 11-50    | 251 |  7.39                | **14.92**  |  **+7.53 (regress)** |
| 51-200   | 105 | 35.24                | **25.08**  |  −10.16 ✓            |
| 201+     |  23 | 312.96               | **280.65** |  −32.31 ✓            |
| overall  | 425 |  **30.71**           | **32.00**  |  **+1.29 (regress)** |

(RMSE drops 93.90 → 89.96 — MCV v1 trades MAE for RMSE.)

The §5 invariant "single-cluster images on the easy bucket cannot
regress" turned out to be **false in practice**: AMG + FINCH on real
FSC-147 small-count images often yields ≥2 non-singleton clusters, so
the MAD-based mode test fires and **F1 mode contamination** dominates.
Among the 137 v1-regressed images in 11-50, mean over-count is +21.7
counts and the anchor cluster size is median 21 / 90th-pct 42 — i.e.
the F1 archetype where the largest non-singleton is itself a small
same-scale background cluster.

### 10.2 v2 — anchor-size guard (`mcv_min_anchor_size`)

A trace-sweep on v1's saved `per_image_val.json::trace` (no GPU needed)
showed a stable plateau when MCV is gated on the anchor cluster size:

| guard A | overall MAE (offline sim) | n_use_mcv / 425 |
|--------:|--------------------------:|----------------:|
|       0 |  32.00 (= v1)             | 425 |
|      10 |  31.71                    | 380 |
|      20 |  30.32                    | 259 |
|  **30** |  **29.91** (chosen)       | 174 |
|      36 |  28.89 (best)             | 132 |
|      40 |  28.99                    | 117 |
|     ∞   |  30.71 (= max)            |   0 |

Plateau A ∈ [30, 40] gives 28.9-29.5; we chose A=30 conservatively (start
of plateau, simpler integer).

A=30 was confirmed end-to-end on GPU
(`results/2026-05-28-1538-eval-mp7-mcv-guard`), matching the offline
prediction to **0.002 MAE**:

| variant                       | overall MAE | 1-10  | 11-50 | 51-200 | 201+   |
|-------------------------------|------------:|------:|------:|-------:|-------:|
| MP7 + max  (prior best)       |       30.71 |  6.50 |  7.39 |  35.24 | 312.96 |
| MP7 + MCV v1                  |       32.00 | 16.67 | 14.92 |  25.08 | 280.65 |
| **MP7 + MCV + A=30 (v2)**     |   **29.91** | 10.80 | 10.75 |  29.14 | 280.65 |

This is **the first FSC-147 result that beats the prior MP7 + max
baseline** in this repo. But note:

- **The guard introduces a new hyperparameter** `mcv_min_anchor_size`,
  violating MCV's original "zero new hyperparameters" claim. The default
  remains 0 (= v1 behaviour); the improved configuration is reported as
  a separate variant. We do not promote `OccamConfig` to A=30 yet
  because the threshold was tuned on the same val 1/3 split — this must
  be confirmed on FSC-147 test before being made the default.
- The small-bucket regression is **reduced** but not eliminated
  (1-10 + 11-50 combined +3.5 MAE vs max). MCV v3 candidates (cohesion
  gate, top-3-anchor, ε-NN-based anchor selection) are listed in the
  v2 README as next directions.

### 10.3 lessons for any future prediction-head experiment

- **Always save the cluster trace** in per-image JSON. The 0.002 MAE
  agreement between offline simulation and the GPU run shows that any
  prediction-only ablation can be evaluated in seconds (not 40 min) by
  replaying saved traces. This is now standard in
  `codes/occam/predict.py::PredictTrace.to_dict`.
- **Failure-mode predictions in §6 must be tested on real data, not
  validated on synthetic fixtures alone.** §5 invariants that depended
  on "only one non-singleton cluster" did not hold; F1 was the dominant
  failure on the buckets §5 predicted "no regression possible".
- **Honesty about hyperparameter count matters.** A method with a guard
  is a method with one more knob; the trace-driven sweep makes the knob
  cheap to tune but does not erase its existence.
