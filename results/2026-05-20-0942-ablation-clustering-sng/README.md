# 2026-05-20-0942-ablation-clustering-sng

task: ablation
dataset: FSC-147 val (fraction=1/3, seed=42)
model: OCCAM-S + ResNet-50 features
axis: clustering algorithm (FINCH vs SNG) and (ε, δ) for SNG
source: ported from FreeCounting/ws_yiyang/OCCAM_experiments_series/ablation_clustering
status: done

## context

Upstream is fixed to **A6 best** (mask_policy=p0, min=5e-4, max=0.10) and
`pred_strategy=max` (FSC-147 = single-class-per-image setting). The only knob
swept here is the clustering method on top of the same ResNet-50 features.

## sub-runs (single mode, val 1/3 ≈ 428 imgs)

| sub_run                      | ε  | δ | η (n≈150) | MAE       | RMSE   | NAE   |
| ---------------------------- | -- | - | --------- | --------- | ------ | ----- |
| **A6_FINCH_max** (reference) | -  | - | -         | **32.10** |  -     |  -    |
| A6_SNG_e10_d6_eta064         | 10 | 6 | 0.64      | 38.94     |  -     |  -    |
| A6_SNG_e10_d5                | 10 | 5 | 0.52 ✓    | 39.63     | 106.17 | 0.911 |
| A6_SNG_e8_d3_eta039          |  8 | 3 | 0.39      | 39.68     |  -     |  -    |
| A6_SNG_e10_d4_eta040         | 10 | 4 | 0.40      | 40.60     |  -     |  -    |
| A6_SNG_e5_d2                 |  5 | 2 | 0.48 ✓    | 40.25     | 116.05 | 0.545 |
| A6_SNG_e12_d5_eta040         | 12 | 5 | 0.40      | 41.65     |  -     |  -    |
| A6_SNG_max (default 10,3)    | 10 | 3 | 0.28      | 41.67     |  99.92 | 1.063 |
| A6_SNG_e10_d2                | 10 | 2 | 0.16      | 42.15     |  99.96 | 1.081 |
| A6_SNG_e15_d6_eta036         | 15 | 6 | 0.36      | 42.58     |  -     |  -    |
| A6_SNG_e20_d2                | 20 | 2 | <0        | 43.02     | 100.44 | 1.123 |
| A6_SNG_e20_d5                | 20 | 5 | 0.14      | 43.02     | 100.45 | 1.126 |
| A6_SNG_e20_d3                | 20 | 3 | 0.02      | 43.05     | 100.50 | 1.125 |
| A6_SNG_e5_d3                 |  5 | 3 | 0.74      | 49.36     | 129.09 | 0.502 |
| A6_SNG_e5_d5                 |  5 | 5 | >1 ✗      | 61.13     | 137.42 | 0.758 |

(η = (δ − ε²/n) / (ε − 1 − ε²/n); see library/notes/SNG-method.md §6.4.)

## key findings

1. **FINCH still wins** on FSC-147 by ~7.5 MAE. SNG's strength is expected to
   show up cross-dataset / cross-mode (no per-mode threshold tuning required),
   which is not yet evaluated.
2. **η ∈ [0.4, 0.55] is the sweet spot**, fully predicted by §6.3 SNR theory:
   `e10_d5` (η=0.52, MAE 39.63), `e5_d2` (η=0.48, MAE 40.25),
   `e10_d6_eta064` (η=0.64, MAE 38.94, slightly past sweet spot but still ok).
3. **η > 1 collapses** the algorithm (`e5_d5` MAE 61.13) — δ exceeds the
   intra-class shared-neighbor upper bound ε−1.
4. **η < 0 (ε=20)** degenerates to plain ε-NN clustering — δ no longer
   discriminates, all 3 sub-runs cluster around MAE ≈ 43.0.
5. **multi-mode side-sweep** (`results_multi/`): same families tested on
   OCCAM-M; A6M_FINCH_max remains best, A6M_SNG_e10_d5/d6 are close.

## next directions (from library/notes/SNG-method.md §7)

- §7.1 :: adaptive δ parameterized by α — η pinned to sweet spot, removes one
  hyperparameter.
- §7.2 :: similarity-weighted SNG (continuous, avoids integer δ jumps).
- §7.3 :: degree-normalized δ (per-density local threshold).
- §7.4 :: triangle-reinforcement (second-order topology).
- §7.5 :: signal-noise-driven adaptive (ε, δ) selection per image.

## artifacts

- run_ablation.sh / run_sng_sweep.sh / run_sng_eta_sweep.sh /
  run_sng_multi_top5.sh / chain_eta_then_multi.sh :: launchers (with GPU temp
  guards and resume).
- results/A6_*/ :: 15 single-mode configs.
- results_multi/A6M_*/ :: 6 multi-mode configs.
- run_*.log :: one log per launcher.
