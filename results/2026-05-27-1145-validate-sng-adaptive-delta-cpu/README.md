# 2026-05-27-1145-validate-sng-adaptive-delta-cpu

task: ablation
dataset: synthetic Gaussians (CPU-only; no SAM2 / GPU required)
model: SNG (`codes/occam/clustering.sng_cluster`) — fixed vs adaptive δ
source: codes/scripts/synth_validate_sng.py (newly authored 2026-05-27)
status: done

## thermal note

This run is **CPU-only**. `metrics.json::thermal.enabled = false`, with
`reason: "CPU-only synthetic validation; no GPU access required."` This is
the documented CPU-only exception to the project-wide GPU thermal guard
policy in `results/index.md::GPU_THERMAL_POLICY`.

## hypothesis (§7.1 of library/notes/SNG-method.md)

The adaptive rule

```
δ* = floor(α(ε−1) + (1−α)ε²/n)
```

should keep clustering quality within a small constant factor of the best
fixed-δ baseline across a range of `n`, *without* needing to retune δ per
dataset. If true, the (ε, δ) pair collapses to a single dimensionless knob α.

## protocol

- 5 sizes :: `n ∈ {50, 100, 150, 250, 500}` (FSC-147 candidate-mask range).
- 5 seeds per (n, ε).
- ε = 10 (best single-mode SNG ε on FSC-147).
- 5 fixed baselines :: δ ∈ {1, 2, 3, 5, 7}.
- 3 adaptive blends :: α ∈ {0.3, 0.4, 0.5}.
- synthetic generator :: 3 isotropic Gaussian clusters in d=64,
  intra-std = 0.8, inter-distance = 4.0, plus 5 % noise points
  (label = -1, std = inter-distance) — represents OCCAM background masks.
- metrics :: ARI vs ground-truth labels, counting MAE under the FSC-147
  "max cluster size" head, and a few diagnostic columns
  (`δ_used`, `η`, `n_clusters_found`, `elapsed_sec`).
- aggregation :: mean and worst-case across all (n, seed) per scheme.
  200 runs total; full grid in `per_run.csv`.

## headline

| scheme            |   δ values   | ARI mean | ARI worst | MAEmax mean | MAEmax worst |
| ----------------- | ------------ | -------- | --------- | ----------- | ------------ |
| **adaptive α=0.50** | {3, 4}     | **0.691** | 0.062    | **22.32**   | **70**       |
| fixed δ=5         | {5}          | 0.624    | 0.062    | 25.92       | 66           |
| adaptive α=0.40   | {3, 4}       | 0.612    | 0.000    | 34.40       | 132          |
| fixed δ=7         | {7}          | 0.457    | 0.078    | 32.72       | 117          |
| fixed δ=3         | {3}          | 0.533    | 0.000    | 41.36       | 132          |
| adaptive α=0.30   | {2, 3}       | 0.464    | 0.000    | 71.00       | 167          |
| fixed δ=2         | {2}          | 0.251    | 0.000    | 84.76       | 167          |
| fixed δ=1         | {1}          | 0.003    | 0.000    | 142.00      | 338          |

## conclusions

1. **adaptive α=0.50 beats every fixed-δ baseline on mean MAE_max
   (22.32 vs best-fixed 25.92, 14 % improvement) at essentially identical
   worst-case (70 vs 66).** This is the core §7.1 claim, validated.
2. **adaptive α=0.40 (current implementation default) is mid-pack** —
   safer than δ=3, worse than δ=5 on this synthetic setting. On FSC-147
   real features (lower noise, n≈150 fixed) α=0.4 produced η=0.4 and a
   reasonable MAE; on a wider n range, α=0.5 looks better.
3. **Both fixed-δ extremes fail differently** at the ends:
   - δ=1 / δ=2 :: η < 0 ⇒ noise-floor regime ⇒ no pruning ⇒ MAE explodes.
   - δ=7 :: η ≈ 0.7–1.0 ⇒ over-pruning at small n ⇒ ARI collapses.
   - Adaptive automatically slides δ between {3, 4} as n grows, avoiding
     both failure modes.
4. **n=500 sub-results** (where η is smallest for any fixed δ) most
   clearly favour adaptive — see `per_run.csv` rows for n=500.
5. **CPU runtime** :: full 200-run sweep finished in ~3 s on a single core
   (no GPU). This makes the validator a useful regression guard for any
   future SNG variant (§7.2 / §7.3 / §7.4 / §7.5 of SNG-method.md).

## actionable next steps

- **DO NOT change the default `OccamConfig.sng_alpha = 0.4`** without a
  real-FSC-147 run :: the synthetic α=0.5 win is on a single SNR setting;
  on real ResNet-50 features the optimum α may shift. The default should
  be confirmed by an FSC-147 val/test sweep over α ∈ {0.3, 0.4, 0.5}.
- **Run FSC-147 with α=0.5** (next GPU run): `python codes/eval/eval_fsc147_full.py
  --mode single --splits val --fraction 0.333 --seed 42 --cluster-method sng
  --sng-epsilon 10 --sng-alpha 0.5 --pred-strategy max
  --min-mask-area 0.0005 --max-mask-area 0.10
  --output-dir results/<NEW_RUN_ID>/`. Compare against the synced
  2026-05-20-0942 SNG sweep.
- **Extend the validator** :: add ε ∈ {5, 8, 12, 15} and rerun to map the
  full (α, ε) sweet-spot surface. Cheap (CPU-seconds).

## artifacts

- metrics.json :: full aggregate + best-fixed comparison + thermal block.
- per_run.csv  :: 200 rows, one per (n, ε, seed, scheme); use for plots.
- summary.txt  :: human-readable report mirroring the headline above.
- run.log      :: full stdout of the run (saved via `tee`).
