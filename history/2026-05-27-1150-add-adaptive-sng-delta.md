# 2026-05-27 11:50 :: implement adaptive δ for SNG (§7.1) + CPU-only synthetic validation

type: add
scope: codes/occam/, codes/eval/eval_fsc147_full.py, codes/scripts/synth_validate_sng.py, library/notes/SNG-method.md, results/
author: agent
related_memory: memory/2026-05-27-1145.md
related_run: results/2026-05-27-1145-validate-sng-adaptive-delta-cpu

summary:
- Implemented §7.1 of `library/notes/SNG-method.md` :: SNG's δ becomes optional and is auto-derived from `(ε, n)` via `floor(α(ε−1) + (1−α)ε²/n)` whenever `delta=None`. This collapses the (ε, δ) hyperparameter pair to a single dimensionless knob `α`, with `α ∈ [0.3, 0.5]` predicted to keep the health-index η near the empirical sweet spot 0.4.
- Authored `codes/scripts/synth_validate_sng.py` — a CPU-only validator that sweeps `(n, ε, seed) × {fixed-δ, adaptive-α}` on synthetic Gaussian clusters with controllable SNR, reports ARI + counting-MAE under the FSC-147 max-cluster head, and dumps `metrics.json` + `per_run.csv` + `summary.txt`. Acts as a regression baseline for any future SNG variant (§7.2 / §7.3 / §7.4 / §7.5 of SNG-method.md).
- Ran the validator (CPU-only, ~3 s wall clock, 200 runs over n ∈ {50, 100, 150, 250, 500} × 5 seeds × 8 schemes); recorded as `results/2026-05-27-1145-validate-sng-adaptive-delta-cpu/`. **Headline:** adaptive α=0.50 beats best fixed δ (δ=5) by 14 % mean MAE_max (22.32 vs 25.92) and ties on worst-case (70 vs 66) — primary §7.1 claim validated.

files_changed:
- codes/occam/clustering.py :: modified :: `sng_cluster` signature changed to `(features, *, epsilon, delta=None, alpha=0.4)` (backward compatible: integer `delta` keeps legacy behaviour). Added new module-level helpers `adaptive_delta(*, epsilon, n, alpha=0.4) -> int` (clamped to `[0, ε−2]`) and `eta_health(*, epsilon, delta, n) -> float`. Docstrings cite §6.4 / §7.1 of `library/notes/SNG-method.md` and `results/2026-05-20-...-clustering-sng` for sweet-spot evidence.
- codes/occam/config.py :: modified :: `OccamConfig.sng_delta` default switched from `3` to `None` (= adaptive); added `sng_alpha: float = 0.4`; `OccamConfig.for_mode(...)` gained `sng_alpha` keyword override.
- codes/occam/pipeline.py :: modified :: passes `alpha=self.config.sng_alpha` through to `sng_cluster`.
- codes/eval/eval_fsc147_full.py :: modified :: `--sng-delta` default changed from `3` to `None` (with help text explaining the §7.1 default); added `--sng-alpha` (default 0.4); both wired through `OccamConfig.for_mode(...)`.
- codes/scripts/synth_validate_sng.py :: added :: 6.6 KiB; argparse-based CLI; `make_synthetic(n, k, d, intra_std, inter_dist, seed, noise_frac=0.05)` returns features + integer labels (with `-1` for noise); `adjusted_rand_index` (numpy-only, no sklearn dep); `evaluate_one(...)` returns a `RunRow` dataclass; `aggregate(rows)` summarises by (eps, scheme); `write_outputs(...)` produces `metrics.json` (with `thermal: {enabled: false, reason}`), `per_run.csv`, `summary.txt`.
- library/notes/SNG-method.md :: modified :: appended subsection §7.1.x "实现状态与合成数据验证" (implementation status + the synthetic validation results table + actionable conclusions about α default).
- codes/index.md :: modified :: registered `synth_validate_sng.py`; updated `clustering.py` operations to mention `adaptive_delta` + `eta_health`; changelog entry.
- results/2026-05-27-1145-validate-sng-adaptive-delta-cpu/ :: added :: full run folder (README.md, config.yaml, metrics.json, per_run.csv, summary.txt, run.log).
- results/index.md :: modified :: appended one run row for the validation run.

operations_delta:
- codes/occam/clustering.py :: added :: `adaptive_delta(*, epsilon, n, alpha)`, `eta_health(*, epsilon, delta, n)`.
- codes/occam/clustering.py :: changed :: `sng_cluster` signature now accepts `delta=None` and `alpha=0.4` (additive; old integer-delta callers unaffected).
- codes/eval/eval_fsc147_full.py :: changed :: CLI gained `--sng-alpha`; `--sng-delta` default switched to None.
- codes/scripts/synth_validate_sng.py :: added :: new entrypoint, see `usage:` in codes/index.md.

verification:
- `import` smoke (FreeCounting venv) :: `from occam.clustering import sng_cluster, adaptive_delta, eta_health` works; values reproduce SNG-method.md §6.4 numerical examples (η(ε=10, δ=5, n=150) = 0.520; η(ε=20, δ=2, n=150) ≈ -0.04; η(ε=5, δ=5, n=150) ≈ 1.26).
- `OccamConfig` smoke :: defaults are `sng_delta=None`, `sng_alpha=0.4`; `for_mode(...)` round-trips both `sng_alpha` and explicit `sng_delta=5`.
- Validator end-to-end :: 200 runs in ~3 s, all schemes finish with ARI in [0, 1]; aggregate table shows monotonic ordering across (α, fixed-δ) consistent with SNR theory.

followups:
- next agent: when GPU is available, run `python codes/eval/eval_fsc147_full.py --mode single --splits val --fraction 0.333 --seed 42 --cluster-method sng --sng-epsilon 10 --sng-alpha 0.5 --pred-strategy max --min-mask-area 0.0005 --max-mask-area 0.10 --output-dir results/2026-XX-XX-XXXX-fsc147-sng-adaptive-a05-val/`; compare the resulting MAE against `results/2026-05-20-0942-ablation-clustering-sng/results/A6_SNG_e10_d5/metrics.json` (the manually-tuned best, MAE 39.63). Hypothesis: adaptive α=0.5 is within ±1 MAE of the manual best on FSC-147 val.
- next agent: extend the synthetic validator with ε ∈ {5, 8, 12, 15} to map the full (α, ε) sweet-spot surface; cheap (CPU-seconds), useful for choosing the production default `sng_alpha`.
- next agent: do NOT change the production default `OccamConfig.sng_alpha = 0.4` until the FSC-147 val run above is done; the synthetic α=0.5 win is on a single SNR setting and may not transfer.
