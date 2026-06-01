# 2026-05-27 11:45 :: harden GPU thermal-safety contract + finish FreeCounting sync

type: refactor
scope: codes/eval/, codes/index.md, AGENTREAD.md, results/index.md, results/2026-05-17-1138-eval-occam-fsc147-multi/
author: agent
related_memory: memory/2026-05-27-1145.md
related_run: results/2026-05-27-1145-validate-sng-adaptive-delta-cpu (covers the new contract from the user side)

summary:
- User asked to (a) keep the FreeCounting sync byte-aligned, (b) make the GPU thermal guard a first-class project policy, and (c) continue the autonomous research from where it was interrupted.
- Extracted the inline `_gpu_temp` / `_throttle_if_hot` from `eval_fsc147_full.py` into a shared module `codes/eval/_gpu_safety.py` exposing `GpuGuard`, `add_cli_args(parser)`, `guard_from_args(args)`. The new module auto-detects `nvidia-smi` (covers WSL2 + native Linux + WSL fallback paths), records peak temp + cooldown stats, and writes a `thermal: {...}` block into `metrics.json` for every run.
- Mandated the guard in 3 places: AGENTREAD.md §6 `operating_constraints`, AGENTREAD.md §4 `results/index.md` row, and a new `GPU_THERMAL_POLICY` section in `results/index.md`. CPU-only runs explicitly exempt themselves via `--gpu-guard-off` + a justification in their README.
- Recovered the upstream `occam_multi/results/` nested directory under `results/2026-05-17-1138-eval-occam-fsc147-multi/results/` so the synced run is now byte-for-byte the same as `FreeCounting/ws_yiyang/OCCAM_experiments_series/occam_multi/` (modulo our added README + config.yaml).

files_changed:
- codes/eval/_gpu_safety.py :: added :: 7.5 KiB; `GpuGuard` dataclass with `maybe_throttle(idx, total)`, `to_dict()` (for metrics.json), `add_cli_args(parser)`, `guard_from_args(args)`; auto-detects `nvidia-smi` across PATH + 4 absolute-path fallbacks.
- codes/eval/eval_fsc147_full.py :: modified :: removed local `_gpu_temp` / `_throttle_if_hot` / `_GPU_TEMP_LIMIT` / `_GPU_COOLDOWN_SEC`; imports from `_gpu_safety`; `run_split` accepts `guard: GpuGuard | None`; `metrics.json` now wraps `all_metrics` + adds `thermal: guard.to_dict()`; CLI gains `--gpu-temp-limit`, `--gpu-cooldown-sec`, `--gpu-hysteresis`, `--gpu-check-every`, `--gpu-index`, `--gpu-guard-off` via `add_gpu_cli_args(p)`. Behaviour identical when defaults are used (78 C, 30 s, 5 C hysteresis, every 5 imgs).
- AGENTREAD.md :: modified :: §4 row for `results/index.md` mentions mandatory `thermal:` block; §6 `operating_constraints` gains a `gpu thermal safety ::` bullet with default knobs and CPU-only exception rule.
- codes/index.md :: modified :: tree gains `eval/_gpu_safety.py` + `scripts/synth_validate_sng.py`; eval/ section gets a `shared_modules:` subsection; occam/ tree comment updated to mention adaptive δ + η; changelog entries appended.
- results/index.md :: modified :: appended `GPU_THERMAL_POLICY` block; appended one row for the new CPU-only run; changelog entry appended.
- results/2026-05-17-1138-eval-occam-fsc147-multi/results/ :: added :: restored upstream nested structure (cp from FreeCounting/.../occam_multi/results/); README + config.yaml rewritten so artifact paths match.
- results/2026-05-17-1138-eval-occam-fsc147-multi/{metrics.json,per_image_val.json,summary.txt} :: deleted :: removed flattened duplicates; nested copy is now the single source of truth for this run's artifacts.

operations_delta:
- codes/eval/_gpu_safety.py :: added :: `GpuGuard.maybe_throttle(idx, total)` polls every `check_every` calls; `query_temp(nvidia_smi, gpu_index)`; `add_cli_args(parser)`; `guard_from_args(args)`.
- codes/eval/eval_fsc147_full.py :: changed :: `metrics.json` schema gained `"thermal"` key; `run_split(...)` gained `guard: GpuGuard | None` keyword.

verification:
- `python -c "import ast; ast.parse(open('codes/eval/eval_fsc147_full.py').read())"` :: SYNTAX OK.
- `add_cli_args` + `guard_from_args` smoke run on the dev box :: detected `/usr/lib/wsl/lib/nvidia-smi`, returned `start_temp=47C`, `to_dict()` contains all 9 expected keys.
- `diff -rq FreeCounting/.../OCCAM_experiments_series/<src> GOC/results/<run_id>/` for all 6 synced runs :: only differences are README.md / config.yaml that we authored. occam_multi nested layout now matches.

followups:
- next agent: a future smoke run on the GPU machine should sanity-check that `metrics.json::thermal.peak_temp_c > 0` and that the run does NOT have `cooldown_events > 0` for short evaluations (FSC-147 val 1/3 ≈ 425 images at 14 s/img on OCCAM-S means ~100 min wall-clock; at 78 C limit the guard usually fires 0–2 times during summer / 0 in winter).
- if a launcher script wraps the evaluator, it should pass `--gpu-temp-limit` / `--gpu-cooldown-sec` once at the top, not hard-code them per-config.
