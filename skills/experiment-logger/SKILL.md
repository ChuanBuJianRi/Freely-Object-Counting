---
name: experiment-logger
description: Enforce a uniform experiment record at the start of EVERY training/eval run. Use this skill whenever the user launches, queues, or describes starting any training, fine-tuning, or evaluation job — ALWAYS invoke before the first training command runs, even if the user only says "start training", "kick off a run", "let's try this config". Generates `experiment.md` with git commit hash, config diff, seed, hardware, timing, and final metrics, and enforces the project's `exp_YYYYMMDD_<dataset>_<method>_<tag>` naming convention.
---

# experiment-logger

Make every experiment self-documenting so 3-month-later "which run was the best one?" never becomes archaeology.

## When to invoke
- user is about to launch a training / fine-tuning / large eval job.
- user says "start", "kick off", "queue", "run experiment", "submit job".
- user is about to write to `results/`.

## Naming convention (mandatory)
`exp_YYYYMMDD_<dataset>_<method>_<tag>`
- `<dataset>` :: short id, e.g. `fsc147`, `carpk`, `shta`, `shtb`, `trancos`.
- `<method>` :: short id, e.g. `occam`, `occam-clip`, `occam-dinov2`.
- `<tag>` :: kebab-case, ≤16 chars, describes the variant (e.g. `r2-nms05`, `gmm5`).
- map to `results/<run_id>/` :: `run_id` may add an HHMM suffix if multiple same-day runs share a tag.

## Required `experiment.md` fields (write to `results/<run_id>/experiment.md` BEFORE training starts)

```
# <run_id>

started_at: <YYYY-MM-DD HH:MM>
status: running

## environment
git_commit: <full sha>
git_branch: <branch>
git_dirty: <yes|no, list dirty files if yes>
python: <version>
torch: <version>
cuda: <version|cpu>
hardware: <e.g. 1×A100-40GB, host=...>
device_visible: <CUDA_VISIBLE_DEVICES>

## config
config_path: <path>
config_diff_vs_baseline: <diff vs `base_config` or "n/a">
seed: <int>
deterministic: <yes|no>

## data
dataset: <name>
split: <train|val|test|all>
n_train / n_val / n_test: <ints>

## method
method_id: <method>
backbone: <e.g. clip-vitb16, dinov2-vitb14>
key_hyperparams:
  - <name>: <value>
  - ...

## launch
command: |
  <exact CLI invocation, multi-line ok>

## (filled in on completion)
ended_at: <YYYY-MM-DD HH:MM>
status: <done|failed>
duration_min: <float>
final_metrics: <flat object, mirrors metrics.json>
log_path: results/<run_id>/log.txt
notes: <one line>
```

## Workflow
1. before any training command runs:
   a. determine `run_id` per the naming convention; refuse to proceed if name conflicts.
   b. capture `git rev-parse HEAD`, branch, and `git status --porcelain`.
   c. snapshot `config.yaml` into the run folder; compute diff vs the documented baseline config.
   d. write `experiment.md` with status=running.
   e. append a row to `results/index.md` per `UPDATE_RULES` (`on_run_start`).
2. after the run finishes (or fails):
   a. update `experiment.md` :: `ended_at`, `status`, `duration_min`, `final_metrics`, `notes`.
   b. update the row in `results/index.md` :: `status=done|failed`, `primary_metric=<value|n/a>`.
3. if config diff vs baseline is non-trivial AND code changed for this run, ensure a `history/` entry exists; cross-reference its filename in `notes`.

## Constraints
- never start training before `experiment.md` is written.
- never overwrite an existing `experiment.md`; if rerunning the same `run_id`, fail loudly and ask the user to bump the tag.
- if the working tree is dirty, list the dirty files in `git_dirty`; do not try to auto-stash.
- if seed is unspecified, set one and record it; never leave seed blank.
- `final_metrics` must mirror `metrics.json` produced by `counting-eval`; values must agree.

## Failure modes to guard
- silent overwrite of a prior run with the same name.
- dirty working tree being treated as clean.
- missing seed leading to non-reproducible runs.
- metrics in `experiment.md` and `metrics.json` drifting apart.
