# results/ index (agent-readable)

PURPOSE: Index of experiment outputs under `results/`. Every reproducible run (training, evaluation, inference) lives in its own subfolder; this index lists those runs with their config, metrics, and status. Agents MUST register a new run here as soon as it produces output, and MUST update it on completion / deletion / rename.

LAYOUT:
- one folder per run :: `results/<run_id>/`
- run_id format :: `YYYY-MM-DD-HHMM-<task>-<short_tag>`
  - `<task>` :: `train` | `eval` | `infer` | `ablation` | `debug`
  - `<short_tag>` :: kebab-case, ≤24 chars, describes the run (e.g. `occam-baseline`, `clip-vitb-fsc147`)
  - example :: `results/2026-05-26-1430-eval-occam-fsc147/`
- recommended contents inside each run folder:
  - `config.yaml` (or `.json`) :: exact config used to launch the run
  - `metrics.json` :: final metrics (one flat object, e.g. `{"MAE": 8.21, "RMSE": 12.5}`)
  - `log.txt` :: stdout/stderr or training log
  - `checkpoints/` :: model weights (optional, may be large — consider symlink / external storage)
  - `vis/` :: visualizations, qualitative outputs (optional)
  - `predictions/` :: per-sample predictions (optional)
  - `README.md` :: free-form notes for this run (optional but encouraged)

UPDATE_RULES:
- on_run_start: create `results/<run_id>/`, write `config.yaml`, append a row to `## runs` with `status: running`.
- on_run_finish: update the row's `status` to `done` and fill `primary_metric`.
- on_run_fail: set `status: failed`, keep the row, leave a one-line reason in `notes`.
- on_run_delete: remove the row from `## runs` AND delete the folder (or move it to `tmp/` first if uncertain).
- on_run_rename: rename the folder and update the row in place.
- never overwrite an existing run folder; always create a new run_id.

ENTRY_FORMAT (one line per run, under `## runs`):
```
- <run_id> :: <task> :: <dataset_or_input> :: <model_or_method> :: status=<running|done|failed> :: <primary_metric_name>=<value|n/a> :: <one-line notes>
```
- `primary_metric_name` :: pick the single most informative metric for this task (e.g. `MAE`, `RMSE`, `mAP`, `loss`, `acc@1`); record full metrics inside the run folder's `metrics.json`.
- `notes` :: short, e.g. `seed=0`, `8xA100`, `ckpt epoch_30`, `failed: OOM`.

## tree
```
results/
├── index.md                              # this index
└── YYYY-MM-DD-HHMM-<task>-<short_tag>/   # one folder per run
    ├── config.yaml
    ├── metrics.json
    ├── log.txt
    ├── checkpoints/    # optional
    ├── vis/            # optional
    ├── predictions/    # optional
    └── README.md       # optional
```

## runs
- (empty)

## changelog
- 2026-05-26 :: init results index; defined run_id format, per-run folder layout, and entry format.
