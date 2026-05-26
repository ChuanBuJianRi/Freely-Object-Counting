---
name: ablation-runner
description: Generate full ablation grids — configs, launch scripts, log parsers, and the final ablation table — from a high-level matrix description. Use this skill whenever the user describes a configuration sweep, e.g. "two refinement rounds × three NMS thresholds × two backbones", or asks to "run an ablation", "sweep over", "grid search", "compare these settings". ALWAYS prefer this over hand-writing one-off scripts; produces both Markdown and LaTeX tables for paper writing.
---

# ablation-runner

Turn an English description of a configuration matrix into an executable ablation grid + a clean comparison table, without hand-writing per-cell scripts.

## When to invoke
- user describes a sweep with multiplicative dimensions ("A × B × C").
- user says "ablation", "grid", "sweep", "compare these N settings".
- user asks for an ablation table for the paper.

## Required inputs
- `axes` :: list of named dimensions, each with discrete values, e.g.
  - `refinement_rounds: [0, 1, 2]`
  - `nms_iou: [0.3, 0.5, 0.7]`
  - `backbone: [clip-vitb, dinov2-vitb]`
- `base_config` :: path to the baseline config (any cell = base + axis overrides).
- `dataset` :: target benchmark for evaluation (used for the result column).
- `primary_metric` :: e.g. `MAE`; secondary metrics optional.
- `run_id_prefix` :: stem under `results/`, full id = `<prefix>-<axis_hash>`.

## Outputs (write to `results/<run_id_prefix>/ablation/`)
1. `configs/` :: one config file per cell, named by axis values, e.g.
   `configs/r1_nms0.5_clip.yaml`
2. `run_grid.sh` :: shell script that launches all cells (sequentially or via `xargs -P` if user requested parallelism).
3. `parse_logs.py` :: reads each cell's `metrics.json` (per `counting-eval` skill output) and assembles a DataFrame.
4. `table.md` :: pivot table; rows and columns chosen so the most informative axis is on rows. Bold the best cell per row.
5. `table.tex` :: LaTeX `tabular` mirror of `table.md`, with `\toprule \midrule \bottomrule` (booktabs) and the same bolding.
6. `summary.md` :: one paragraph stating the best cell, the marginal effect of each axis (held-others-at-best), and any non-monotonic axes flagged.

## Algorithm
1. expand the axis cross-product into N cells; refuse if N > 64 unless user confirms.
2. for each cell: copy `base_config`, apply overrides, write to `configs/`.
3. emit `run_grid.sh` that loops cells, sets `run_id`, calls the project's training/eval entrypoint (resolved via `codes/index.md`).
4. after runs complete (or in dry-run mode just generate scripts), `parse_logs.py` aggregates `results/<cell_run_id>/eval/metrics.json` into a tidy CSV.
5. build pivot table; if more than 2 axes, place the 2 highest-cardinality axes on row/column and append a small per-(other-axis) facet.
6. write `table.md` + `table.tex`; bold per-row max (or min if metric is `MAE`/`RMSE`/`NAE`).
7. on completion, append rows to `results/index.md` (one per cell) and log ONE `history/` entry of `type: add` covering the whole grid (not one per cell).

## Constraints
- never overwrite an existing `ablation/` folder — use `ablation-v2/` if rerunning.
- LaTeX table column spec must be deterministic so paper edits are clean.
- numerical formatting :: 2 decimals; show std if ≥3 seeds per cell.
- if a cell fails (training crash, OOM), the table marks it as `—` and `summary.md` lists failed cells; do not silently drop.

## Failure modes to guard
- combinatorial blow-up :: hard cap at 64 unless user confirms.
- accidental shared state across cells :: each cell must use a fresh `run_id`.
- mixing metrics where lower-is-better with higher-is-better :: explicit per-metric direction in `parse_logs.py`.
