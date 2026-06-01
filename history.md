# history.md (agent-readable code-change log index)

PURPOSE: Append-only log of every code change made under this repo. Each change has its own markdown file in `history/`; this file is the index that lists them in chronological order. Agents MUST create a new change file AND append a new entry here whenever they modify, add, or delete code.

SCOPE_OF_CHANGE:
- code under `codes/` :: ALWAYS log here.
- scripts under `library/scripts/` :: ALWAYS log here.
- agent-readable index files (`*/index.md`, `library/category.md`, `AGENTREAD.md`, this file) :: log here only if the change alters rules / conventions; not for routine entry edits.
- generated artifacts under `results/`, `tmp/` :: do NOT log here (those are tracked by their own indexes).
- `library/paper/`, `memory/` :: do NOT log here.

DIFFERENCE_FROM_MEMORY:
- `memory/` :: per-conversation context, free-form summary, task-oriented.
- `history.md` + `history/` :: per-code-change record, structured, code-oriented, append-only audit trail.
- a single conversation may produce one memory file AND multiple history entries (one per logical change).

LAYOUT:
- change file path :: `history/YYYY-MM-DD-HHMM-<short-tag>.md` (24h local time, zero-padded, kebab-case tag ≤24 chars)
  - example :: `history/2026-05-26-1430-add-train-script.md`
- one file per logical change (one feature / one fix / one refactor); do not bundle unrelated edits.

CHANGE_FILE_TEMPLATE:
```
# <YYYY-MM-DD HH:MM> :: <one-line title>

type: <add|modify|delete|refactor|fix|rename>
scope: <which folders / files this change affects>
author: <agent|user|agent+user>
related_memory: <memory/YYYY-MM-DD-HHMM.md|none>
related_run: <results/<run_id>|none>

summary: <2-5 lines, what was changed and why>

files_changed:
- <path> :: <added|modified|deleted|renamed> :: <one-line reason>

operations_delta:   # optional; fill if `codes/` operations changed
- <path> :: <added|removed|changed> :: <operation description>

verification:       # optional; how the change was sanity-checked
- <command run, test added, manual check, etc.>

followups:
- <open question or next step, if any>
```

UPDATE_RULES:
- on_code_change: create the file at `history/YYYY-MM-DD-HHMM-<short-tag>.md` using the template, then append one row to `## entries` below, format:
  - `- YYYY-MM-DD HH:MM :: history/YYYY-MM-DD-HHMM-<short-tag>.md :: <type> :: <one-line summary>`
- on_delete_change_file: NEVER delete a history entry once written; if a change must be reverted, log the revert as a NEW entry of `type: modify` (or `fix`) referencing the original file path.
- on_rename_change_file: avoid renaming; if absolutely required, update the path in `## entries` and add a follow-up row noting the rename.
- entries are append-only and kept in chronological order (oldest first).
- if `codes/` operations changed, the corresponding `codes/index.md` MUST also be updated; cross-reference is done via the `operations_delta` field.

## tree
```
./
├── history.md                                  # this index
└── history/
    └── YYYY-MM-DD-HHMM-<short-tag>.md          # one file per logical code change
```

## entries
- 2026-05-26 13:58 :: history/2026-05-26-1358-install-claude-skills.md :: add :: installed 6 Claude Skills under skills/ (3 official + 3 community pdf-reading variants); registered skills/ in AGENTREAD.md.
- 2026-05-26 14:08 :: history/2026-05-26-1408-author-local-skills.md :: add :: authored 6 local project-specific skills (counting-eval, paper-reading, ablation-runner, pseudo-label-pipeline, experiment-logger, figure-maker); restructured skills/index.md.
- 2026-05-26 14:31 :: history/2026-05-26-1431-add-research-skills.md :: add :: linked 3 external research-process skills (academic-researcher, research, content-research-writer) + authored 1 local SOP skill (cs-research-workflow) + downloaded 5 classic CS-research guide PDFs into library/paper/research-guides/.
- 2026-05-27 10:45 :: history/2026-05-27-1045-sync-occam-and-results.md :: add :: synced OCCAM baseline package + SNG method write-up + 6 evaluation/ablation runs from FreeCounting/ws_yiyang into codes/occam/, codes/scripts/, codes/eval/, library/notes/, results/.
- 2026-05-27 11:45 :: history/2026-05-27-1145-harden-gpu-safety.md :: refactor :: extracted GPU thermal guard into codes/eval/_gpu_safety.py (auto-detects nvidia-smi; CLI knobs; thermal stats land in metrics.json); mandated via AGENTREAD.md / results/index.md GPU_THERMAL_POLICY; restored upstream `occam_multi/results/` nested layout for byte-aligned sync.
- 2026-05-27 11:50 :: history/2026-05-27-1150-add-adaptive-sng-delta.md :: add :: implemented §7.1 adaptive δ in codes/occam/clustering.py (sng_cluster supports delta=None ⇒ formula; new helpers adaptive_delta + eta_health); OccamConfig + eval CLI wired (--sng-alpha); authored CPU-only validator codes/scripts/synth_validate_sng.py and ran results/2026-05-27-1145-validate-sng-adaptive-delta-cpu (adaptive α=0.50 beats best fixed-δ by 14% mean MAE).
- 2026-05-27 14:06 :: history/2026-05-27-1406-add-mcv-pred-head.md :: add :: added MCV (Mode-Cluster-Vote) prediction head to codes/occam/predict.py + wired --pred-strategy mode_cluster_vote/mcv in codes/eval/eval_fsc147_full.py; reuses mask_iqr_k=1.5 (zero new hyperparameters); per-image JSON now also stores cluster trace; library/notes/MCV-method.md documents motivation/algorithm/4 failure modes; offline 5-scenario sanity test passes.
- 2026-05-28 15:38 :: history/2026-05-28-1538-add-mcv-guard.md :: add :: added `mcv_min_anchor_size` guard to predict_count + `--mcv-min-anchor-size` CLI flag, motivated by the v1 regression on 1-10/11-50 buckets observed in run 2026-05-28-1434-eval-mp7-mcv-full; trace-sweep finds plateau at A ∈ [30,40] with overall MAE ~28.9-29.5; default 0 keeps v1 behaviour unchanged.

## changelog
- 2026-05-26 :: init history index; defined per-change file format and append-only rules.
