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

## changelog
- 2026-05-26 :: init history index; defined per-change file format and append-only rules.
