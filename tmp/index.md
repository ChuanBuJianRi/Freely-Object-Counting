# tmp/ index (agent-readable)

PURPOSE: Index of all temporary files under `tmp/`. Anything in `tmp/` is short-lived: scratch outputs, intermediate artifacts, debug dumps, throwaway downloads. Agents MUST register every file they place in `tmp/` here, and MUST update the index on add/delete/rename.

UPDATE_RULES:
- on_add_file: append one entry to `## files`, format:
  - `- <filename> :: <created_by> :: <created_at YYYY-MM-DD HH:MM> :: <expires|keep_until> :: <one-line purpose>`
- on_delete_file: remove the corresponding line.
- on_rename_file: update the name in place.
- on_promote: if a file graduates to a permanent home (e.g. `codes/`, `results/`, `library/`), delete it from `tmp/` AND remove its entry here; record the move in the destination folder's index.
- field rules:
  - `created_by` :: short tag, e.g. `agent`, `user`, `script:train.py`.
  - `expires` :: rough lifetime hint, e.g. `1d`, `1w`, `session`, `until_next_run`. Use `keep_until YYYY-MM-DD` for a hard date.
  - never store anything in `tmp/` that must survive long-term; promote it instead.
- keep entries one-line, lowercase keys, no prose.

## tree
```
tmp/
└── index.md   # this index (no temp files yet)
```

## files
- dr-bootstrap-prompt.md :: agent :: 2026-05-26 15:20 :: keep_until 2026-06-02 :: 首轮 OpenAI Deep Research 的项目背景 + 调研问题 prompt

## changelog
- 2026-05-26 :: init tmp index; defined entry format and lifetime conventions.
