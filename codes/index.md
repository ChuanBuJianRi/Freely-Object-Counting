# codes/ index (agent-readable)

PURPOSE: Index of all code under `codes/`. Each entry describes WHAT the code is and WHICH operations it supports. Agents MUST read this before running or modifying anything in `codes/`, and MUST update this file on any add/delete/rename/behavior-change inside `codes/`.

UPDATE_RULES:
- on_add_folder: append a new `## <folder>/` section with `purpose:`, `entrypoint:`, `operations:`, `files:` fields.
- on_add_file: append a new `### <relative/path>` block with `purpose:`, `operations:`, `usage:` fields.
- on_modify_behavior: update the corresponding `operations:` and/or `usage:` field; add a `changelog` entry.
- on_delete: remove the corresponding entry.
- on_rename: update path/name in place.
- field rules:
  - `purpose:` :: one line, what this code is for.
  - `operations:` :: bullet list, one verb per line, each describing a discrete action the code can perform (e.g. `- train model on dataset X`, `- run inference on a single image`, `- export checkpoint to ONNX`).
  - `entrypoint:` :: command(s) used to invoke the main behavior (e.g. `python train.py --config configs/base.yaml`).
  - `usage:` :: minimal example invocation if the file is directly runnable; omit if not runnable.
  - `inputs:` / `outputs:` :: optional, list expected paths, formats, or data shapes when relevant.
- keep keys lowercase, values concise, no prose paragraphs.

## tree
```
codes/
└── index.md   # this index (no code yet)
```

## (no code registered yet)
Add a `## <folder>/` or `### <file>` section here as soon as code is placed under `codes/`.

## changelog
- 2026-05-26 :: init; codes/ is empty.
