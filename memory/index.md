# memory/ index (agent-readable)

PURPOSE: Index of session memories under `memory/`. Each memory file is a concise summary of what was done in a chunk of conversation. Agents MUST read this index when looking up past context, and MUST append a new memory file (and a new index entry) periodically as the session progresses.

LAYOUT:
- memory file path :: `memory/YYYY-MM-DD-HHMM.md` (24h local time, zero-padded, flat — no nested folders)
  - example: `memory/2026-05-26-1328.md`
- one file per memory chunk; a chunk = several user turns of related work (rule of thumb: every ~3-5 user prompts, or whenever topic shifts).

WHEN_TO_CREATE_NEW_MEMORY:
- every ~3-5 user prompts within the same topic, OR
- whenever the topic / task shifts, OR
- at the end of a session.

MEMORY_FILE_TEMPLATE:
```
# <YYYY-MM-DD HH:MM> :: <one-line title>

scope: <which folders / files / topics this chunk touched>
summary: <2-5 lines, what was actually done>
key_decisions:
- <decision or convention established>
files_touched:
- <path> :: <created|modified|deleted> :: <one-line reason>
followups:
- <open question or next step, if any>
```

UPDATE_RULES:
- on_create_memory: create the file at `memory/YYYY-MM-DD-HHMM.md` using the template above, then append one entry to the `## entries` list in this index, format:
  - `- YYYY-MM-DD HH:MM :: memory/YYYY-MM-DD-HHMM.md :: <one-line summary>`
- on_delete_memory: remove the corresponding line from `## entries`.
- on_rename_memory: update the path in place.
- entries are kept in chronological order (oldest first).
- keep one-line summaries concise; full detail lives in the memory file itself.

## tree
```
memory/
├── index.md                    # this index
└── YYYY-MM-DD-HHMM.md          # one file per memory chunk (flat)
```

## entries
- 2026-05-26 13:28 :: memory/2026-05-26-1328.md :: bootstrapped agent-readable index files for library/, codes/, memory/.
- 2026-05-26 13:58 :: memory/2026-05-26-1358.md :: installed initial Claude Skills under skills/ and registered skills/ in AGENTREAD.md.
- 2026-05-26 14:08 :: memory/2026-05-26-1408.md :: authored 6 local project-specific skills (counting-eval, paper-reading, ablation-runner, pseudo-label-pipeline, experiment-logger, figure-maker).
- 2026-05-26 14:20 :: memory/2026-05-26-1420-handoff.md :: HANDOFF snapshot — full state of contract, skills, decisions, open questions, and next concrete steps for incoming agent.
- 2026-05-26 14:31 :: memory/2026-05-26-1431.md :: linked 3 external research-process skills + authored cs-research-workflow SOP skill + downloaded 5 classic CS-research guide PDFs into library/paper/research-guides/.

## changelog
- 2026-05-26 :: init memory index; defined naming, template, and update rules.
- 2026-05-26 :: switched layout from nested `YYYY/MM/DD/HHMM.md` to flat `YYYY-MM-DD-HHMM.md`.
