# 2026-05-26 13:58 :: install initial Claude Skills + register skills/ in agent contract

type: add
scope: skills/, AGENTREAD.md
author: agent+user
related_memory: memory/2026-05-26-1358.md
related_run: none

summary:
- Installed 6 Claude Skills under `skills/` per user request: 3 official (`pdf`, `frontend-design`, `skill-creator`) from anthropics/skills, plus 3 community variants for "pdf-reading" since no official skill of that exact name exists (`pdf-reading-to-markdown`, `pdf-reading-split`, `pdf-reading-quantum`).
- `file-reading` skill was deferred — no canonical source found.
- Authored `skills/index.md` as an agent-readable index following project conventions (lowercase keys, one-line entries, `UPDATE_RULES` for install/remove/rename/update).
- Updated `AGENTREAD.md` to register `skills/` in the directory tree, the index registry table, and the "files that MUST be kept up to date" table.

files_changed:
- skills/pdf/ :: added :: copied from anthropics/skills (PDF merge / form-fill / image extraction).
- skills/frontend-design/ :: added :: copied from anthropics/skills (demo pages, dashboards).
- skills/skill-creator/ :: added :: copied from anthropics/skills (meta-skill for authoring skills).
- skills/pdf-reading-to-markdown/ :: added :: from aliceisjustplaying/claude-skill-pdf-to-markdown.
- skills/pdf-reading-split/ :: added :: from scunning1975/MixtapeTools (.claude/skills/split-pdf).
- skills/pdf-reading-quantum/ :: added :: from SPA3K/quantum-reading-skill.
- skills/index.md :: added :: agent-readable index with sources, purposes, and update rules.
- AGENTREAD.md :: modified :: added skills/ to directory tree, index registry, and must-update table.

operations_delta:
- (none — skills are agent capabilities, not project code under codes/)

verification:
- ls skills/ :: 6 skill folders present, each with SKILL.md.
- grep `skills/` AGENTREAD.md :: appears in directory tree, registry table, and must-update table.

followups:
- decide whether `file-reading` is still needed; if so, either find a canonical source or use `skill-creator` to author a local router skill.
- consider symlinking `skills/` to `.claude/skills/` so Claude Code auto-loads these (currently only project-rooted skills/ is in place).
- once a counting demo is ready, invoke `frontend-design` to scaffold a project landing / demo page.
