# 2026-05-26 14:08 :: author 6 local project-specific skills

type: add
scope: skills/
author: agent+user
related_memory: memory/2026-05-26-1408.md
related_run: none

summary:
- Authored 6 local skills under `skills/` to cover the project's repeated workflows (per user spec).
- counting-eval :: standardized MAE/RMSE/NAE evaluation on FSC-147 / CARPK / ShanghaiTech, with `metrics.json` + per-category table + GT-vs-Pred scatter.
- paper-reading :: forced 6-section note (Problem / Method / Key Insight / Experiments / Limitations / Relation to my work) extracting ALL numerical results.
- ablation-runner :: turn an "A × B × C" matrix into configs + run_grid.sh + parse_logs.py + table.md + table.tex.
- pseudo-label-pipeline :: SEEM/SAM → GMM → NMS → 2× refinement with VRAM probe + checkpointing + resume.
- experiment-logger :: enforce `exp_YYYYMMDD_<dataset>_<method>_<tag>` naming + write `experiment.md` with git sha / config diff / seed / hw / timing / metrics.
- figure-maker :: locked style (Times/Arial, color-blind palette, ≥300 DPI, vector PDF) for density maps, qualitative grids, curves, bars, scatter.
- Updated `skills/index.md` :: split into `## external skills` and `## local (project-specific) skills`, refreshed tree + suggested-usage map.

files_changed:
- skills/counting-eval/SKILL.md :: added :: counting-eval skill body and YAML frontmatter.
- skills/paper-reading/SKILL.md :: added :: paper-reading skill body.
- skills/ablation-runner/SKILL.md :: added :: ablation-runner skill body.
- skills/pseudo-label-pipeline/SKILL.md :: added :: pseudo-label-pipeline skill body.
- skills/experiment-logger/SKILL.md :: added :: experiment-logger skill body.
- skills/figure-maker/SKILL.md :: added :: figure-maker skill body.
- skills/index.md :: modified :: added 6 local skills, restructured into external/local sections, expanded suggested usage.

operations_delta:
- (none — skills/ does not expose codes/ operations; future scripts spawned by these skills will themselves register operations in codes/index.md.)

verification:
- ls skills/ :: 12 skill folders present (6 external + 6 local), each with SKILL.md.
- each new SKILL.md starts with valid YAML frontmatter (`name`, `description`).
- skills/index.md tree section matches actual folders.

followups:
- once any of these skills is invoked for the first time, capture the produced artifacts (e.g. eval scripts, figure scripts) under `codes/` and register their operations in `codes/index.md`.
- consider authoring `file-reading` (router) via skill-creator if dataset ingestion gets repetitive.
- when a paper is read with `paper-reading`, confirm output lands at `library/paper/<stem>.notes.md` and library/category.md is updated.
