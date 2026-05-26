# skills/ index (agent-readable)

PURPOSE: Index of installed Claude Skills under `skills/`. Each skill is a self-contained directory with at minimum a `SKILL.md`. Agents MUST consult this index before invoking or modifying any skill.

LAYOUT:
- one folder per skill :: `skills/<skill-name>/`
- required file :: `skills/<skill-name>/SKILL.md` (YAML frontmatter + markdown body)
- optional :: `scripts/`, `references/`, `assets/`, `README.md`, etc.

INVOCATION:
- automatic :: Claude/Cursor agents may auto-load a skill when its frontmatter `description` matches the user request.
- explicit :: invoke by directory name, e.g. `/pdf`, `/counting-eval`, `/skill-creator`.

UPDATE_RULES:
- on_install_skill: copy / clone / author the skill into `skills/<skill-name>/`, then append a row to the matching section below.
- on_remove_skill: delete the folder and remove the row.
- on_rename_skill: rename the folder and update the row in place.
- on_update_skill: bump the `updated:` field of the matching row when content changes.
- entry format (one line per skill):
  - `- <skill-name> :: <source url or "local">  :: installed=<YYYY-MM-DD> :: <one-line purpose>`
- keep entries one-line; full docs live inside each skill's `SKILL.md`.

## tree
```
skills/
├── index.md
├── pdf/                          # external :: PDF ops (merge, fill forms, extract images)
├── frontend-design/              # external :: build demo pages / dashboards / project sites
├── skill-creator/                # external :: meta-skill, create new skills
├── pdf-reading-to-markdown/      # external :: convert full PDF -> markdown (text + images)
├── pdf-reading-split/            # external :: 4-page-chunk deep-read for academic papers
├── pdf-reading-quantum/          # external :: skim/scan large docs, save tokens
├── academic-researcher/          # external :: literature reviews, paper analysis, citation formatting
├── research/                     # external :: multi-source web research with citations
├── content-research-writer/      # external :: research-grounded writing, citations, hooks, outlines
├── counting-eval/                # local    :: standardized counting eval (MAE/RMSE/NAE)
├── paper-reading/                # local    :: structured 6-section paper notes
├── ablation-runner/              # local    :: configs + grid + ablation tables (md + tex)
├── pseudo-label-pipeline/        # local    :: SEEM/SAM -> GMM -> NMS -> 2× refinement
├── experiment-logger/            # local    :: enforce experiment.md + naming convention
├── figure-maker/                 # local    :: paper-ready figures, locked style
└── cs-research-workflow/         # local    :: 7-phase CS research SOP, conducts all other skills
```

## external skills
- pdf :: https://github.com/anthropics/skills/tree/main/skills/pdf :: installed=2026-05-26 :: official PDF skill — merge multiple reports, fill PDF forms, extract images from scanned papers.
- frontend-design :: https://github.com/anthropics/skills/tree/main/skills/frontend-design :: installed=2026-05-26 :: build project sites, demo pages, visualization dashboards (e.g. counting demo: upload image -> show density map).
- skill-creator :: https://github.com/anthropics/skills/tree/main/skills/skill-creator :: installed=2026-05-26 :: meta-skill, create your own skills (used to author the local skills below).
- pdf-reading-to-markdown :: https://github.com/aliceisjustplaying/claude-skill-pdf-to-markdown :: installed=2026-05-26 :: convert an entire PDF into clean structured markdown (preserves images, tables); good for full-context paper reading.
- pdf-reading-split :: https://github.com/scunning1975/MixtapeTools (subdir `.claude/skills/split-pdf`) :: installed=2026-05-26 :: download + split academic PDFs into 4-page chunks, deep-read in batches with structured notes — avoids context-window crashes.
- pdf-reading-quantum :: https://github.com/SPA3K/quantum-reading-skill :: installed=2026-05-26 :: skim / quantum-read large documents using 8 scanning strategies; ~40% token savings; auto-activates on `.pdf` `.docx` `.txt` `.md` >50KB or >500 lines.
- academic-researcher :: ~/.agents/skills/academic-researcher (awesome-llm-apps, MIT) :: installed=2026-05-26 :: literature review framework, paper analysis (5-section), citation formatting (APA/MLA/Chicago), academic writing standards.
- research :: ~/.agents/skills/research :: installed=2026-05-26 :: multi-source web research with explicit citations; for comparisons, market analysis, current events.
- content-research-writer :: ~/.agents/skills/content-research-writer :: installed=2026-05-26 :: collaborative writing partner — adds citations, improves hooks, iterates outlines, real-time per-section feedback.

## local (project-specific) skills
- counting-eval :: local :: installed=2026-05-26 :: standardized counting evaluation; given checkpoint/preds + dataset, computes MAE/RMSE/NAE on FSC-147/CARPK/ShanghaiTech, writes `metrics.json` + per-category table + GT-vs-Pred scatter under `results/<run_id>/eval/`.
- paper-reading :: local :: installed=2026-05-26 :: forces every paper read into a fixed 6-section note (Problem / Method / Key Insight / Experiments / Limitations / Relation to my work); extracts ALL numerical results so later comparison tables stitch mechanically.
- ablation-runner :: local :: installed=2026-05-26 :: from a config-matrix description ("A × B × C") generates configs/, run_grid.sh, parse_logs.py, and the final ablation `table.md` + `table.tex`.
- pseudo-label-pipeline :: local :: installed=2026-05-26 :: encodes the project's SEEM/SAM → GMM → NMS → 2× refinement pipeline; resume-from-failure, GPU memory probe, per-image checkpointing built in.
- experiment-logger :: local :: installed=2026-05-26 :: before any training run, enforces `exp_YYYYMMDD_<dataset>_<method>_<tag>` naming and writes `experiment.md` (git sha, config diff, seed, hardware, timing, metrics).
- figure-maker :: local :: installed=2026-05-26 :: paper-ready figures with locked style (Times/Arial, color-blind palette, ≥300 DPI, vector PDF); supports density maps, qualitative grids, curves, ablation bars, GT-vs-Pred scatter.
- cs-research-workflow :: local :: installed=2026-05-26 :: end-to-end 7-phase SOP (frame → litreview → hypothesis → run → ablate → write → submit); conducts every other local + external skill at the right phase; the "what's next?" decision tree.

## suggested usage in this project
- "what's next?" / new sub-question / re-planning :: `cs-research-workflow` FIRST — it tells you which skill to invoke next.
- read OCCAM and related papers in `library/paper/` :: `paper-reading` (uses `pdf-reading-split` or `pdf-reading-to-markdown` under the hood).
- cross-paper synthesis / lit-review prose / citation formatting :: `academic-researcher`.
- multi-source web research with citations :: `research`.
- writing the paper draft (hooks, outline, per-section feedback) :: `content-research-writer`.
- prepare a new dataset / generate masks for unlabeled images :: `pseudo-label-pipeline`.
- launching ANY training or eval job :: `experiment-logger` first, then run.
- evaluating a checkpoint on a counting benchmark :: `counting-eval`.
- comparing module variants (proposals / features / matching / aggregation) :: `ablation-runner`.
- producing every figure for the paper / demo :: `figure-maker`.
- merging multiple eval reports / extracting figures from scanned references :: `pdf` (external).
- building the project landing page / counting demo (upload image -> density map) :: `frontend-design` (external).
- authoring a new project-specific skill :: `skill-creator` (external).

## changelog
- 2026-05-26 13:58 :: init skills index; installed 6 external skills (3 official + 3 community pdf-reading variants); deferred `file-reading`.
- 2026-05-26 14:08 :: authored 6 local project-specific skills (counting-eval, paper-reading, ablation-runner, pseudo-label-pipeline, experiment-logger, figure-maker).
- 2026-05-26 14:31 :: linked 3 external research-process skills (academic-researcher, research, content-research-writer) from `~/.agents/skills/`; authored 1 local SOP skill (cs-research-workflow) that conducts all other skills across 7 phases.
