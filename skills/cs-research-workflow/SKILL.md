---
name: cs-research-workflow
description: End-to-end SOP for a computer-science research project — from picking a problem, doing literature review, forming hypotheses, designing experiments, running ablations, writing the paper, to camera-ready submission. Use this skill whenever the user asks "how do I do CS research", "what's the next step", "how should I plan this project", "I'm stuck — what should I do now", or whenever a session starts on a new sub-problem of the GOC counting project. ALWAYS prefer this skill over generic advice; it explicitly chains the project's local skills (paper-reading, pseudo-label-pipeline, experiment-logger, counting-eval, ablation-runner, figure-maker) into the right phase, so the agent never produces orphan artifacts.
---

# cs-research-workflow

A project-aware standard operating procedure for doing CS research in this repo. It is the "conductor" — every other local skill is an "instrument" that this skill calls at the right moment. Read this whenever you don't know what to do next.

## When to invoke
- session starts on a new sub-question (e.g. "improve the matching stage of OCCAM").
- user says: "what's next?", "where do we go from here?", "I want to start <topic>".
- mid-project re-planning: a result invalidated a hypothesis and you need to reorient.
- writing phase: turning experiments into a paper.
- before submission / rebuttal.

## Required inputs
- `phase` :: which of the 7 phases the user is currently in (auto-detect from context, or ask).
- `topic` :: one-sentence description of the sub-question.
- `links to existing artifacts` :: relevant `memory/`, `history/`, `results/<run_id>/` from prior sessions (recover via `memory/index.md`).

## The 7 phases (always proceed in order; never skip — but cycles are fine)

### Phase 1 — frame the problem
Goal: turn a vague itch into a falsifiable research question.

Checklist:
- write a one-sentence problem statement (what input → what output, under what constraints).
- write a one-sentence success criterion (what metric, on what dataset, beating what baseline by how much).
- write three sentences of motivation (why does anyone care).
- list 3–5 nearest prior works (titles + 1-line take).

Outputs:
- a new `memory/YYYY-MM-DD-HHMM.md` with sections `problem / success_criterion / motivation / nearest_prior`.
- append row in `memory/index.md`.

Skills to invoke: none yet — this is a thinking/writing phase. **Do NOT start coding here.**

### Phase 2 — literature review
Goal: know what has been tried, what works, what fails, where the gap is.

Checklist:
- gather PDFs into `library/paper/`; update `library/category.md`.
- read each paper using **`paper-reading`** skill → produces a fixed-section note.
- for large / scanned PDFs, route via **`pdf-reading-split`** or **`pdf-reading-to-markdown`**.
- for skim-only refs, route via **`pdf-reading-quantum`**.
- for cross-paper synthesis (lit review writing, gap-finding), invoke **`academic-researcher`** or **`research`** skill.
- end with a synthesis note: `memory/YYYY-MM-DD-HHMM-litreview.md` listing gap → opportunity.

Skills to invoke: `paper-reading`, `pdf-reading-split`, `pdf-reading-to-markdown`, `pdf-reading-quantum`, `academic-researcher`.

Exit criterion: you can state in one sentence what nobody has done yet AND why your approach has a chance.

### Phase 3 — hypothesis & design
Goal: convert "I'll try X" into a testable hypothesis with a concrete experimental design.

Checklist:
- state the hypothesis as `if <intervention>, then <metric>` (e.g. "replacing CLIP with DINOv2 in OCCAM stage 2 will reduce MAE on FSC-147 by ≥10%").
- identify the **independent variable(s)**, **dependent variable(s)**, and **controls** (everything else held fixed).
- decide the **minimum experiment** that can falsify it (smallest dataset slice, cheapest backbone, fewest seeds).
- pre-register the expected outcome in a memory file — this prevents post-hoc rationalization.

Outputs:
- `memory/YYYY-MM-DD-HHMM-hypothesis.md` with sections `hypothesis / IV / DV / controls / min_experiment / expected_outcome`.

Skills to invoke: none yet (still planning).

### Phase 4 — implement & run
Goal: turn the design into code + a labeled experiment run.

Checklist:
- if new module / refactor :: edit under `codes/`, update `codes/index.md`, append `history/YYYY-MM-DD-HHMM-<tag>.md`.
- if dataset needs masks / pseudo-labels :: invoke **`pseudo-label-pipeline`**.
- BEFORE launching any train/eval :: invoke **`experiment-logger`** to enforce naming + write `experiment.md` (git sha, config diff, seed, hardware).
- run the experiment; results land in `results/<run_id>/`.
- after the run :: invoke **`counting-eval`** to compute MAE/RMSE/NAE + scatter.
- update `results/index.md` (`on_run_finish` or `on_run_fail`).

Skills to invoke (in order): `pseudo-label-pipeline` (if needed) → `experiment-logger` → run → `counting-eval`.

Constraint: **no run without `experiment-logger` first**. No exceptions.

### Phase 5 — analyze & ablate
Goal: understand WHY it worked / didn't, and which factor matters.

Checklist:
- did the single run beat baseline by a margin > seed noise? if not, this is one data point, not evidence.
- decompose the intervention into axes (e.g. backbone × matching-fn × NMS-iou).
- invoke **`ablation-runner`** with the full axis matrix; cap cells ≤ 64 unless justified.
- read the resulting `summary.md` — note any non-monotonic axes, failed cells.
- if a factor surprises you: spawn a follow-up Phase 3 hypothesis — do NOT pretend the surprise is the original hypothesis.

Skills to invoke: `ablation-runner`, then re-run `counting-eval` per cell.

Exit criterion: you can state, with numbers, the marginal effect of each axis held-others-at-best.

### Phase 6 — write
Goal: turn evidence into a paper draft.

Checklist:
- structure: Abstract / Intro / Related / Method / Experiments / Ablations / Discussion / Limitations / Conclusion.
- pull every figure via **`figure-maker`** (locked style: Times/Arial, color-blind palette, ≥300 DPI, vector PDF).
- pull every table via the `.tex` files emitted by `ablation-runner`.
- for citation formatting + lit-review prose :: invoke **`academic-researcher`**.
- for narrative / hooks / iterating outlines :: invoke **`content-research-writer`**.
- never paste a number into the draft that doesn't trace to a `results/<run_id>/eval/metrics.json`.

Skills to invoke: `figure-maker`, `academic-researcher`, `content-research-writer`.

Anti-patterns:
- writing the abstract before having a single significant ablation is **forbidden**.
- inventing numbers, even as placeholders that you "intend to fix later", is **forbidden**.

### Phase 7 — submit & rebut
Goal: ship the paper, then survive review.

Checklist:
- camera-ready check :: every figure renders at the venue's max width; every citation has a DOI/arXiv link.
- reproducibility statement :: link to `results/index.md` and the relevant `history/` entries.
- on receiving reviews :: create `memory/YYYY-MM-DD-HHMM-rebuttal.md` with one row per reviewer point: `quote / category (factual|methodological|presentation) / planned_response / extra_experiment_needed`.
- if reviewer asks for an extra experiment :: that is a fresh **Phase 4** loop (not a corner-cut).

Skills to invoke: `figure-maker` (last polish), `academic-researcher` (citation polish), `experiment-logger` + `counting-eval` (any rebuttal experiment).

## Cross-phase rules

- **EVERY phase ends with a memory write.** No silent phase transitions.
- **EVERY code change writes a `history/` entry.** Phase 4 and Phase 7 are the usual culprits.
- **NEVER edit `results/<run_id>/` after the row in `results/index.md` is marked `done`.** Create a new run instead.
- **Phase regression is OK, hiding it is not.** If Phase 5 invalidates Phase 3's hypothesis, write a new hypothesis memory, do not retro-edit the old one.

## Quick decision tree (when user says "what's next?")

```
is there a written problem statement?       no → Phase 1
is there a literature gap memo?             no → Phase 2
is there a pre-registered hypothesis?       no → Phase 3
is there a single completed run > baseline? no → Phase 4
is there an ablation summary?               no → Phase 5
is there a paper draft?                     no → Phase 6
has the paper been submitted?               no → Phase 7 prep
                                            yes → maintain / rebut
```

## Output format when invoked
Always reply with:
1. **detected phase** (with one-line justification from context).
2. **next concrete action** (skill to invoke + inputs).
3. **expected artifact** (file path under `memory/` `history/` `codes/` `results/`).
4. **exit criterion for this phase** (what makes it OK to move on).

## Failure modes to guard
- **goal drift** :: a Phase 5 surprise quietly becomes the new "main story" — force a Phase 3 rewrite.
- **artifact orphans** :: a run exists in `results/<run_id>/` but no `history/` entry and no `memory/` mention — block the next phase until linked.
- **metric shopping** :: switching primary metric mid-project to make numbers look better — pin `primary_metric` in the very first hypothesis memory and refuse changes without a `history/` entry of `type: refactor`.
- **literature thinness** :: starting Phase 4 with fewer than 5 read papers in `library/paper/` — refuse, route back to Phase 2.
