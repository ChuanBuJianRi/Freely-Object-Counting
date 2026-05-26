---
name: paper-reading
description: Structured deep-read of a research paper into a fixed-template summary. Use this skill whenever the user uploads, links, or points to a PDF paper and wants it summarized, compared, or added to a literature review — ALWAYS invoke when the user says "read this paper", "summarize", "compare papers", "lit review", "what does this paper do", or mentions any arXiv / venue paper, even if they don't explicitly ask for a structured summary. Produces a deterministic 6-section markdown note suitable for stitching into comparison tables later.
---

# paper-reading

Force every paper read in this project into the same 6-section structured note, so later comparison tables stitch together mechanically.

## When to invoke
- user gives a PDF (e.g. `library/paper/*.pdf`), an arXiv link, or a paper title and wants it understood.
- user asks for a literature review, related-work table, or "compare X with Y".
- user is reading something in `library/paper/` and wants notes saved alongside.

## Reading strategy (token-aware)
- if the PDF is in this repo, prefer `pdf-reading-split` (4-page chunks) for deep read; otherwise `pdf-reading-to-markdown` for full markdown.
- never try to read an entire raw PDF in one shot.

## Required output
A markdown file written to `library/paper/<paper-stem>.notes.md` with EXACTLY these sections, in this order:

```
# <paper title> (<venue/year>, <arXiv id if any>)

## Problem
- 1-3 bullets: what gap / failure of prior work this paper targets.

## Method
- 5-10 bullets describing the pipeline.
- enumerate every component shown in the method figure(s); do not skip blocks.
- for each component: input shape/type -> operation -> output shape/type.

## Key Insight
- 1-3 bullets: the conceptual claim that distinguishes the paper from baselines.

## Experiments
- datasets used :: <list>
- metrics :: <list>
- main numbers :: a table with columns `dataset | metric | this paper | best prior | delta`.
- ablations :: short bullet list of what was ablated and the takeaway.
- ALL numerical results from the paper's main and ablation tables MUST be extracted, not just the headline number.

## Limitations
- 2-5 bullets: what the paper itself admits + what we observe as weak.
- explicit failure modes if shown.

## Relation to my work
- baseline mapping :: which OCCAM stage(s) does this paper touch (proposals / features / matching / aggregation / multi-class)?
- swap-in candidate :: yes / no / partial — and why.
- open question for our project :: 1-2 lines.
```

## Constraints
- never paraphrase numbers; copy verbatim and cite the table/figure number.
- if a section has no content, write `(none reported)` rather than skipping.
- include the method-figure component list even if it is long; this is the key field for downstream comparison.
- if the paper is being compared against another already-summarized paper in `library/paper/`, append a final `## Compared with <other-paper>` section using the same structure deltas.

## After writing
- append a one-line entry to `library/category.md` under `paper/` :: `<file>.notes.md :: structured notes for <paper title>`.
- if this paper is OCCAM-relevant, log a `history/` entry of `type: add` so the project's reasoning trail captures which papers informed which design changes.
