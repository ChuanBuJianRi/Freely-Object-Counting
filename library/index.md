# library/ index (agent-readable)

PURPOSE: Index of all folders and files under `library/`. Agents MUST read this before accessing `library/`, and MUST update this file on any add/delete/rename inside `library/`.

UPDATE_RULES:
- on_add_folder: append a new `## <folder>/` section with `purpose:` and `files:` fields.
- on_add_file: append `- <filename> :: <one-line description>` under the matching folder's `files:` list.
- on_delete: remove the corresponding entry.
- on_rename: update path/name in place.
- keep entries one-line, lowercase keys, no prose.

## tree
```
library/
├── index.md      # this index
├── notes/        # project-internal method notes / synced docs from FreeCounting
├── paper/        # academic papers / reference PDFs
│   └── research-guides/   # classic CS-research methodology PDFs
└── scripts/      # project scripts (data, experiments, utilities)
```

## paper/
purpose: store academic papers and reference materials (PDF) for research, citation, reading.
files:
- OCCAM.pdf :: OCCAM paper, reference for object counting / open-vocabulary counting.

## paper/research-guides/
purpose: classic CS-research methodology / how-to-do-research PDFs; consumed by `cs-research-workflow` and `paper-reading` skills.
files:
- hamming-you-and-your-research.pdf :: Richard Hamming (1986), Bell Labs talk on doing research that matters; problem selection, working on important problems, courage to commit.
- keshav-how-to-read-a-paper.pdf :: S. Keshav, the canonical "three-pass method" for reading academic papers efficiently.
- patterson-bad-career.pdf :: David Patterson, "How to Have a Bad Career in Research/Academia"; reverse-perspective survival guide for grad students and faculty.
- chapman-mit-ai-lab.pdf :: David Chapman ed. (1988, MIT AI Lab WP-316), "How to do Research At the MIT AI Lab"; reading, writing, programming, advisor selection, methodology, emotional factors.
- bundy-researchers-bible.pdf :: Alan Bundy, Ben du Boulay, Jim Howe, Gordon Plotkin (Edinburgh, 1985, rev. 1995), "The Researchers' Bible"; PhD/MPhil pitfalls, choosing & executing a project, journals.

## scripts/
purpose: store project scripts (data processing, experiment runners, utilities).
files:
- (empty)

## notes/
purpose: project-internal method notes and synced documents from upstream (FreeCounting). Source of truth for non-paper write-ups; cross-referenced from results/ READMEs.
files:
- SNG-method.md :: full write-up of the (ε, δ) Shared-Neighbor Graph clustering method (definition, complexity, FSC-147 ablation table, η-based theoretical analysis, 5 architecture-improvement directions §7.1–§7.5, reproduction commands). Synced from `Model_innovation.md`.
- MCV-method.md :: write-up of the Mode-Cluster-Vote prediction head proposed to fix OCCAM-MP7's 201+ bucket failure (MAE 312.96). Defines `predict_count(strategy=mode_cluster_vote)`: anchored at the largest cluster, sum every cluster within `k * MAD` of its `log10(bbox_area_ratio)`; `k=1.5` reused from existing `mask_iqr_k` ⇒ zero new hyperparameters. Documents 4 failure modes + relation to SNG + reproduction command.
- freecounting-original-readme.md :: original FreeCounting repo README (FSC-147 Val MAE 43.65 / Test MAE 45.47 for OCCAM-S baseline; structure overview).
- occam-impl-original-readme.md :: original OCCAM reimplementation README (pipeline description, single/multi mode parameters, run instructions).

## changelog
- 2026-05-26 :: init; registered paper/ (OCCAM.pdf), scripts/ (empty).
- 2026-05-26 14:31 :: registered paper/research-guides/ subfolder with 5 classic CS-research methodology PDFs (Hamming, Keshav, Patterson, Chapman, Bundy).
- 2026-05-27 10:45 :: registered notes/ subfolder with SNG-method.md (synced from FreeCounting Model_innovation.md) and 2 upstream READMEs.
- 2026-05-27 14:06 :: added notes/MCV-method.md (Mode-Cluster-Vote prediction head proposed to fix MP7's 201+ bucket; companion to SNG-method.md; cross-referenced from history/2026-05-27-1406-add-mcv-pred-head.md).
