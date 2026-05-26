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
├── category.md   # this index
├── paper/        # academic papers / reference PDFs
│   └── research-guides/   # classic CS-research methodology PDFs (Hamming, Chapman, Bundy, ...)
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

## changelog
- 2026-05-26 :: init; registered paper/ (OCCAM.pdf), scripts/ (empty).
- 2026-05-26 14:31 :: registered paper/research-guides/ subfolder with 5 classic CS-research methodology PDFs (Hamming, Keshav, Patterson, Chapman, Bundy).
