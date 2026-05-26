# 2026-05-26 14:31 :: install research-process skills + add CS-research guides

type: add
scope: skills/, library/paper/research-guides/, skills/index.md, library/index.md
author: agent
related_memory: memory/2026-05-26-1431.md
related_run: none

summary:
- copied 3 external research-process skills from `~/.agents/skills/` into `skills/` (academic-researcher, research, content-research-writer).
- authored 1 new local SOP skill `cs-research-workflow/SKILL.md` (7 phases; dispatches every other local + external skill).
- created `library/paper/research-guides/` and downloaded 5 classic methodology PDFs (Hamming, Keshav, Patterson, Chapman, Bundy).
- updated `skills/index.md` and `library/index.md` to register all new artifacts; convention changes: skill-section now distinguishes external vs local with `~/.agents/skills/...` source URLs accepted alongside GitHub URLs.

files_changed:
- skills/academic-researcher/SKILL.md :: added :: external skill copy.
- skills/research/SKILL.md :: added :: external skill copy.
- skills/research/scripts/ :: added :: external skill helper scripts.
- skills/content-research-writer/SKILL.md :: added :: external skill copy.
- skills/cs-research-workflow/SKILL.md :: added :: new local SOP skill (7 phases + decision tree).
- library/paper/research-guides/hamming-you-and-your-research.pdf :: added :: Hamming 1986 talk.
- library/paper/research-guides/keshav-how-to-read-a-paper.pdf :: added :: three-pass paper-reading method.
- library/paper/research-guides/patterson-bad-career.pdf :: added :: Patterson "Bad Career" talk.
- library/paper/research-guides/chapman-mit-ai-lab.pdf :: added :: MIT AI Lab WP-316 (1988).
- library/paper/research-guides/bundy-researchers-bible.pdf :: added :: Edinburgh "Researchers' Bible" 1985 (rev. 1995).
- skills/index.md :: modified :: tree + 4 new rows + suggested-usage + changelog.
- library/index.md :: modified :: new subfolder section + 5 file entries + changelog.

operations_delta:
- (none — no `codes/` operations changed; this change is meta/skills/library only.)

verification:
- `ls skills/` confirms 16 directories (was 12, +4 new: academic-researcher, research, content-research-writer, cs-research-workflow).
- `file library/paper/research-guides/*.pdf` confirms all 5 are valid PDF files (sizes: 74KB–531KB).
- spot-checked Chapman PDF: actual page count = 38 (the `file` tool's "2 pages" was misleading).

followups:
- SPJ "How to write a great research paper" PDF still missing (MSR 403/timeout from this env); add later via alternative mirror.
- consider an aggregate `library/paper/research-guides/SUMMARY.md` keyed by `cs-research-workflow` phases.
