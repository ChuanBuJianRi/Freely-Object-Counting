# AGENTREAD.md (project entry point for agents)

PURPOSE: First file every agent reads when entering this repo. Defines the project, the directory contract, the index files, and the current research task. Agents MUST read this file at the start of any session, then read the relevant `index.md` for any folder they touch.

## 1. project

- name :: GOC-Freely-Object-Counting (GOC Free-Counting)
- goal :: training-free, label-free, class-free object counting in open-world settings — given an arbitrary image, estimate the count of any object category without task-specific training, ground-truth annotations, or a predefined class list.
- baseline :: OCCAM (Class-Agnostic, Training-Free, Prior-Free and Multi-Class Object Counting; Spanakis et al., arXiv 2601.13871). PDF in `library/paper/OCCAM.pdf`.
- approach :: take OCCAM as the baseline pipeline and replace / augment its internal modules to improve counting accuracy and robustness while preserving its training-free / class-free properties.

## 2. directory contract

Top-level layout and what each folder is for. Every folder owns its own `index.md` (or `category.md`); agents MUST read that index before touching files inside.

```
.
├── AGENTREAD.md            # this file (project entry point)
├── README.md               # human-facing project description
├── LICENSE
├── history.md              # index of code-change history (append-only)
├── history/                # one md file per logical code change
├── codes/                  # all project code
│   └── index.md            # code index: purpose, operations, entrypoint, usage
├── library/                # external references (papers, helper scripts)
│   ├── category.md         # library index
│   ├── paper/              # academic papers / reference PDFs
│   └── scripts/            # third-party / utility scripts
├── results/                # experiment outputs, one folder per run
│   └── index.md            # run index: run_id, task, dataset, metric, status
├── memory/                 # session memories (per-conversation summaries)
│   ├── index.md            # memory index
│   └── YYYY-MM-DD-HHMM.md  # one file per conversation chunk
├── skills/                 # installed Claude Skills (auto-loadable agent capabilities)
│   ├── index.md            # skills index: source, install date, purpose
│   └── <skill-name>/       # one folder per skill, must contain SKILL.md
└── tmp/                    # short-lived scratch files
    └── index.md            # tmp index: created_by, expires, purpose
```

## 3. index registry

| folder | index file | role |
| --- | --- | --- |
| `library/` | `library/category.md` | catalog of papers + helper scripts |
| `codes/` | `codes/index.md` | what each code unit does and which operations it exposes |
| `results/` | `results/index.md` | one row per experiment run + per-run folder layout |
| `tmp/` | `tmp/index.md` | every temporary file with creator + expiry |
| `memory/` | `memory/index.md` | conversation memory chunks (`YYYY-MM-DD-HHMM.md`) |
| `skills/` | `skills/index.md` | installed Claude Skills (one folder per skill, each with `SKILL.md`) |
| `history/` | `history.md` | append-only log of code changes (`history/YYYY-MM-DD-HHMM-<tag>.md`) |

All index files are **agent-readable**: lowercase keys, fixed fields, one-line entries, no prose paragraphs. Open the index before reading any file in the folder; update the index whenever you add / modify / delete / rename anything.

## 4. files that MUST be kept up to date (read this carefully)

These files are the live state of the project. Forgetting to update them is a bug. Each row tells you: **the path**, **when to update it**, **how to update it**, and **the matching trigger**.

| path | when to update (trigger) | how to update |
| --- | --- | --- |
| `AGENTREAD.md` (this file) | research goal, baseline, or directory contract changes | edit the affected section + bump section number if structure changes |
| `library/category.md` | any add / delete / rename of a folder OR file under `library/` | follow `UPDATE_RULES` inside the file (`on_add_folder`, `on_add_file`, ...) |
| `codes/index.md` | any add / delete / rename of code under `codes/`, OR when a code unit's executable operations change | follow `UPDATE_RULES` (`on_add_folder`, `on_add_file`, `on_modify_behavior`); always keep `operations:` accurate |
| `results/index.md` | every experiment run: at start, on finish, on fail, on delete, on rename | follow `UPDATE_RULES` (`on_run_start`, `on_run_finish`, `on_run_fail`, ...); one row per run; full metrics live in `results/<run_id>/metrics.json` |
| `tmp/index.md` | every file placed in `tmp/`, every deletion / promotion out of `tmp/` | follow `UPDATE_RULES` (`on_add_file`, `on_delete_file`, `on_promote`); always record `created_by` + `expires` |
| `memory/index.md` | every ~3–5 user prompts, OR on topic shift, OR at session end | create `memory/YYYY-MM-DD-HHMM.md` from the template, then append one row to `## entries` |
| `skills/index.md` | install / remove / rename / update of any skill under `skills/` | follow `UPDATE_RULES` (`on_install_skill`, `on_remove_skill`, `on_rename_skill`, `on_update_skill`); one row per skill |
| `history.md` | every logical code change (add / modify / delete / refactor / fix / rename) under `codes/` or `library/scripts/` | create `history/YYYY-MM-DD-HHMM-<tag>.md` from the template, then append one row to `## entries`; **append-only**, never delete past entries |

Per-folder index files always own the **detailed** `UPDATE_RULES` and field formats. This table is the **summary**: agents must consult the matching `index.md` for the exact rule before writing.

Companion files (created on demand, not pre-existing in the index list above):

| path pattern | created when | created from |
| --- | --- | --- |
| `memory/YYYY-MM-DD-HHMM.md` | on creating a new memory chunk | template inside `memory/index.md` |
| `history/YYYY-MM-DD-HHMM-<tag>.md` | on logging a code change | template inside `history.md` |
| `results/<run_id>/` (folder + `config.yaml`, `metrics.json`, `log.txt`, ...) | on starting a new experiment run | layout inside `results/index.md` |

Cross-update obligations (a single action often requires updating multiple files; do them all in the same change):

- **changing code that exposes operations** :: update `codes/index.md` (`operations:`) + create a `history/` entry (`operations_delta:`).
- **starting an experiment run** :: create `results/<run_id>/` + append row in `results/index.md`; if the run is tied to a specific code change, reference its `history/` file from the run's `README.md`.
- **promoting a file out of `tmp/`** :: delete its row from `tmp/index.md` + add an entry in the destination folder's index.
- **closing a memory chunk that touched code** :: list `files_touched` in the memory file and verify each is also reflected in `codes/index.md`, `history.md`, or `results/index.md` as appropriate.

## 5. agent workflow (every session)

1. read `AGENTREAD.md` (this file).
2. read `memory/index.md` and the latest few memory entries to recover context.
3. read the index of any folder you plan to touch (`codes/index.md`, `results/index.md`, etc.).
4. perform the task, following each folder's `UPDATE_RULES`.
5. on every code change: create a new `history/YYYY-MM-DD-HHMM-<tag>.md` and append a row to `history.md`.
6. periodically (every ~3-5 user prompts, or on topic shift, or at session end): create a new memory file `memory/YYYY-MM-DD-HHMM.md` and append a row to `memory/index.md`.

## 6. current task

task_id :: improve-occam-baseline
status :: active
goal :: starting from the OCCAM pipeline, replace / augment selected modules to achieve better counting performance (lower MAE/RMSE on standard benchmarks such as FSC-147, CARPK, and similar open-world counting datasets) while preserving training-free / label-free / class-free properties.

baseline_summary (OCCAM, as captured in `library/paper/OCCAM.pdf`):
- stage_1 :: class-agnostic mask / region proposals (SAM-family).
- stage_2 :: feature extraction over proposals (vision-language / self-supervised backbones, e.g. CLIP, DINO).
- stage_3 :: prompt / exemplar matching to identify the target category in feature space (text prompt or reference patch).
- stage_4 :: similarity-based selection / clustering of proposals → final count (and optional multi-class extension).

candidate_axes_for_improvement (do NOT commit to one without ablation; explore as needed):
- proposal generator :: alternatives or post-filtering for SAM (mask granularity, NMS, scale handling, density-aware sampling).
- feature backbone :: swap / ensemble CLIP, DINOv2, EVA-CLIP, SigLIP; investigate frozen vs. adapted features.
- matching / scoring :: better similarity functions, calibration, prototype construction from text and/or exemplars, hard-negative handling.
- aggregation / counting :: robust counting from noisy proposals (clustering, density estimation, redundancy removal).
- multi-class handling :: disambiguating co-occurring categories.
- efficiency :: latency / memory profile of each module.

operating_constraints:
- training-free :: no fine-tuning / gradient updates on counting datasets unless explicitly approved.
- label-free :: no use of dataset ground-truth during inference.
- class-free :: no fixed class list; queries arrive as text prompt or exemplar.
- reproducibility :: every run goes into `results/<run_id>/` with `config.yaml`, `metrics.json`, `log.txt`.

next_concrete_steps (suggested; revise as work progresses):
1. set up `codes/` skeleton mirroring OCCAM's stages (proposals / features / matching / aggregation) so individual modules can be swapped.
2. reproduce OCCAM baseline numbers on at least one benchmark; record the run under `results/`.
3. pick one axis above for the first ablation; document hypothesis in a memory file before running.
4. log every code change under `history/`, every experiment under `results/`, every conversation chunk under `memory/`.

## 7. cross-folder contracts

- a code change that alters the operations of a script in `codes/` :: MUST update `codes/index.md` AND log a `history/` entry whose `operations_delta` field reflects the change.
- a new experiment run :: MUST create `results/<run_id>/` AND append a row in `results/index.md`; if the run was triggered by a specific code change, reference the matching `history/` file in the run's `README.md`.
- a memory file :: SHOULD list `files_touched` so that history / results / index updates can be cross-checked.
- never put long-lived artifacts in `tmp/`; promote them to `codes/`, `results/`, or `library/` and update both indexes.

## 8. style for index entries

- lowercase keys (`purpose`, `operations`, `files`, `status`, ...).
- one-line entries; multi-line prose belongs in dedicated files (memory / history / per-run README).
- entry separator inside a line :: ` :: ` (space-colon-colon-space).
- timestamps :: `YYYY-MM-DD HH:MM` (24h, local, zero-padded).
- file naming :: kebab-case for tags, no spaces, ≤24 chars for tag fields.
