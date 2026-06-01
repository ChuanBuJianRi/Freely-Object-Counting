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
├── library/                # external references (papers, helper scripts) and synced notes
│   ├── index.md            # library index
│   ├── paper/              # academic papers / reference PDFs
│   ├── notes/              # project-internal method notes (e.g. SNG write-up) + synced upstream docs
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
| `library/` | `library/index.md` | catalog of papers, notes, and helper scripts |
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
| `library/index.md` | any add / delete / rename of a folder OR file under `library/` | follow `UPDATE_RULES` inside `library/index.md` (`on_add_folder`, `on_add_file`, ...) |
| `codes/index.md` | any add / delete / rename of code under `codes/`, OR when a code unit's executable operations change | follow `UPDATE_RULES` (`on_add_folder`, `on_add_file`, `on_modify_behavior`); always keep `operations:` accurate |
| `results/index.md` | every experiment run: at start, on finish, on fail, on delete, on rename | follow `UPDATE_RULES` (`on_run_start`, `on_run_finish`, `on_run_fail`, ...); one row per run; full metrics live in `results/<run_id>/metrics.json`; `metrics.json` MUST include a `thermal:` block from `_gpu_safety.GpuGuard` for every GPU run |
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
status :: active (synced from FreeCounting working tree on 2026-05-27)
goal :: starting from the OCCAM pipeline, replace / augment selected modules to achieve better counting performance (lower MAE/RMSE on standard benchmarks such as FSC-147, CARPK, and similar open-world counting datasets) while preserving training-free / label-free / class-free properties.

baseline_summary (OCCAM, as captured in `library/paper/OCCAM.pdf`):
- stage_1 :: class-agnostic mask / region proposals (SAM-family).
- stage_2 :: feature extraction over proposals (vision-language / self-supervised backbones, e.g. CLIP, DINO).
- stage_3 :: prompt / exemplar matching to identify the target category in feature space (text prompt or reference patch).
- stage_4 :: similarity-based selection / clustering of proposals → final count (and optional multi-class extension).

current_state (as of 2026-05-27 11:50):
- runnable baseline :: `codes/occam/` (paper-faithful OCCAM-S/M reimplementation, ResNet-50 features, FINCH/SNG clustering, 8 mask-policy variants) + `codes/eval/eval_fsc147_full.py` (now wires the mandatory `_gpu_safety.GpuGuard`).
- proposed contribution :: (ε, δ) Shared-Neighbor Graph clustering — now with §7.1 **adaptive δ** in production (`sng_cluster(features, *, epsilon, delta=None, alpha=0.4)` + helpers `adaptive_delta`, `eta_health`). `OccamConfig.sng_delta` default flipped to `None`; new CLI flag `--sng-alpha`. Full method, complexity, η theory, AND synthetic validation in `library/notes/SNG-method.md`.
- registered runs :: 7 runs in `results/` — 6 GPU runs synced from FreeCounting (single + multi FSC-147 baselines + 4 ablation campaigns) + 1 CPU-only synthetic validation of §7.1 (adaptive α=0.50 beats best fixed-δ by 14 % mean MAE_max). See `results/index.md`.
- shared infra :: `codes/eval/_gpu_safety.py::GpuGuard` (auto-detects nvidia-smi across PATH + WSL fallbacks) is mandatory for any GPU run; `metrics.json::thermal` is a contract field.

key_findings_so_far (FSC-147 val 1/3, n≈423, seed=42):
- OCCAM-S baseline (paper P0 + FINCH + pred=total) :: MAE = 43.65, RMSE = 100.63 (run 2026-05-11-0703).
- OCCAM-M baseline                                  :: MAE = 41.98 (run 2026-05-17-1138).
- A6 mask area window (min=5e-4, max=0.10)          :: best single-mode area config, MAE = 42.94 (run 2026-05-17-0152).
- A6 + FINCH + pred=max (FSC-147-correct head)      :: MAE = 32.10 (reference inside run 2026-05-20-0942).
- A6 + SNG (ε=10, δ=6, η=0.64) + pred=max           :: MAE = 38.94, best SNG config so far (run 2026-05-20-0942).
- M6 + P7 (no project-side filter) + FINCH + max    :: MAE = 30.71, best overall (run 2026-05-21-1328).
- η ∈ [0.4, 0.55] is the empirical SNG sweet spot, fully consistent with the SNR theory in `library/notes/SNG-method.md` §6.3–§6.4.
- §7.1 adaptive δ on synthetic Gaussians (run 2026-05-27-1145, CPU-only, n ∈ {50..500}, 5 seeds) :: **adaptive α=0.50 mean MAE_max = 22.32** vs best fixed (δ=5) 25.92, worst-case 70 vs 66 — 14 % mean win + tied worst-case **without** retuning δ across n. Validates §7.1 claim end-to-end on a clean fixture; FSC-147 confirmation pending (see step 2 below).

open_questions:
- SNG vs FINCH on FSC-147 :: FINCH still ahead by ~7.5 MAE; SNG's predicted advantage is cross-dataset / cross-mode robustness — NOT YET evaluated on a second benchmark.
- MP7 anomaly :: dropping every project-side mask filter beats the paper P0 area window on multi-mode FSC-147 (val 1/3). Verified on val only; needs FSC-147 test reproduction.
- pred_strategy :: switching `total → max` accounts for ~10 MAE on FSC-147 (single-class-per-image setting); revisit when multi-class benchmarks are added.

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
- reproducibility :: every run goes into `results/<run_id>/` with `config.yaml`, `metrics.json`, `log.txt` (or `run.log`), and a `README.md` summarising configuration + headline metrics.
- gpu thermal safety :: every GPU run MUST go through `codes/eval/_gpu_safety.GpuGuard` (auto-attached via `add_cli_args(parser)` in every evaluator). Project default = `--gpu-temp-limit 78 --gpu-cooldown-sec 30 --gpu-hysteresis 5 --gpu-check-every 5`. The `thermal: {...}` block (peak_temp_c, cooldown_events, cooldown_seconds, polls) MUST land in `metrics.json` so post-mortem reviewers can see whether the run was throttled. `--gpu-guard-off` is allowed only for CPU-only smoke tests; the run README must say so explicitly.

next_concrete_steps (priority-ordered; revise as work progresses):
1. **install + smoke-test** :: `pip install -r codes/occam/requirements.txt` + SAM2 from `git+https://github.com/facebookresearch/sam2.git`; run `python codes/eval/eval_fsc147_full.py --mode single --splits val --fraction 0.333 --seed 42 --output-dir results/<new_run_id>/` and confirm MAE ≈ 43.65 against the synced 2026-05-11 baseline. The evaluator auto-attaches `GpuGuard` (default 78 °C / 30 s / 5 °C / every 5 imgs); confirm `metrics.json::thermal` is populated. Register the run in `results/index.md`.
2. **§7.1 FSC-147 confirmation (adaptive δ on real features)** :: NOW that the formula is implemented + CPU-validated, run `python codes/eval/eval_fsc147_full.py --mode single --splits val --fraction 0.333 --seed 42 --cluster-method sng --sng-epsilon 10 --sng-alpha 0.5 --pred-strategy max --min-mask-area 0.0005 --max-mask-area 0.10 --output-dir results/<new_run_id>/`. Compare against the synced `results/2026-05-20-0942-.../A6_SNG_e10_d5` (MAE 39.63). Hypothesis: adaptive α=0.5 within ±1 MAE of manual best. If the synthetic α=0.5 finding transfers, promote `OccamConfig.sng_alpha = 0.5` (record the change in a `history/` entry).
3. **MP7 anomaly verification** :: rerun MP7 / MP3 / MP6 on FSC-147 **test** split (currently only val 1/3 is measured) using OCCAM-M; if MP7 ≤ MP0, this is itself a finding worth writing up.
4. **second benchmark** :: pick CARPK or OmniCount-191 and reproduce the OCCAM baseline there. Use `codes/scripts/eval_omnicount.py` as the starting CLI; extend or fork into `codes/eval/eval_carpk.py` if needed (and call `_gpu_safety.add_cli_args(parser)` from the new evaluator).
5. **extend `synth_validate_sng.py`** with ε ∈ {5, 8, 12, 15} and SNR scan (intra_std ∈ {0.5, 0.8, 1.2}, inter_dist ∈ {3, 4, 5}) to map the full (α, ε, SNR) sweet-spot surface; cheap (CPU-seconds), informs the production `sng_alpha` default.
6. **§7.2–§7.5 SNG variants** :: prioritise §7.3 (degree-normalised local δ) for FSC-147's per-image-density variability before §7.4 (triangle reinforcement) or §7.5 (signal-noise-driven adaptive selection); each variant gets its own `history/` entry + `results/<run_id>/`. **Required regression check :: each variant must run through `synth_validate_sng.py` and not regress vs the current adaptive baseline** before being merged.
7. **paper-reading scaffold** :: run the `paper-reading` skill on `library/paper/OCCAM.pdf` and emit `library/paper/OCCAM.notes.md`; cross-reference to `library/notes/SNG-method.md`.

style_for_new_work:
- one logical change → one `history/<YYYY-MM-DD-HHMM>-<tag>.md` entry + one `results/<run_id>/` (when applicable).
- update `codes/index.md::operations:` whenever an operation surface changes.
- keep `library/notes/SNG-method.md` as the single source of truth for SNG; if you derive a new variant, add a new section there before implementing.

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
