# codes/ index (agent-readable)

PURPOSE: Index of all code under `codes/`. Each entry describes WHAT the code is and WHICH operations it supports. Agents MUST read this before running or modifying anything in `codes/`, and MUST update this file on any add/delete/rename/behavior-change inside `codes/`.

UPDATE_RULES:
- on_add_folder: append a new `## <folder>/` section with `purpose:`, `entrypoint:`, `operations:`, `files:` fields.
- on_add_file: append a new `### <relative/path>` block with `purpose:`, `operations:`, `usage:` fields.
- on_modify_behavior: update the corresponding `operations:` and/or `usage:` field; add a `changelog` entry.
- on_delete: remove the corresponding entry.
- on_rename: update path/name in place.
- field rules:
  - `purpose:` :: one line, what this code is for.
  - `operations:` :: bullet list, one verb per line, each describing a discrete action the code can perform (e.g. `- train model on dataset X`, `- run inference on a single image`, `- export checkpoint to ONNX`).
  - `entrypoint:` :: command(s) used to invoke the main behavior (e.g. `python train.py --config configs/base.yaml`).
  - `usage:` :: minimal example invocation if the file is directly runnable; omit if not runnable.
  - `inputs:` / `outputs:` :: optional, list expected paths, formats, or data shapes when relevant.
- keep keys lowercase, values concise, no prose paragraphs.

## tree
```
codes/
├── index.md                  # this index
├── deep_research.py          # OpenAI Deep Research CLI
├── requirements.txt          # deep_research deps (python-dotenv, openai)
├── occam/                    # OCCAM baseline reimplementation (synced from FreeCounting)
│   ├── __init__.py
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py             # OccamConfig dataclass (mode, mask, cluster, pred)
│   ├── pipeline.py           # OccamCounter end-to-end + draw_result helpers
│   ├── masks.py              # AMG-based mask gen + 8 post-filter policies (P0..P7)
│   ├── features.py           # ResNet-50 frozen feature extractor (224 / 500 crop)
│   ├── clustering.py         # thresholded_finch + sng_cluster (ε,δ-SNG, adaptive δ + η)
│   ├── predict.py            # prediction heads: total / max / mode_cluster_vote (MCV)
│   └── sam2_loader.py        # SAM2 AMG / Predictor build helpers
├── scripts/                  # runnable CLIs around occam/
│   ├── run_occam.py          # single-image inference + visualization
│   ├── eval_omnicount.py     # quick OmniCount-191 eval (COCO-style)
│   └── synth_validate_sng.py # CPU-only synthetic validation of §7.1 adaptive δ
└── eval/                     # benchmark evaluation entrypoints
    ├── _gpu_safety.py        # shared GPU thermal guard (mandatory for GPU runs)
    ├── eval_fsc147_full.py   # FSC-147 val/test evaluator (MAE/MSE/RMSE/NAE + buckets)
    └── aggregate_excel.py    # collate metrics.json across runs into an XLSX
```

## occam/
purpose: OCCAM baseline reimplementation; the pipeline that all GOC ablations replace or augment.
entrypoint: `python codes/scripts/run_occam.py ...` or `python codes/eval/eval_fsc147_full.py ...`
operations:
- generate class-agnostic candidate masks via SAM2 AMG (8 post-filter policies: p0 area-window, p1 score-thresh, p2 topk, p3 area-iqr, p4 area-otsu, p5 score+area, p6 score-NMS, p7 no-filter)
- extract frozen ImageNet ResNet-50 features over masked crops (mode=single → 224×224, mode=multi → 500×500)
- cluster features with `thresholded_finch` (paper-faithful, distance threshold schedule) OR `sng_cluster` (ε,δ Shared-Neighbor Graph; this project's contribution; supports adaptive δ via `delta=None, alpha`)
- compute the dimensionless health-index η for any (ε, δ, n) via `eta_health(...)` — used to qualify SNG configurations against §6.4 of `library/notes/SNG-method.md`
- predict count with `pred_strategy=total` (sum of all clusters) OR `max` (largest cluster, FSC-147 default) OR `mode_cluster_vote` / `mcv` (anchored at the largest cluster, sum every cluster within `mask_iqr_k * MAD` of the anchor's `log10(bbox_area_ratio)`; falls back to `max` when no non-singleton cluster exists; reuses `mask_iqr_k=1.5` so it adds zero new hyperparameters; see `library/notes/MCV-method.md`)
files:
- __init__.py :: re-exports OccamConfig / OccamCounter / OccamResult / predict_count / PredictTrace.
- config.py :: OccamConfig dataclass; knobs for AMG, mask filtering, clustering, prediction. SNG knobs: `sng_epsilon`, `sng_delta` (None ⇒ adaptive), `sng_alpha` (default 0.4). `pred_strategy ∈ {total, max, mode_cluster_vote, mcv}`; MCV reuses `mask_iqr_k` as its MAD multiplier.
- pipeline.py :: OccamCounter.count(image) → OccamResult; supports multiscale fallback.
- masks.py :: SAM2 AMG → CandidateMask conversion + apply_mask_policy(p0..p7) + greedy IoU dedup.
- features.py :: ResNetFeatureExtractor.extract(image, masks) → (n, 2048) float32.
- clustering.py :: thresholded_finch(features, thresholds, steady_threshold); sng_cluster(features, *, epsilon, delta=None, alpha=0.4); adaptive_delta(*, epsilon, n, alpha); eta_health(*, epsilon, delta, n).
- predict.py :: predict_count(result, strategy, *, image_shape, k=1.5) → (int, PredictTrace); single source of truth for the three prediction heads. Pure NumPy, side-effect free, callable both from pipeline and from offline trace replay.
- sam2_loader.py :: build_sam2_amg / build_sam2_predictor (SAM2 is an optional dep).

## scripts/
purpose: thin CLIs that wire `occam/` to image-level inference / quick smoke evaluation; also hosts CPU-only validators that don't need SAM2/GPU.
files:
- run_occam.py :: single-image inference; prints JSON with cluster counts and optionally writes a visualization.
  usage: `python codes/scripts/run_occam.py --image <img> --sam2-config <cfg> --sam2-checkpoint <ckpt> --mode single|multi --output <vis.jpg>`
- eval_omnicount.py :: COCO-style sanity eval on OmniCount-191; reports per-image GT vs pred + class-count MAE.
  usage: `python codes/scripts/eval_omnicount.py --coco-json _annotations.coco.json --image-dir <dir> --sam2-config <cfg> --sam2-checkpoint <ckpt> --limit 5 --output-dir <out>`
- synth_validate_sng.py :: CPU-only synthetic validation of `sng_cluster` (any δ scheme: fixed integer or adaptive α). Generates K Gaussian clusters + noise, sweeps `(n, ε, seed) × {fixed-δ, adaptive-α}`, reports ARI + counting-MAE + η. Use as a regression guard before merging any new SNG variant. Runs in CPU-seconds.
  usage: `python codes/scripts/synth_validate_sng.py --output-dir results/<run_id>/ --ns 50 100 150 250 500 --seeds 0 1 2 3 4 --epsilons 10 --fixed-deltas 1 2 3 5 7 --alphas 0.3 0.4 0.5 --k 3 --intra-std 0.8 --inter-dist 4.0`
  outputs: `metrics.json` (with `thermal: {enabled: false, reason}`) + `per_run.csv` + `summary.txt`.

## eval/
purpose: full-benchmark evaluators that produce `metrics.json` + `per_image_*.json` + `summary.txt`, ready to drop into `results/<run_id>/`.
shared_modules:
- _gpu_safety.py :: `GpuGuard` thermal-safety guard (auto-detects nvidia-smi; throttles when temp ≥ limit; records peak / cooldown into metrics.json). Every evaluator wires it via `add_cli_args(parser)` + `guard_from_args(args)`. **Project policy: every GPU run MUST keep the guard ON; `--gpu-guard-off` is allowed only for CPU-only smoke tests.**
files:
- eval_fsc147_full.py :: FSC-147 val/test full eval; supports `--mode single|multi`, `--splits val test`, `--fraction <0..1>`, `--seed`, all OccamConfig overrides (`--mask-policy`, `--cluster-method`, `--sng-epsilon`, `--sng-delta`, `--sng-alpha`, `--pred-strategy ∈ {total,max,mode_cluster_vote,mcv}`, area ratios, ...). MCV reuses `--mask-iqr-k` (default 1.5) as its MAD multiplier — no extra flag. GPU thermal guard auto-injected (`--gpu-temp-limit 78 --gpu-cooldown-sec 30 --gpu-hysteresis 5 --gpu-check-every 5` defaults; written into `metrics.json::thermal`). `per_image_<split>.json` now also records a `trace` field with cluster sizes, log-area, anchor, mode-member set, sigma — letting future prediction-head experiments replay results offline without re-running SAM2/ResNet. Built-in resume.
  usage: `python codes/eval/eval_fsc147_full.py --mode single --splits val --fraction 0.333 --seed 42 --output-dir results/<run_id>/`
- aggregate_excel.py :: walk a results-dir tree, collate every `metrics.json` into one XLSX (one sheet per ablation campaign).
  usage: `python codes/eval/aggregate_excel.py --results-root results/ --out results/summary.xlsx`

### deep_research.py
purpose: OpenAI Deep Research CLI for surveying SOTA / generating literature reports.
operations:
- submit a research question to `o4-mini-deep-research` (background mode, polled).
- save query + intermediate response.json + final report.md under `tmp/deep-research/<timestamp>/`.
usage: `python codes/deep_research.py "你的研究问题"` or `python codes/deep_research.py --file <prompt.md>`
inputs: `.env` with `OPENAI_API_KEY` (see `.env.example`); optional `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `OPENAI_DEEP_RESEARCH_MODEL`, `DEEP_RESEARCH_TIMEOUT`.
outputs: `tmp/deep-research/<YYYYMMDD-HHMMSS>/{query.md,response_id.txt,response.json,report.md}`.

## changelog
- 2026-05-26 :: init; codes/ is empty.
- 2026-05-26 1431 :: registered deep_research.py.
- 2026-05-27 1045 :: synced OCCAM baseline reimplementation from FreeCounting → codes/occam/, codes/scripts/, codes/eval/. Now exposes the full OCCAM pipeline (mask gen P0..P7 + ResNet-50 features + FINCH/SNG clustering + total/max prediction) plus FSC-147 evaluator.
- 2026-05-27 1145 :: extracted GPU thermal guard into codes/eval/_gpu_safety.py (auto-detects nvidia-smi on WSL/native; CLI flags `--gpu-temp-limit`, `--gpu-cooldown-sec`, `--gpu-hysteresis`, `--gpu-check-every`, `--gpu-index`, `--gpu-guard-off`); wired into `eval_fsc147_full.py`; thermal stats now land inside `metrics.json::thermal`.
- 2026-05-27 1145 :: added §7.1 adaptive-δ to `occam.clustering.sng_cluster` :: signature now `sng_cluster(features, *, epsilon, delta=None, alpha=0.4)` with `delta=None ⇒ adaptive_delta(epsilon, n, alpha) = floor(α(ε−1)+(1−α)ε²/n)`; added helper `adaptive_delta` and `eta_health` for direct use; OccamConfig gained `sng_alpha=0.4`, `sng_delta` default switched to `None`; eval CLI gained `--sng-alpha`. Backwards compatible (passing integer `delta` keeps legacy behaviour).
- 2026-05-27 1406 :: added MCV (Mode-Cluster-Vote) prediction head in `codes/occam/predict.py` :: `predict_count(result, strategy ∈ {total,max,mode_cluster_vote,mcv}, *, image_shape, k=1.5)`. MCV anchors at the largest cluster and sums every cluster within `k * MAD` of the anchor's `log10(bbox_area_ratio)`; `k` reuses `OccamConfig.mask_iqr_k` (default 1.5) so MCV adds zero new hyperparameters. Falls back to `max` when no non-singleton cluster exists. eval CLI: `--pred-strategy` choices extended; `per_image_<split>.json` now also stores a per-image `trace` (cluster sizes / log-area / anchor / mode-member set / sigma) for offline replay. `library/notes/MCV-method.md` documents motivation, algorithm, and 4 failure modes.
