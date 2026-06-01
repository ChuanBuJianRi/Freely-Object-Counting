# 2026-05-27 10:45 :: sync OCCAM baseline + ablation results from FreeCounting

type: add
scope: codes/, library/notes/, results/
author: agent
related_memory: memory/2026-05-27-1045.md
related_run: results/2026-05-11-0703-eval-occam-fsc147-baseline (and 5 more synced)

summary:
- Synced the OCCAM reimplementation, the SNG method write-up, and 6 evaluation/ablation runs from `FreeCounting/ws_yiyang/...` into the GOC repo following the existing agent-readable contract.
- After this sync, GOC has a runnable baseline (codes/occam/ + codes/eval/eval_fsc147_full.py), the (ε, δ) Shared-Neighbor Graph contribution documented end-to-end (library/notes/SNG-method.md), and 6 reproducible runs registered under results/ with config.yaml + README.md + raw metrics.
- Goal: future agents can pick up directly from the SNG sweet-spot finding (η ∈ [0.4, 0.55], best MAE 38.94) and pursue §7.1–§7.5 of SNG-method.md, or attack the open mask_policy result (MP7 / MP3 / MP6 cluster around MAE 30.7).

files_changed:
- codes/occam/{__init__,clustering,config,features,masks,pipeline,sam2_loader}.py :: added :: OCCAM baseline package, ported as-is from FreeCounting/ws_yiyang/OCCAM/occam/.
- codes/occam/{README.md,requirements.txt} :: added :: package-level docs + numpy/opencv/torch/torchvision deps (SAM2 still installed separately).
- codes/scripts/{run_occam.py,eval_omnicount.py} :: added :: ported single-image and OmniCount sanity CLIs.
- codes/eval/eval_fsc147_full.py :: added :: ported FSC-147 evaluator; PROJECT_ROOT path adjusted from `parents[2]/OCCAM` to `parents[1]` so it resolves to `codes/` and imports `codes/occam/`.
- codes/eval/aggregate_excel.py :: added :: ported `generate_excel.py` from OCCAM_experiments_series for cross-run summarisation.
- library/notes/SNG-method.md :: added :: full SNG method write-up (was `Model_innovation.md` in the workspace root).
- library/notes/freecounting-original-readme.md :: added :: snapshot of the upstream FreeCounting README.
- library/notes/occam-impl-original-readme.md :: added :: snapshot of the upstream OCCAM impl README.
- library/index.md :: modified :: registered notes/ subfolder + 3 new files.
- results/2026-05-11-0703-eval-occam-fsc147-baseline/ :: added :: OCCAM-S FSC-147 val+test baseline run (MAE val=43.65, test=45.47), config.yaml + README.md authored.
- results/2026-05-17-0152-ablation-mask-area-single/ :: added :: 8 single-mode area-window configs (A0..A7), best A6 MAE=42.94.
- results/2026-05-17-1138-eval-occam-fsc147-multi/ :: added :: OCCAM-M FSC-147 val baseline (MAE=41.98).
- results/2026-05-19-1015-ablation-mask-area-multi/ :: added :: 8 multi-mode area-window configs (M0..M7), best M6 MAE=41.58.
- results/2026-05-20-0942-ablation-clustering-sng/ :: added :: 15 single-mode + 6 multi-mode (FINCH vs SNG (ε,δ)) configs; FINCH MAE=32.10, SNG best MAE=38.94 (e10_d6, η=0.64).
- results/2026-05-21-1328-ablation-mask-policy-multi/ :: added :: 8 mask-policy configs (P0..P7), best MP7 MAE=30.71.
- results/OCCAM_experiment_results.xlsx :: added :: cross-run XLSX (synced; ~212 KiB).
- results/index.md :: modified :: appended 6 run rows + tree update + changelog.
- codes/index.md :: modified :: registered occam/, scripts/, eval/ trees and per-file purpose/operations/usage.

operations_delta:
- codes/occam/clustering.py :: added :: `thresholded_finch(features, thresholds, steady_threshold)`, `sng_cluster(features, *, epsilon, delta)`.
- codes/occam/config.py :: added :: `OccamConfig.for_mode(mode, ...)` factory with all knob overrides (`cluster_method`, `sng_epsilon/delta`, `pred_strategy`, `mask_policy`, `mask_score_thresh`, `mask_topk`, `mask_iqr_k`, area ratios, multiscale).
- codes/occam/masks.py :: added :: `apply_mask_policy(cands, *, policy, image_area, ...)` with 8 policies P0..P7; `generate_masks_with_amg(image, amg, *, ...)`; `generate_masks_with_predictor(image, predictor, *, ...)`.
- codes/occam/features.py :: added :: `ResNetFeatureExtractor.extract(image, masks)` returning (n, 2048) float32.
- codes/occam/pipeline.py :: added :: `OccamCounter.count(image)` end-to-end + `draw_result` / `read_rgb` / `write_rgb` helpers.
- codes/occam/sam2_loader.py :: added :: `build_sam2_amg(...)` + `build_sam2_predictor(...)` (SAM2 optional dep).
- codes/scripts/run_occam.py :: added :: single-image CLI (prints JSON, optional vis output).
- codes/scripts/eval_omnicount.py :: added :: OmniCount-191 quick eval CLI.
- codes/eval/eval_fsc147_full.py :: added :: FSC-147 full evaluator with `--mode`, `--splits`, `--fraction`, `--seed`, `--mask-policy`, `--cluster-method`, `--sng-epsilon`, `--sng-delta`, `--pred-strategy`, GPU temp guard, resume.
- codes/eval/aggregate_excel.py :: added :: walk results/ tree, collate metrics.json into one XLSX.

verification:
- imports were not run because SAM2 / torch are not installed in this environment; the OCCAM package is ported byte-for-byte from FreeCounting (verified via diff before sync) so behaviour is unchanged.
- evaluator path adjustment was a single-line change (`parents[2]/"OCCAM"` → `parents[1]`); no other path lookups in the ported code.

followups:
- next agent: run `pip install -r codes/occam/requirements.txt` plus SAM2 from `git+https://github.com/facebookresearch/sam2.git`, then `python codes/eval/eval_fsc147_full.py --mode single --splits val --fraction 0.333 --seed 42 --output-dir results/<new_run_id>/` to confirm metrics match the synced baseline (MAE val ≈ 43.65).
- next agent: pursue §7.1 (adaptive δ via α) of `library/notes/SNG-method.md` first — lowest implementation cost, removes the ε/δ tuning burden across datasets.
- next agent: investigate why MP7 (no filter) beats MP0 (paper area window) by ~1 MAE; replicate on FSC-147 test split.
