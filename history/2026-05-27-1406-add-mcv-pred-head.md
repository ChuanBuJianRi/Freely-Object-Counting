# 2026-05-27 14:06 :: add MCV (Mode-Cluster-Vote) prediction head

type: add
scope: codes/occam/, codes/eval/eval_fsc147_full.py, library/notes/
author: agent
related_memory: none
related_run: results/<pending>-eval-mp7-mcv  (to be created in next step)

summary:
- Adds a third prediction head, "mode_cluster_vote" (alias "mcv"), as a
  drop-in replacement for OCCAM's pred_strategy in {total, max}.
- Designed to fix OCCAM-MP7's catastrophic under-counting on the FSC-147
  201+ bucket (n=23, MAE=312.96) caused by `pred=max` only returning the
  largest cluster while the query class is fragmented across multiple
  same-scale clusters. MCV anchors at the largest cluster and sums every
  cluster within k*MAD of the anchor's log10-area-ratio, with k=1.5
  reused verbatim from OccamConfig.mask_iqr_k -> introduces ZERO new
  hyperparameters.
- Training-free, parameter-light, backward compatible: total/max remain
  exact and unchanged; falls back to `max` whenever no non-singleton
  cluster exists.
- Side benefit: per-image evaluator JSON now also records the cluster
  trace (sizes, log-areas, anchor, mode-member set, sigma) so future
  prediction-head experiments do not require re-running SAM2/ResNet.

files_changed:
- codes/occam/predict.py :: added :: pure-NumPy module exposing predict_count(result, strategy, image_shape, k) -> (pred, PredictTrace); single source of truth for the three prediction heads.
- codes/occam/__init__.py :: modified :: re-export predict_count + PredictTrace.
- codes/occam/config.py :: modified :: extended pred_strategy doc to cover mode_cluster_vote/mcv; clarified that MCV reuses mask_iqr_k as the MAD multiplier (no new field).
- codes/eval/eval_fsc147_full.py :: modified :: --pred-strategy gained {mode_cluster_vote, mcv}; run_split now delegates count reduction to predict_count and threads mcv_k=mask_iqr_k through; per_image_<split>.json now stores trace dicts alongside pred/gt for offline replay.
- library/notes/MCV-method.md :: added :: write-up (motivation, algorithm, invariants, expected per-bucket effect, 4 failure modes, relation to SNG, reproduction command).

operations_delta:
- codes/occam/ :: changed :: pipeline now exposes a swappable prediction head; "mode_cluster_vote" reduction registered.
- codes/eval/eval_fsc147_full.py :: changed :: gained --pred-strategy={mode_cluster_vote,mcv}; per-image JSON schema gained "trace" field.

verification:
- offline sanity test (no SAM2, no GPU) on 5 hand-built OccamResult fixtures:
  - S1 (1 dominant cluster of 8 + 1 huge noise singleton): max=8, MCV=8 (singleton rejected) ✓
  - S2 (3 same-scale clusters of 5 + 2 odd singletons; FSC-147 201+ archetype): total=17, max=5, MCV=15 (mode = clusters 0,1,2; sigma=0) ✓
  - S3 (only singletons): MCV falls back to max=1 (strategy reports "mode_cluster_vote->max(fallback)") ✓
  - S4 (empty clusters): MCV=0 ✓
  - S5 (one large + one tiny non-singleton): MCV=6=max (small cluster's log-area outside k*MAD; sigma=0.602) ✓
- ruff/lint clean on all modified files (no diagnostics).
- shell command:
  /home/czp/ws_yiyang/FreeCounting/venv/bin/python -c "<inline 5-scenario script>"
  -> ALL OK

followups:
- run results/<pending>-eval-mp7-mcv against FSC-147 val (fraction=1/3, seed=42), same upstream pipeline as MP7 (OCCAM-M + min=5e-4/max=0.10 + p7 + finch + pred=max baseline) but with --pred-strategy mode_cluster_vote; success criterion = MAE <= 30.71 AND 201+ bucket MAE <= 250.
- if MCV beats MP7, follow up with composability runs: (1) MCV + SNG-adaptive (alpha=0.4), (2) MCV + density-adaptive multiscale; see library/notes/MCV-method.md §7.
