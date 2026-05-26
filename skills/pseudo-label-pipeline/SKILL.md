---
name: pseudo-label-pipeline
description: Run the project's validated pseudo-label pipeline (SEEM mask proposals → GMM fit → NMS merge → two rounds of refinement) over an image folder. Use this skill whenever the user gives an image folder and wants pseudo-labels, masks, or counting prompts generated — ALWAYS invoke when the user says "generate pseudo labels", "run SEEM/SAM on this folder", "make masks for unlabeled images", or sets up a new dataset for the OCCAM-style training-free pipeline. Handles GPU memory, checkpointing, and resume-from-failure automatically.
---

# pseudo-label-pipeline

Encode the project's verified know-how for class-agnostic pseudo-labeling so changing dataset = changing config, not code.

## When to invoke
- user has an unlabeled image folder and needs masks / pseudo counts / candidate exemplars.
- user mentions SEEM, SAM, mask proposals, GMM, NMS, refinement rounds for counting.
- new dataset preparation step before any OCCAM-style inference.

## Required inputs
- `image_dir` :: path to image folder (recursively scanned for `.jpg/.png/.jpeg/.tif`).
- `output_dir` :: path under `results/<run_id>/pseudo/` (per `results/index.md` convention).
- `proposal_model` :: `seem` (default) | `sam` | `sam2`.
- `class_prompt` :: optional text or exemplar; if absent, run class-agnostic mode.
- `gmm_components` :: int, default 3 (used in size/feature clustering of proposals).
- `nms_iou` :: float, default 0.5.
- `refinement_rounds` :: int in {0,1,2}, default 2.
- `device` :: `cuda` | `cpu` | `auto`.
- `batch_size` :: int, default decided by VRAM probe.

## Pipeline stages (DO NOT reorder)
1. **stage_proposals** :: run SEEM/SAM over each image; write per-image `proposals.npz` (masks + scores + boxes).
2. **stage_features** :: extract per-proposal features with the project's feature backbone (default DINOv2 ViT-B); cache to `features/<image_id>.npz`.
3. **stage_gmm** :: fit `gmm_components` GMM in feature space (or feature+size space if specified); assign cluster id per proposal.
4. **stage_nms** :: per-cluster NMS at `nms_iou` to remove duplicate proposals.
5. **stage_refine_1** :: re-score surviving proposals against cluster centroid; drop bottom 10%.
6. **stage_refine_2** :: re-fit GMM on survivors and re-NMS (only if `refinement_rounds == 2`).
7. **stage_export** :: write `pseudo_labels.json` with per-image `[{box, mask_path, score, cluster_id, count_contribution}]`.

## Robustness requirements
- **GPU memory** :: auto-probe at start; reduce `batch_size` and image-side resolution stepwise on OOM; never crash silently.
- **Checkpointing** :: every stage writes `state/<stage>.done` sentinel + per-image cache; rerun MUST resume from the latest completed sentinel.
- **Resume from failure** :: support `--resume`; skip images whose per-image artifact already exists and validates.
- **Determinism** :: set torch / numpy / random seeds from `seed` field; log effective seed in `state/run.json`.
- **Failure log** :: any image that fails through all retries is appended to `state/failed.txt` with reason; pipeline keeps going.

## Outputs (under `results/<run_id>/pseudo/`)
- `proposals/` :: per-image npz.
- `features/` :: per-image npz.
- `pseudo_labels.json` :: final aggregated labels.
- `state/` :: stage sentinels, `run.json` (config snapshot, seed, device, timing), `failed.txt`.
- `vis/` :: optional, top-K and bottom-K mask overlays for sanity check.

## Constraints
- never mutate `image_dir`; read-only.
- never write outside `results/<run_id>/pseudo/`.
- log every stage to `log.txt` with timestamps and per-image timing.
- on completion, update `results/index.md` (`status=done`, `notes=pseudo-labels generated`) AND log a `history/` entry only if the pipeline code itself changed.

## Failure modes to guard
- mixed image extensions / corrupt files :: skip-with-log, never crash.
- empty proposal list for some images :: write empty entry, do not poison GMM.
- GMM degenerate covariance :: fall back to k-means with a warning.
- exemplar mode with fewer than 3 exemplars :: warn and switch to text/class-agnostic mode if `class_prompt` is also missing.
