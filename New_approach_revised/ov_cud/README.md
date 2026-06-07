# OV-CUD — Open-Vocabulary Counting Unit Discovery

Implementation of `design.md` **pipeline stages 1–6** (inference skeleton) plus
the training tracks for the two learnable heads. Counting / refinement
(stages 7–9, design.md §12–14) are intentionally **out of scope** for this
milestone — the pipeline stops at *coarse semantic groups*.

## Pipeline (stages 1–6)

```
1. SAM2 Candidate Proposal      proposals/sam2_proposal.py   (injectable)
2. Candidate Canonicalization   candidates/{crops,geometry,canonicalize,filtering}.py
3. DINOv2 Region Encoding       encoders/dinov2_encoder.py
4. CLIP Category Head           heads/category_head.py + encoders/clip_encoder.py
5. Pairwise Relation Head       heads/relation_head.py + matrix/pairwise_features.py
6. Category-Aware Clustering    matrix/affinity.py + clustering/*.py
```

## Key design decisions (agreed in planning)

- **Category branch = frozen CLIP image encoder** matched to frozen CLIP text
  prototypes, with an optional **trainable projection + temperature** (Stage 1).
  With no trained projection it is exactly zero-shot CLIP.
- **Relation head**: `HeuristicRelationHead` (geometric proxies — IoU, appearance
  cosine, containment) is the default inference stub; `LearnedRelationHead` (MLP
  over `phi_ij`) drops in via the same interface once trained (Stage 2).
- **Pairwise relations are full N²** (no pruning).
- **No real training dataset is committed.** Training logic depends only on the
  GT *contract* `{mask, class_name, instance_id, bbox}` and is verified on a
  `SyntheticGTDataset`. `CocoInstanceDataset` / `LvisInstanceDataset` are
  placeholders (with notes on `iscrowd` / federated-label handling).
- Heavy backends (SAM2 / DINOv2 / CLIP) are **injectable with deterministic
  offline fallbacks**, so the pipeline and smoke tests run with no network and
  no heavy weights.

## Running

```bash
# Offline wiring demo (no weights; trivial grid proposals)
python -m ov_cud.run_image --image img.jpg --offline --grid-proposals

# Real run
python -m ov_cud.run_image --image img.jpg --sam2-config <cfg> --sam2-checkpoint <ckpt>
```

## Tests (stdlib `unittest`, no pytest needed)

```bash
python -m unittest discover -s tests -t tests -v
```

- Offline inference / matching / clustering tests run with **numpy only**.
- Training smoke tests require **torch** (skipped automatically if absent).

## Out of scope (next milestones)

Refinement (§12), instance dedup + representative selection (§13), label
aggregation (§14), real-dataset adapters + accuracy evaluation, 3-crop fusion.
