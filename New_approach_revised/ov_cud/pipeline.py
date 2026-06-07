"""OV-CUD inference pipeline, stages 1-6 (design.md sec 4, 15).

Stops at coarse semantic groups. Refinement (sec 12), instance counting
(sec 13) and representative-based label aggregation (sec 14) are out of scope for
this milestone; the per-group label here is a provisional mean-probability label.

All heavy components are injected, so the pipeline runs with real backends or
with deterministic offline fallbacks (used by the smoke tests).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .candidates.canonicalize import canonicalize
from .candidates.filtering import filter_candidates
from .clustering.first_neighbor import category_aware_clustering
from .config import Config
from .data import Candidate, CoarseSemanticGroup, SemanticCountResult
from .heads.category_head import ClipProjectionCategoryHead
from .heads.relation_head import HeuristicRelationHead
from .matrix.affinity import build_group_affinity
from .matrix.pairwise_features import build_pairwise_context
from .vocabulary import NON_COUNTABLE_FALLBACK, VocabularyBank


class OvCudPipeline:
    def __init__(self, *, config, proposal_fn, region_encoder, clip_encoder,
                 category_head, relation_head, vocabulary):
        self.config = config
        self.proposal_fn = proposal_fn
        self.region_encoder = region_encoder
        self.clip_encoder = clip_encoder
        self.category_head = category_head
        self.relation_head = relation_head
        self.vocab = vocabulary

    def run(self, image: np.ndarray) -> SemanticCountResult:
        cfg = self.config

        # Stage 1: proposals + filtering.
        candidates: List[Candidate] = list(self.proposal_fn(image))
        candidates = filter_candidates(candidates, image.shape, cfg.filter)

        if not candidates:
            return SemanticCountResult(
                groups=[], candidates=[], image_shape=image.shape[:2],
                meta={"n_candidates": 0},
            )

        # Stage 2: canonicalization (crops + geometry).
        canonicalize(image, candidates, cfg)
        box_crops = [c.crops["box"] for c in candidates]

        # Stage 3: DINOv2 region encoding.
        region_feats = self.region_encoder.encode(box_crops)
        for c, z in zip(candidates, region_feats):
            c.features["region"] = z

        # Stage 4: CLIP open-vocabulary category head.
        image_embs = self.clip_encoder.encode_image(box_crops)
        category_probs = self.category_head.predict_from_embeddings(image_embs)
        annotations = self.category_head.annotate(category_probs)
        for c, p, ann in zip(candidates, category_probs, annotations):
            c.predictions["category_probs"] = p
            c.predictions.update(ann)

        # Stage 5: pairwise relations (full N^2).
        ctx = build_pairwise_context(candidates, region_feats, category_probs, image.shape)
        relations = self.relation_head(ctx)

        # Stage 6: category-aware clustering on A_group.
        a_group = build_group_affinity(category_probs, relations["A_sem"])
        groups_idx = category_aware_clustering(category_probs, a_group, cfg.cluster)

        groups = [self._build_group(g, category_probs) for g in groups_idx]
        groups.sort(key=lambda g: g.n_candidates, reverse=True)

        return SemanticCountResult(
            groups=groups,
            candidates=candidates,
            image_shape=image.shape[:2],
            meta={
                "n_candidates": len(candidates),
                "n_groups": len(groups),
                "relation_head": type(self.relation_head).__name__,
                "stage": "coarse_groups",
            },
        )

    def _build_group(self, indices: List[int], category_probs: np.ndarray) -> CoarseSemanticGroup:
        names = self.vocab.class_names
        mean_p = category_probs[indices].mean(axis=0)
        top = int(np.argmax(mean_p))
        class_name = names[top]
        is_countable = self.vocab.is_countable(class_name) and class_name not in NON_COUNTABLE_FALLBACK
        dist = {names[c]: float(mean_p[c]) for c in np.argsort(mean_p)[::-1][:5]}
        return CoarseSemanticGroup(
            class_name=class_name,
            candidate_indices=sorted(indices),
            n_candidates=len(indices),
            confidence=float(mean_p[top]),
            is_countable=is_countable,
            class_distribution=dist,
        )


def build_default_pipeline(
    config: Optional[Config] = None, vocabulary: Optional[VocabularyBank] = None
) -> OvCudPipeline:
    """Construct a pipeline with real backends (offline fallbacks if unavailable).

    The proposal step requires SAM2; when not configured/installed it raises a
    clear error on use. Tests inject a synthetic proposal_fn instead.
    """
    from .encoders.clip_encoder import build_clip_encoder
    from .encoders.dinov2_encoder import build_region_encoder
    from .proposals.sam2_proposal import Sam2ProposalGenerator

    config = config or Config()
    vocabulary = vocabulary or VocabularyBank()

    region_encoder = build_region_encoder(config)
    clip_encoder = build_clip_encoder(config)
    category_head = ClipProjectionCategoryHead(
        clip_encoder, vocabulary, temperature=config.head.category_temperature
    )
    category_head.build_prototypes()
    relation_head = HeuristicRelationHead()

    def _proposal_fn(image):
        if config.sam2_config is None or config.sam2_checkpoint is None:
            raise RuntimeError(
                "SAM2 not configured. Set config.sam2_config / sam2_checkpoint, "
                "or inject a custom proposal_fn (e.g. for testing)."
            )
        gen = Sam2ProposalGenerator(
            model_config=config.sam2_config,
            checkpoint=config.sam2_checkpoint,
            device=config.resolve_device(),
        )
        return gen(image)

    return OvCudPipeline(
        config=config, proposal_fn=_proposal_fn, region_encoder=region_encoder,
        clip_encoder=clip_encoder, category_head=category_head,
        relation_head=relation_head, vocabulary=vocabulary,
    )
