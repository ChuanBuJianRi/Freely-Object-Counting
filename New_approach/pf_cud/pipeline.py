"""Parameter-Free Counting Unit Discovery pipeline."""

from typing import List, Optional

import numpy as np

from pf_cud.candidates.blob_candidates import BlobCandidateGenerator
from pf_cud.candidates.edge_candidates import EdgeCandidateGenerator
from pf_cud.candidates.merge_candidates import deduplicate_candidates
from pf_cud.candidates.sam_candidates import SAMCandidateGenerator
from pf_cud.data import Candidate, CountResult
from pf_cud.features.color import attach_color_features
from pf_cud.features.fusion import fused_distance
from pf_cud.features.shape import attach_shape_features
from pf_cud.features.spatial import attach_spatial_features
from pf_cud.features.visual import build_visual_extractor
from pf_cud.graph.components import graph_to_groups
from pf_cud.graph.cut import otsu_cut_mst
from pf_cud.graph.mst import build_mst
from pf_cud.mdl.refine import mdl_merge_refinement, mdl_split_refinement
from pf_cud.ranking.hypothesis import rank_groups


class PFCUDPipeline:
    """Parameter-Free Counting Unit Discovery pipeline.

    用户只需要输入 image，不需要设置 epsilon、delta、k、IoU threshold、
    FINCH threshold 等算法参数。

    Args:
        sam_model: optional SAM/SAM2 automatic mask generator.
        visual_extractor: optional visual feature extractor. If None, one is
            built lazily on first run (DINOv2 -> ResNet50 -> Null fallback).
        use_blob: include blob candidates.
        use_edge: include edge / closed-contour candidates.
        use_mdl_split: also attempt MDL split refinement after merge.
    """

    def __init__(
        self,
        sam_model=None,
        visual_extractor=None,
        use_blob: bool = True,
        use_edge: bool = False,
        use_mdl_split: bool = False,
        use_visual: bool = True,
    ):
        self.sam_generator = (
            SAMCandidateGenerator(sam_model) if sam_model is not None else None
        )
        self.blob_generator = BlobCandidateGenerator() if use_blob else None
        self.edge_generator = EdgeCandidateGenerator() if use_edge else None
        self._visual_extractor = visual_extractor
        self.use_visual = use_visual
        self.use_mdl_split = use_mdl_split

    @property
    def visual_extractor(self):
        if self._visual_extractor is None:
            if self.use_visual:
                self._visual_extractor = build_visual_extractor()
            else:
                from pf_cud.features.visual import NullVisualExtractor

                self._visual_extractor = NullVisualExtractor()
        return self._visual_extractor

    def generate_candidates(self, image_rgb: np.ndarray) -> List[Candidate]:
        candidates: List[Candidate] = []

        if self.sam_generator is not None:
            candidates.extend(self.sam_generator.generate(image_rgb))
        if self.blob_generator is not None:
            candidates.extend(self.blob_generator.generate(image_rgb))
        if self.edge_generator is not None:
            candidates.extend(self.edge_generator.generate(image_rgb))

        # Snapshot the pre-dedup blob scale histogram: dedup merges same-scale
        # blobs and would destroy the per-scale count curve that scale-layer
        # counting reads. Stored as (sigma, count) pairs in the result meta.
        self._raw_blob_sigmas = [
            float(c.meta["sigma"])
            for c in candidates
            if c.source == "blob" and "sigma" in c.meta
        ]

        candidates = deduplicate_candidates(candidates)
        return candidates

    def attach_features(self, image_rgb: np.ndarray, candidates: List[Candidate]) -> None:
        self.visual_extractor.attach(image_rgb, candidates)
        attach_shape_features(candidates)
        attach_color_features(image_rgb, candidates)
        attach_spatial_features(candidates)

    def run(self, image_rgb: np.ndarray) -> CountResult:
        image_rgb = np.ascontiguousarray(image_rgb)
        candidates = self.generate_candidates(image_rgb)

        if len(candidates) == 0:
            return CountResult(
                groups=[],
                candidates=[],
                image_shape=image_rgb.shape[:2],
                meta={
                    "status": "no_candidates",
                    "raw_blob_sigmas": getattr(self, "_raw_blob_sigmas", []),
                },
            )

        self.attach_features(image_rgb, candidates)

        d = fused_distance(candidates)
        mst = build_mst(d)
        cut_graph = otsu_cut_mst(mst)

        groups = graph_to_groups(cut_graph)
        groups = mdl_merge_refinement(candidates, groups)
        if self.use_mdl_split:
            groups = mdl_split_refinement(candidates, groups)
            groups = mdl_merge_refinement(candidates, groups)
        groups = rank_groups(candidates, groups)

        return CountResult(
            groups=groups,
            candidates=candidates,
            image_shape=image_rgb.shape[:2],
            meta={
                "num_candidates": len(candidates),
                "num_groups": len(groups),
                "raw_blob_sigmas": getattr(self, "_raw_blob_sigmas", []),
            },
        )
