"""Stage 1: SAM2 over-complete candidate proposal (design.md sec 6.1).

The proposal step is an injectable callable ``image -> List[Candidate]`` so the
pipeline and its smoke tests do not depend on SAM2 being installed. The real
SAM2 generator is provided here; tests inject synthetic candidates instead.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from ..data import Candidate

ProposalFn = Callable[[np.ndarray], List[Candidate]]


def candidates_from_sam_masks(records: List[dict]) -> List[Candidate]:
    """Convert SAM2 AutomaticMaskGenerator records into ``Candidate`` objects."""
    candidates: List[Candidate] = []
    for r in records:
        mask = np.asarray(r["segmentation"], dtype=bool)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        score = r.get("predicted_iou")
        if score is None:
            score = r.get("stability_score")
        candidates.append(
            Candidate(
                mask=mask,
                bbox=bbox,
                area=float(mask.sum()),
                source="sam2",
                source_score=None if score is None else float(score),
                meta={k: r[k] for k in ("stability_score", "predicted_iou") if k in r},
            )
        )
    return candidates


class Sam2ProposalGenerator:
    """Real SAM2 AutomaticMaskGenerator wrapper (lazy import)."""

    def __init__(self, *, model_config: str, checkpoint: str, device: str, **amg_kwargs):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        model = build_sam2(model_config, checkpoint, device=device)
        defaults = dict(
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            box_nms_thresh=0.7,
            min_mask_region_area=100,
            output_mode="binary_mask",
        )
        defaults.update(amg_kwargs)
        self._amg = SAM2AutomaticMaskGenerator(model=model, **defaults)

    def __call__(self, image: np.ndarray) -> List[Candidate]:
        records = self._amg.generate(image)
        return candidates_from_sam_masks(records)
