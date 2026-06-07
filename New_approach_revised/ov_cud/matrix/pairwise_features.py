"""Stage 5: pairwise feature construction (design.md sec 9.3).

Builds a ``PairwiseContext`` of precomputed geometric/appearance pairwise arrays
(full N x N -- no pruning, per the design decision), and assembles the per-pair
feature vector phi_ij used by the learned relation head and by training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..data import Candidate


@dataclass
class PairwiseContext:
    n: int
    z: np.ndarray            # [N, Dz] region embeddings (L2)
    category_probs: np.ndarray  # [N, C]
    areas: np.ndarray        # [N]
    iou_mask: np.ndarray     # [N, N]
    iou_box: np.ndarray      # [N, N]
    containment: np.ndarray  # [N, N], containment[i, j] = |Mi ^ Mj| / |Mi|
    center_dist: np.ndarray  # [N, N] normalized
    scale_ratio: np.ndarray  # [N, N] in (0, 1]
    area_ratio: np.ndarray   # [N, N] = area_i / area_j
    cos_z: np.ndarray        # [N, N]
    cat_dot: np.ndarray      # [N, N] = P @ P.T

    @property
    def z_dim(self) -> int:
        return int(self.z.shape[1])


def phi_dim(z_dim: int) -> int:
    # [z_i, z_j, |z_i-z_j|, z_i*z_j] = 4*z_dim, plus 9 scalar pair features.
    return 4 * z_dim + 9


def _flatten_masks(candidates: Sequence[Candidate], image_shape) -> np.ndarray:
    h, w = image_shape[:2]
    flat = np.zeros((len(candidates), h * w), dtype=np.float32)
    for i, c in enumerate(candidates):
        flat[i] = c.mask.reshape(-1).astype(np.float32)
    return flat


def build_pairwise_context(
    candidates: List[Candidate],
    region_feats: np.ndarray,
    category_probs: np.ndarray,
    image_shape,
) -> PairwiseContext:
    n = len(candidates)
    if n == 0:
        z_dim = region_feats.shape[1] if region_feats.ndim == 2 else 0
        c = category_probs.shape[1] if category_probs.ndim == 2 else 0
        zer = np.zeros((0, 0), dtype=np.float32)
        return PairwiseContext(
            n=0, z=np.zeros((0, z_dim), np.float32),
            category_probs=np.zeros((0, c), np.float32), areas=np.zeros(0),
            iou_mask=zer, iou_box=zer, containment=zer, center_dist=zer,
            scale_ratio=zer, area_ratio=zer, cos_z=zer, cat_dot=zer,
        )

    flat = _flatten_masks(candidates, image_shape)
    areas = flat.sum(axis=1)  # [N]
    inter = flat @ flat.T      # [N, N] BLAS
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_mask = np.where(union > 0, inter / union, 0.0)
        containment = np.where(areas[:, None] > 0, inter / areas[:, None], 0.0)

    boxes = np.array([c.bbox for c in candidates], dtype=np.float32)  # [N,4]
    iou_box = _box_iou(boxes)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    h, w = image_shape[:2]
    diag = float(np.hypot(h, w)) or 1.0
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    center_dist = np.sqrt(dx * dx + dy * dy) / diag

    a = areas.copy()
    a[a == 0] = 1.0
    amax = np.maximum(a[:, None], a[None, :])
    amin = np.minimum(a[:, None], a[None, :])
    scale_ratio = amin / amax
    area_ratio = a[:, None] / a[None, :]

    z = region_feats.astype(np.float32)
    cos_z = z @ z.T  # z is L2-normalized
    cat_dot = category_probs.astype(np.float32) @ category_probs.astype(np.float32).T

    return PairwiseContext(
        n=n, z=z, category_probs=category_probs.astype(np.float32), areas=areas,
        iou_mask=iou_mask.astype(np.float32), iou_box=iou_box.astype(np.float32),
        containment=containment.astype(np.float32), center_dist=center_dist.astype(np.float32),
        scale_ratio=scale_ratio.astype(np.float32), area_ratio=area_ratio.astype(np.float32),
        cos_z=cos_z.astype(np.float32), cat_dot=cat_dot.astype(np.float32),
    )


def _box_iou(boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def build_phi_pairs(ctx: PairwiseContext, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Assemble phi_ij for the given (i, j) pairs -> [P, phi_dim]."""
    if len(pairs) == 0:
        return np.zeros((0, phi_dim(ctx.z_dim)), dtype=np.float32)
    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    zi, zj = ctx.z[ii], ctx.z[jj]
    scalars = np.stack([
        ctx.cos_z[ii, jj],
        ctx.cat_dot[ii, jj],
        ctx.iou_mask[ii, jj],
        ctx.iou_box[ii, jj],
        ctx.center_dist[ii, jj],
        ctx.scale_ratio[ii, jj],
        ctx.area_ratio[ii, jj],
        ctx.containment[ii, jj],   # containment_i_in_j
        ctx.containment[jj, ii],   # containment_j_in_i
    ], axis=1)
    return np.concatenate([zi, zj, np.abs(zi - zj), zi * zj, scalars], axis=1).astype(np.float32)


def all_ordered_pairs(n: int, include_self: bool = False) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(n) if include_self or i != j]
