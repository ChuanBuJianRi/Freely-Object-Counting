"""Mask generation and post-processing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import cv2
import numpy as np


class MaskPredictor(Protocol):
    """Small protocol for SAM-like predictors used by OCCAM."""

    def set_image(self, image: np.ndarray) -> None:
        ...

    def predict(
        self,
        *,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class CandidateMask:
    """A retained object candidate mask."""

    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float = 0.0

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def canonical_grid(height: int, width: int, spacing: int) -> np.ndarray:
    """Return dense seed points over the image in ``(x, y)`` order."""

    xs = np.arange(spacing // 2, width, spacing, dtype=np.float32)
    ys = np.arange(spacing // 2, height, spacing, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if labels_count <= 1:
        return mask.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return labels == largest_label


def deduplicate_masks(
    candidates: Iterable[CandidateMask],
    *,
    iou_threshold: float,
) -> list[CandidateMask]:
    """Greedily discard masks that overlap a previously retained mask.

    Uses bbox IoU as a cheap pre-filter so we only run pixel-IoU on plausibly
    overlapping pairs. Pixel IoU is also restricted to the union bounding box
    instead of the full image to avoid O(H*W) work per comparison.
    """

    sorted_candidates = sorted(
        candidates, key=lambda item: item.area, reverse=True
    )
    retained: list[CandidateMask] = []
    for candidate in sorted_candidates:
        keep = True
        for kept in retained:
            if bbox_iou(candidate.bbox, kept.bbox) <= iou_threshold:
                continue
            if _local_mask_iou(candidate, kept) > iou_threshold:
                keep = False
                break
        if keep:
            retained.append(candidate)
    return retained


def _local_mask_iou(a: CandidateMask, b: CandidateMask) -> float:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    x0 = min(ax0, bx0)
    y0 = min(ay0, by0)
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    crop_a = a.mask[y0:y1, x0:x1]
    crop_b = b.mask[y0:y1, x0:x1]
    inter = int(np.logical_and(crop_a, crop_b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(crop_a, crop_b).sum())
    if union == 0:
        return 0.0
    return inter / union


def postprocess_raw_masks(
    raw_masks: Iterable[tuple[np.ndarray, float]],
    *,
    image_shape: tuple[int, int],
    duplicate_iou_threshold: float,
    max_mask_area_ratio: float,
    min_mask_area_ratio: float = 0.0,
) -> list[CandidateMask]:
    """Filter SAM masks into single-object candidates."""

    height, width = image_shape
    image_area = height * width
    min_area = max(1, int(round(image_area * min_mask_area_ratio)))
    max_area = int(round(image_area * max_mask_area_ratio))
    candidates: list[CandidateMask] = []

    for raw_mask, score in raw_masks:
        mask = largest_connected_component(raw_mask.astype(bool))
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue

        x0, y0, x1, y1 = bbox
        box_width = x1 - x0
        box_height = y1 - y0
        if box_width <= 1 or box_height <= 1:
            continue

        area = int(mask.sum())
        if area < min_area or area > max_area:
            continue

        candidates.append(CandidateMask(mask=mask, bbox=bbox, score=float(score)))

    return deduplicate_masks(candidates, iou_threshold=duplicate_iou_threshold)


def generate_masks_with_predictor(
    image: np.ndarray,
    predictor: MaskPredictor,
    *,
    spacing: int,
    duplicate_iou_threshold: float,
    max_mask_area_ratio: float,
    min_mask_area_ratio: float = 0.0,
    keep_top_k_per_prompt: int = 1,
) -> list[CandidateMask]:
    """Generate candidate masks by prompting SAM2 on a dense seed grid.

    Each seed point becomes one independent prompt. ``set_image`` is called
    once and ``predict`` reuses the cached image embedding for every prompt.
    Only the top-K masks (by SAM2 score) per prompt are kept to suppress the
    over-segmented sub-parts that SAM2's multimask output produces.
    """

    predictor.set_image(image)
    height, width = image.shape[:2]
    points = canonical_grid(height, width, spacing)
    if len(points) == 0:
        return []

    label_one = np.array([1], dtype=np.int32)
    raw_masks: list[tuple[np.ndarray, float]] = []
    keep_k = max(1, int(keep_top_k_per_prompt))

    for point in points:
        masks, scores, _ = predictor.predict(
            point_coords=point[None, :],
            point_labels=label_one,
            multimask_output=True,
        )
        if len(scores) == 0:
            continue
        order = np.argsort(scores)[::-1][:keep_k]
        for index in order:
            raw_masks.append((masks[index].astype(bool), float(scores[index])))

    return postprocess_raw_masks(
        raw_masks,
        image_shape=(height, width),
        duplicate_iou_threshold=duplicate_iou_threshold,
        max_mask_area_ratio=max_mask_area_ratio,
        min_mask_area_ratio=min_mask_area_ratio,
    )


def _amg_to_candidates(raw) -> list[CandidateMask]:
    """Convert AMG raw output into CandidateMask list (no filtering)."""
    candidates: list[CandidateMask] = []
    for item in raw:
        mask: np.ndarray = item["segmentation"]
        bx, by, bw, bh = item["bbox"]
        x0, y0, x1, y1 = int(bx), int(by), int(bx + bw), int(by + bh)
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            continue
        score = float(item.get("predicted_iou", 0.0))
        candidates.append(
            CandidateMask(mask=mask.astype(bool), bbox=(x0, y0, x1, y1), score=score)
        )
    return candidates


def _filter_area_window(
    cands: list[CandidateMask],
    *,
    image_area: int,
    min_ratio: float,
    max_ratio: float,
) -> list[CandidateMask]:
    min_area = max(1, int(round(image_area * min_ratio)))
    max_area = int(round(image_area * max_ratio))
    return [c for c in cands if min_area <= c.area <= max_area]


def _filter_score_thresh(
    cands: list[CandidateMask], *, tau: float
) -> list[CandidateMask]:
    return [c for c in cands if c.score >= tau]


def _filter_topk_score(
    cands: list[CandidateMask], *, k: int
) -> list[CandidateMask]:
    if k <= 0 or len(cands) <= k:
        return list(cands)
    return sorted(cands, key=lambda c: c.score, reverse=True)[:k]


def _filter_area_iqr(
    cands: list[CandidateMask], *, k: float = 1.5
) -> list[CandidateMask]:
    """Adaptive area filter via IQR: keep area in [Q1 - k*IQR, Q3 + k*IQR].

    Scale-free; no ratio tuning needed. Falls back to identity for n<5.
    """
    if len(cands) < 5:
        return list(cands)
    areas = np.array([c.area for c in cands], dtype=float)
    q1, q3 = np.percentile(areas, [25.0, 75.0])
    iqr = q3 - q1
    if iqr <= 0:
        return list(cands)
    lo = max(1.0, q1 - k * iqr)
    hi = q3 + k * iqr
    return [c for c, a in zip(cands, areas) if lo <= a <= hi]


def _filter_area_otsu(cands: list[CandidateMask]) -> list[CandidateMask]:
    """Otsu threshold on log-area histogram; keep the lower-mean bucket
    (foreground objects tend to be smaller and more numerous than background blobs).
    Falls back to identity for n<10 or degenerate distributions.
    """
    n = len(cands)
    if n < 10:
        return list(cands)
    log_areas = np.log(np.maximum(1, [c.area for c in cands]))
    hist, edges = np.histogram(log_areas, bins=min(32, max(8, n // 4)))
    total = hist.sum()
    if total == 0:
        return list(cands)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * centers)
    global_mean = cum_mean[-1] / total
    best_var, best_thr = -1.0, centers[len(centers) // 2]
    for i in range(len(hist) - 1):
        w0 = cum[i] / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = cum_mean[i] / cum[i]
        m1 = (cum_mean[-1] - cum_mean[i]) / (total - cum[i])
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_thr = var, edges[i + 1]
    kept = [c for c, la in zip(cands, log_areas) if la <= best_thr]
    return kept if len(kept) >= max(3, n // 10) else list(cands)


def _filter_score_nms(
    cands: list[CandidateMask], *, iou_threshold: float
) -> list[CandidateMask]:
    """NMS-style dedup using SAM2 score as priority instead of area."""
    sorted_c = sorted(cands, key=lambda c: c.score, reverse=True)
    retained: list[CandidateMask] = []
    for c in sorted_c:
        keep = True
        for kept in retained:
            if bbox_iou(c.bbox, kept.bbox) <= iou_threshold:
                continue
            if _local_mask_iou(c, kept) > iou_threshold:
                keep = False
                break
        if keep:
            retained.append(c)
    return retained


def apply_mask_policy(
    cands: list[CandidateMask],
    *,
    policy: str,
    image_area: int,
    min_mask_area_ratio: float = 0.0005,
    max_mask_area_ratio: float = 0.10,
    score_thresh: float = 0.85,
    topk: int = 100,
    iqr_k: float = 1.5,
    duplicate_iou_threshold: float = 0.5,
) -> list[CandidateMask]:
    """Apply one of the 8 mask filtering policies (P0..P7).

    All policies end with greedy IoU dedup (area-priority) for fair comparison,
    except P6 which uses score-priority NMS by design.
    """
    p = policy.lower()
    if p in ("p0", "baseline", "area_window"):
        out = _filter_area_window(
            cands,
            image_area=image_area,
            min_ratio=min_mask_area_ratio,
            max_ratio=max_mask_area_ratio,
        )
    elif p in ("p1", "score_thresh"):
        out = _filter_score_thresh(cands, tau=score_thresh)
    elif p in ("p2", "topk"):
        out = _filter_topk_score(cands, k=topk)
    elif p in ("p3", "area_iqr"):
        out = _filter_area_iqr(cands, k=iqr_k)
    elif p in ("p4", "area_otsu"):
        out = _filter_area_otsu(cands)
    elif p in ("p5", "score_and_area"):
        tmp = _filter_score_thresh(cands, tau=score_thresh)
        out = _filter_area_window(
            tmp,
            image_area=image_area,
            min_ratio=min_mask_area_ratio,
            max_ratio=max_mask_area_ratio,
        )
    elif p in ("p6", "score_nms"):
        # Pre-trim by area window to remove obvious garbage, then score-priority NMS.
        tmp = _filter_area_window(
            cands,
            image_area=image_area,
            min_ratio=min_mask_area_ratio,
            max_ratio=max_mask_area_ratio,
        )
        return _filter_score_nms(tmp, iou_threshold=duplicate_iou_threshold)
    elif p in ("p7", "none", "no_filter"):
        out = list(cands)
    else:
        raise ValueError(f"Unknown mask policy: {policy}")

    return deduplicate_masks(out, iou_threshold=duplicate_iou_threshold)


def generate_masks_with_amg(
    image: np.ndarray,
    amg,
    *,
    max_mask_area_ratio: float,
    min_mask_area_ratio: float = 0.0,
    policy: str = "p0",
    score_thresh: float = 0.85,
    topk: int = 100,
    iqr_k: float = 1.5,
    duplicate_iou_threshold: float = 0.5,
) -> list[CandidateMask]:
    """Generate candidate masks using ``SAM2AutomaticMaskGenerator`` and apply
    the selected post-filtering policy.

    AMG handles the dense grid, NMS, and stability filtering internally; this
    function then applies one of the 8 OCCAM-side filtering policies (P0..P7).
    """
    height, width = image.shape[:2]
    image_area = height * width
    raw = amg.generate(image)
    cands = _amg_to_candidates(raw)
    return apply_mask_policy(
        cands,
        policy=policy,
        image_area=image_area,
        min_mask_area_ratio=min_mask_area_ratio,
        max_mask_area_ratio=max_mask_area_ratio,
        score_thresh=score_thresh,
        topk=topk,
        iqr_k=iqr_k,
        duplicate_iou_threshold=duplicate_iou_threshold,
    )
