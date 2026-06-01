"""End-to-end OCCAM counting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .clustering import Cluster, sng_cluster, thresholded_finch
from .config import OccamConfig
from .features import ResNetFeatureExtractor
from .masks import (
    CandidateMask,
    MaskPredictor,
    deduplicate_masks,
    generate_masks_with_amg,
    generate_masks_with_predictor,
)


@dataclass(frozen=True)
class OccamResult:
    """Counting result for one image."""

    clusters: list[Cluster]
    masks: list[CandidateMask]

    @property
    def counts(self) -> list[int]:
        return [len(cluster.indices) for cluster in self.clusters]

    @property
    def total_count(self) -> int:
        return sum(self.counts)

    @property
    def max_cluster_count(self) -> int:
        return max(self.counts) if self.counts else 0


class OccamCounter:
    """A paper-faithful OCCAM counter assembled from public components.

    Pass either ``amg`` (a ``SAM2AutomaticMaskGenerator``) or ``predictor``
    (a ``SAM2ImagePredictor``). AMG is strongly preferred: it produces
    higher-quality, NMS-deduplicated masks without any manual grid tuning.
    """

    def __init__(
        self,
        config: OccamConfig,
        *,
        amg=None,
        predictor: MaskPredictor | None = None,
    ) -> None:
        if amg is None and predictor is None:
            raise ValueError("Provide either 'amg' or 'predictor'.")
        self.amg = amg
        self.predictor = predictor
        self.config = config
        self.feature_extractor = ResNetFeatureExtractor(
            device=config.device,
            crop_size=config.crop_size,
        )

    def count(self, image: np.ndarray) -> OccamResult:
        """Count arbitrary object classes in one RGB image."""

        masks = self._generate_masks_multiscale(image)
        features = self.feature_extractor.extract(image, masks)
        if self.config.cluster_method == "sng":
            clusters = sng_cluster(
                features,
                epsilon=self.config.sng_epsilon,
                delta=self.config.sng_delta,
                alpha=self.config.sng_alpha,
            )
        else:
            clusters = thresholded_finch(
                features,
                thresholds=self.config.finch_thresholds,
                steady_threshold=self.config.steady_threshold,
            )
        return OccamResult(clusters=clusters, masks=masks)

    def _generate_masks_multiscale(self, image: np.ndarray) -> list[CandidateMask]:
        masks = self._generate_masks(image)
        if (
            not self.config.enable_multiscale
            or len(masks) >= self.config.min_masks_before_multiscale
            or self.config.max_multiscale_depth <= 0
        ):
            return masks

        sub_masks = self._generate_from_subimages(image)
        if len(sub_masks) > len(masks):
            return sub_masks
        return masks

    def _generate_masks(self, image: np.ndarray) -> list[CandidateMask]:
        if self.amg is not None:
            return generate_masks_with_amg(
                image,
                self.amg,
                max_mask_area_ratio=self.config.max_mask_area_ratio,
                min_mask_area_ratio=self.config.min_mask_area_ratio,
                policy=self.config.mask_policy,
                score_thresh=self.config.mask_score_thresh,
                topk=self.config.mask_topk,
                iqr_k=self.config.mask_iqr_k,
                duplicate_iou_threshold=self.config.duplicate_iou_threshold,
            )
        return generate_masks_with_predictor(
            image,
            self.predictor,
            spacing=self.config.seed_spacing,
            duplicate_iou_threshold=self.config.duplicate_iou_threshold,
            max_mask_area_ratio=self.config.max_mask_area_ratio,
            min_mask_area_ratio=self.config.min_mask_area_ratio,
            keep_top_k_per_prompt=self.config.keep_top_k_per_prompt,
        )

    def _generate_from_subimages(self, image: np.ndarray) -> list[CandidateMask]:
        height, width = image.shape[:2]
        y_edges = np.linspace(0, height, 4, dtype=int)
        x_edges = np.linspace(0, width, 4, dtype=int)
        candidates: list[CandidateMask] = []

        for row in range(3):
            for col in range(3):
                y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
                x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
                sub_image = image[y0:y1, x0:x1]
                if sub_image.size == 0:
                    continue

                for candidate in self._generate_masks(sub_image):
                    full_mask = np.zeros((height, width), dtype=bool)
                    full_mask[y0:y1, x0:x1] = candidate.mask
                    bx0, by0, bx1, by1 = candidate.bbox
                    candidates.append(
                        CandidateMask(
                            mask=full_mask,
                            bbox=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                            score=candidate.score,
                        )
                    )

        return deduplicate_masks(
            candidates,
            iou_threshold=self.config.duplicate_iou_threshold,
        )


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_result(image: np.ndarray, result: OccamResult) -> np.ndarray:
    """Draw cluster-colored boxes on an RGB image."""

    canvas = image.copy()
    palette = _palette(max(1, len(result.clusters)))

    for cluster_index, cluster in enumerate(result.clusters):
        color = palette[cluster_index]
        for mask_index in cluster.indices:
            x0, y0, x1, y1 = result.masks[mask_index].bbox
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                canvas,
                str(cluster_index),
                (x0, max(0, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return canvas


def write_rgb(path: str | Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _palette(size: int) -> list[tuple[int, int, int]]:
    values = []
    for index in range(size):
        hue = int(179 * index / max(size, 1))
        color = np.uint8([[[hue, 220, 255]]])
        rgb = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)[0, 0]
        values.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return values
