"""Feature extraction for candidate object masks."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

from .masks import CandidateMask


class ResNetFeatureExtractor:
    """ImageNet ResNet-50 without the final classifier."""

    def __init__(self, *, device: str, crop_size: int) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.crop_size = crop_size

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1])).to(self.device).eval()
        self.transforms = weights.transforms(crop_size=crop_size, resize_size=crop_size)

    def extract(self, image: np.ndarray, masks: list[CandidateMask]) -> np.ndarray:
        if not masks:
            return np.empty((0, 2048), dtype=np.float32)

        crops = [self._crop_object(image, candidate) for candidate in masks]
        batch = torch.stack([self.transforms(crop) for crop in crops]).to(self.device)

        with torch.inference_mode():
            features = self.model(batch).flatten(1)
        return features.cpu().numpy().astype(np.float32)

    def _crop_object(self, image: np.ndarray, candidate: CandidateMask) -> torch.Tensor:
        x0, y0, x1, y1 = candidate.bbox
        region = image[y0:y1, x0:x1].copy()
        mask = candidate.mask[y0:y1, x0:x1]
        region[~mask] = 0

        padded = resize_with_aspect_padding(region, self.crop_size)
        tensor = torch.from_numpy(padded).permute(2, 0, 1)
        return tensor


def resize_with_aspect_padding(image: np.ndarray, size: int) -> np.ndarray:
    """Resize RGB image to fit within a square canvas while preserving aspect ratio."""

    height, width = image.shape[:2]
    scale = min(size / max(height, 1), size / max(width, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas
