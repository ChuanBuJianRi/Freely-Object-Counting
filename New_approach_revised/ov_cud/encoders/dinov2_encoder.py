"""Stage 3: DINOv2 region encoder (design.md sec 7).

Encodes the per-candidate box crop into a region embedding ``z_i`` used by the
relation head and for appearance affinity. Real DINOv2 (torch.hub) is used when
available; otherwise a deterministic offline encoder is returned so the pipeline
still runs. Encoders expose ``encode(crops) -> [N, D]`` (L2-normalized rows).
"""

from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np

from ._offline import embed_crop

DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class RegionEncoder(Protocol):
    dim: int

    def encode(self, crops: List[np.ndarray]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class DeterministicRegionEncoder:
    """Offline fallback region encoder."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([embed_crop(c, self.dim, salt=b"dino") for c in crops], axis=0)


class Dinov2RegionEncoder:
    """Real DINOv2 region encoder (torch.hub, lazy)."""

    def __init__(self, model_name: str = "dinov2_vits14", device: Optional[str] = None,
                 crop_size: int = 224, batch_size: int = 64):
        import torch

        self.dim = DINOV2_DIMS.get(model_name, 384)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.crop_size = crop_size
        self.batch_size = batch_size
        self._torch = torch
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _prep(self, crop: np.ndarray):
        torch = self._torch
        import torch.nn.functional as F

        t = torch.from_numpy(np.ascontiguousarray(crop)).to(self.device)
        t = t.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        t = F.interpolate(t, size=(self.crop_size, self.crop_size), mode="bilinear",
                          align_corners=False, antialias=True)
        return ((t - self._mean) / self._std).squeeze(0)

    def encode(self, crops: List[np.ndarray]) -> np.ndarray:
        torch = self._torch
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = torch.stack([self._prep(c) for c in crops[start:start + self.batch_size]])
                feat = self.model(batch)
                if isinstance(feat, dict):
                    feat = feat.get("x_norm_clstoken", next(iter(feat.values())))
                out.append(feat.float().cpu().numpy())
        return _l2(np.concatenate(out, axis=0).astype(np.float32))


def build_region_encoder(config) -> RegionEncoder:
    """DINOv2 if available and not offline, else deterministic fallback."""
    dim = DINOV2_DIMS.get(config.dinov2_model_name, 384)
    if config.offline:
        return DeterministicRegionEncoder(dim=dim)
    try:
        return Dinov2RegionEncoder(model_name=config.dinov2_model_name,
                                   device=config.resolve_device(),
                                   crop_size=config.crop_size)
    except Exception:
        return DeterministicRegionEncoder(dim=dim)
