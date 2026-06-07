"""Stage 4 backbone: CLIP image + text encoder (design.md sec 5, 8).

The category branch uses a CLIP image encoder (naturally aligned with CLIP text
prototypes -- see the design decision in the plan). Real open_clip is used when
available; otherwise a deterministic offline encoder is returned. Encoders
expose ``encode_image(crops) -> [N, D]`` and ``encode_text(texts) -> [C, D]``
with L2-normalized rows.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np

from ._offline import embed_crop, embed_text

CLIP_DIMS = {
    "ViT-B-32": 512,
    "ViT-B-16": 512,
    "ViT-L-14": 768,
}


class ClipEncoder(Protocol):
    dim: int

    def encode_image(self, crops: List[np.ndarray]) -> np.ndarray: ...
    def encode_text(self, texts: List[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class DeterministicClipEncoder:
    """Offline fallback CLIP encoder (deterministic, no semantics)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode_image(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([embed_crop(c, self.dim, salt=b"clip-img") for c in crops], axis=0)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([embed_text(t, self.dim, salt=b"clip-txt") for t in texts], axis=0)


class OpenClipEncoder:
    """Real open_clip image/text encoder (lazy)."""

    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 device: Optional[str] = None, crop_size: int = 224):
        import open_clip
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.dim = self.model.text_projection.shape[1] if hasattr(
            self.model, "text_projection") else CLIP_DIMS.get(model_name, 512)

    def encode_image(self, crops: List[np.ndarray]) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = torch.stack(
            [self.preprocess(Image.fromarray(c.astype(np.uint8))) for c in crops]
        ).to(self.device)
        with torch.inference_mode():
            feat = self.model.encode_image(batch).float().cpu().numpy()
        return _l2(feat.astype(np.float32))

    def encode_text(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            feat = self.model.encode_text(tokens).float().cpu().numpy()
        return _l2(feat.astype(np.float32))


def build_clip_encoder(config) -> ClipEncoder:
    dim = CLIP_DIMS.get(config.clip_model_name, 512)
    if config.offline:
        return DeterministicClipEncoder(dim=dim)
    try:
        return OpenClipEncoder(model_name=config.clip_model_name,
                               pretrained=config.clip_pretrained,
                               device=config.resolve_device(),
                               crop_size=config.crop_size)
    except Exception:
        return DeterministicClipEncoder(dim=dim)
