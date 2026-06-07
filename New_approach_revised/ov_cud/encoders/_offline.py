"""Deterministic, network-free embedding helpers for offline runs / tests.

Embeddings are reproducible (seeded from a stable content hash) and L2
normalized. They carry no real semantics -- they exist so the full pipeline and
its smoke tests run with no SAM2 / DINOv2 / CLIP weights and no network.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _stable_seed(data: bytes) -> int:
    digest = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(digest, "little", signed=False)


def deterministic_vector(data: bytes, dim: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(data))
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _crop_signature(crop: np.ndarray) -> bytes:
    """A small, resize-robust signature of an image crop."""
    arr = np.asarray(crop)
    if arr.ndim == 3:
        # mean-pool to an 8x8x3 thumbnail so similar crops hash similarly-ish.
        h, w = arr.shape[:2]
        gh, gw = max(1, h // 8), max(1, w // 8)
        thumb = arr[: gh * 8, : gw * 8].reshape(8, gh, -1, gw, arr.shape[2])
        thumb = thumb.mean(axis=(1, 3)).astype(np.uint8)
    else:
        thumb = arr.astype(np.uint8)
    return thumb.tobytes()


def embed_crop(crop: np.ndarray, dim: int, salt: bytes = b"") -> np.ndarray:
    return deterministic_vector(salt + _crop_signature(crop), dim)


def embed_text(text: str, dim: int, salt: bytes = b"") -> np.ndarray:
    return deterministic_vector(salt + text.encode("utf-8"), dim)
