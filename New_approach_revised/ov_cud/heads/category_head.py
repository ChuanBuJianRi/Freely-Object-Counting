"""Stage 4: open-vocabulary category prediction head (design.md sec 8).

Design decision: the category branch uses a frozen CLIP image encoder whose
embedding is matched against frozen CLIP text prototypes. An optional trainable
projection ``W`` (initialized to identity) plus a temperature refine the
alignment -- this is what ``training/train_category.py`` learns. With no trained
projection (W = None) the head is exactly zero-shot CLIP.

    h_i        = normalize(e_i @ W)             # e_i = CLIP_image(box_crop_i)
    logit_i,c  = cosine(h_i, t_c) / temperature # t_c = CLIP_text(prompt_c)
    p_i        = softmax_c(logit_i,c)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..vocabulary import VocabularyBank


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class ClipProjectionCategoryHead:
    def __init__(self, clip_encoder, vocabulary: VocabularyBank,
                 temperature: float = 0.07, projection: Optional[np.ndarray] = None):
        self.clip = clip_encoder
        self.vocab = vocabulary
        self.temperature = float(temperature)
        self.projection = projection  # [D, D] or None (identity)
        self._prototypes: Optional[np.ndarray] = None  # [C, D]

    def build_prototypes(self) -> np.ndarray:
        """Encode vocabulary prompts into a [C, D] prototype matrix (cached)."""
        protos = _l2(self.clip.encode_text(self.vocab.prompts()))
        self.vocab.set_text_prototypes(protos)
        self._prototypes = protos
        return protos

    def prototypes(self) -> np.ndarray:
        if self._prototypes is None:
            return self.build_prototypes()
        return self._prototypes

    def _project(self, embs: np.ndarray) -> np.ndarray:
        if self.projection is not None:
            embs = embs @ self.projection
        return _l2(embs)

    def predict_from_embeddings(self, image_embs: np.ndarray) -> np.ndarray:
        """image_embs: [N, D] -> category probabilities [N, C]."""
        protos = self.prototypes()
        if image_embs.shape[0] == 0:
            return np.zeros((0, protos.shape[0]), dtype=np.float32)
        h = self._project(image_embs)
        logits = (h @ protos.T) / max(self.temperature, 1e-6)
        return _softmax(logits).astype(np.float32)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        return self.predict_from_embeddings(self.clip.encode_image(crops))

    def annotate(self, probs: np.ndarray) -> List[dict]:
        """Top-1 class name / score / countable flag per candidate."""
        names = self.vocab.class_names
        out = []
        for row in probs:
            idx = int(np.argmax(row)) if row.size else 0
            name = names[idx] if names else "unknown_object"
            out.append({
                "top_class": name,
                "top_score": float(row[idx]) if row.size else 0.0,
                "is_countable": self.vocab.is_countable(name),
            })
        return out
