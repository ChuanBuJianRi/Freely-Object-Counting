"""Vocabulary bank (design.md sec 5).

Three tiers:
  - countable  : produce final count groups
  - auxiliary  : parts / textures / background patterns, not counted directly
  - fallback   : unknown / pattern / background / noise

The bank is pure data; turning class names into text prototypes is the job of
the CLIP text encoder (see encoders/clip_encoder.py), which fills a cache here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

DEFAULT_COUNTABLE: List[str] = [
    "apple", "orange", "banana", "person", "car", "bottle", "cup", "bowl",
    "book", "cell", "screw", "brick", "tile", "egg", "coin", "ball",
]

DEFAULT_AUXILIARY: List[str] = [
    "leaf part", "wheel part", "background texture", "stripe", "grid",
]

# fallback classes always present
UNKNOWN_OBJECT = "unknown_object"
UNKNOWN_PATTERN = "unknown_repeated_pattern"
BACKGROUND = "background"
NOISE = "noise"
DEFAULT_FALLBACK: List[str] = [UNKNOWN_OBJECT, UNKNOWN_PATTERN, BACKGROUND, NOISE]

# Class names that, even if scored highest, should never produce a count group.
NON_COUNTABLE_FALLBACK = {BACKGROUND, NOISE}


@dataclass
class VocabularyBank:
    countable: List[str] = field(default_factory=lambda: list(DEFAULT_COUNTABLE))
    auxiliary: List[str] = field(default_factory=lambda: list(DEFAULT_AUXILIARY))
    fallback: List[str] = field(default_factory=lambda: list(DEFAULT_FALLBACK))
    prompt_template: str = "a photo of a {}"

    # Filled by the text encoder; class_name -> unit text embedding.
    _text_prototypes: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def class_names(self) -> List[str]:
        # countable first, then auxiliary, then fallback; de-duplicated, ordered.
        seen: Dict[str, None] = {}
        for name in [*self.countable, *self.auxiliary, *self.fallback]:
            seen.setdefault(name, None)
        return list(seen.keys())

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def index_of(self, name: str) -> int:
        return self.class_names.index(name)

    def is_countable(self, name: str) -> bool:
        return name in set(self.countable)

    def prompts(self) -> List[str]:
        return [self.prompt_template.format(name) for name in self.class_names]

    # --- text prototype cache -------------------------------------------------
    def set_text_prototypes(self, matrix: np.ndarray) -> None:
        names = self.class_names
        if matrix.shape[0] != len(names):
            raise ValueError(
                f"prototype matrix rows ({matrix.shape[0]}) != num_classes "
                f"({len(names)})"
            )
        self._text_prototypes = {name: matrix[i] for i, name in enumerate(names)}

    def text_prototype_matrix(self) -> Optional[np.ndarray]:
        if not self._text_prototypes:
            return None
        return np.stack([self._text_prototypes[n] for n in self.class_names], axis=0)

    @classmethod
    def for_classes(cls, countable: List[str], **kwargs) -> "VocabularyBank":
        """Build a bank for an explicit countable list (e.g. FSC147 classes)."""
        return cls(countable=list(countable), **kwargs)
