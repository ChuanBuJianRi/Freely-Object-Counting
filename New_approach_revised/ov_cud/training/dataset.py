"""GT data contract + synthetic dataset + concrete-adapter placeholders.

The training logic (matching, labels, sampling, losses) depends only on this
contract, so swapping datasets is a thin adapter change:

    GTInstance  = { mask, class_name, instance_id, bbox }
    ImageSample = { image, instances }

Concrete COCO / LVIS adapters are placeholders -- they must also decide how to
handle dataset-specific annotation semantics (see notes on each).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass
class GTInstance:
    mask: np.ndarray      # bool [H, W]
    class_name: str
    instance_id: int
    bbox: BBox


@dataclass
class ImageSample:
    image: np.ndarray             # uint8 [H, W, 3]
    instances: List[GTInstance] = field(default_factory=list)


class GTInstanceDataset:
    """Abstract iterable of ImageSample."""

    def __iter__(self) -> Iterator[ImageSample]:  # pragma: no cover - interface
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


def _bbox_of(mask: np.ndarray) -> BBox:
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


class SyntheticGTDataset(GTInstanceDataset):
    """Procedurally generated images with known instance masks/classes/ids.

    Each image contains several solid rectangles of a few "classes" (encoded by
    base color), so candidate<->GT matching and relation labels are well-defined
    with no external data. Deterministic given ``seed``.
    """

    CLASS_COLORS = {
        "apple": (220, 40, 40),
        "orange": (240, 150, 30),
        "car": (40, 80, 220),
    }

    def __init__(self, n_images: int = 6, image_size: int = 96, seed: int = 0):
        self.n_images = n_images
        self.image_size = image_size
        self.seed = seed
        self._samples = [self._make(i) for i in range(n_images)]

    def __iter__(self) -> Iterator[ImageSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return self.n_images

    def _make(self, idx: int) -> ImageSample:
        rng = np.random.default_rng(self.seed * 1000 + idx)
        s = self.image_size
        image = np.full((s, s, 3), 30, dtype=np.uint8)
        instances: List[GTInstance] = []
        classes = list(self.CLASS_COLORS.keys())
        n_inst = int(rng.integers(2, 5))
        inst_id = 0
        for _ in range(n_inst):
            cls = classes[int(rng.integers(0, len(classes)))]
            color = np.array(self.CLASS_COLORS[cls], dtype=np.uint8)
            bw = int(rng.integers(s // 8, s // 4))
            bh = int(rng.integers(s // 8, s // 4))
            x1 = int(rng.integers(0, s - bw - 1))
            y1 = int(rng.integers(0, s - bh - 1))
            mask = np.zeros((s, s), dtype=bool)
            mask[y1:y1 + bh, x1:x1 + bw] = True
            # paint with mild noise so crops differ
            noise = rng.integers(-15, 15, size=(bh, bw, 3))
            patch = np.clip(color.astype(int) + noise, 0, 255).astype(np.uint8)
            image[y1:y1 + bh, x1:x1 + bw] = patch
            instances.append(GTInstance(mask=mask, class_name=cls,
                                        instance_id=inst_id, bbox=_bbox_of(mask)))
            inst_id += 1
        return ImageSample(image=image, instances=instances)

    @property
    def class_names(self) -> List[str]:
        return list(self.CLASS_COLORS.keys())


class CocoInstanceDataset(GTInstanceDataset):
    """Placeholder COCO adapter.

    TODO(dataset): map COCO annotations to GTInstance. Must decide handling of:
      - ``iscrowd`` regions (crowd masks are not single instances) -> exclude
        from instance ids / treat as ignore so they do not pollute valid_i.
      - overlapping masks -> resolve which instance owns a pixel for purity.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("COCO adapter not implemented (see TODO notes).")


class LvisInstanceDataset(GTInstanceDataset):
    """Placeholder LVIS adapter.

    TODO(dataset): LVIS is federated / not-exhaustively annotated. "Different
    class" negatives may be false negatives for un-annotated categories; the
    label generator must consult per-image positive/negative category lists
    before treating a pair as a true negative.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LVIS adapter not implemented (see TODO notes).")
