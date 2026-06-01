"""Result visualization (colors are display only, not algorithm parameters)."""

import numpy as np
from PIL import Image, ImageDraw

from pf_cud.data import CountResult

_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
]


def draw_result(image_rgb: np.ndarray, result: CountResult) -> np.ndarray:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)

    for rank, group in enumerate(result.groups):
        color = _COLORS[rank % len(_COLORS)]

        for idx in group.indices:
            cand = result.candidates[idx]
            x1, y1, x2, y2 = cand.bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"#{rank + 1}: count={len(group.indices)}, {group.group_type}"
        draw.text((8, 8 + rank * 16), label, fill=color)

    return np.array(img)
