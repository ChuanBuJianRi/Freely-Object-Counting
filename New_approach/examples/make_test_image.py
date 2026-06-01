"""Generate a simple synthetic test image with repeated blobs.

Not part of the library; only a helper to exercise the pipeline.
"""

import numpy as np
from PIL import Image, ImageDraw


def make_image(path: str = "examples/test.jpg", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    h, w = 320, 320
    img = Image.new("RGB", (w, h), (245, 245, 240))
    draw = ImageDraw.Draw(img)

    # A grid of red circles (main repeated counting unit).
    for gy in range(4):
        for gx in range(4):
            cx = 50 + gx * 70 + rng.integers(-5, 5)
            cy = 50 + gy * 70 + rng.integers(-5, 5)
            r = 16
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(210, 40, 40))

    img.save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    import os

    os.makedirs("examples", exist_ok=True)
    make_image()
