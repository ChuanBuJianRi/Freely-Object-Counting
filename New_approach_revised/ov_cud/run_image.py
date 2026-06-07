"""CLI: run OV-CUD stages 1-6 on a single image and dump coarse groups as JSON.

Examples
--------
# Offline deterministic run (no weights) but still needs candidates; uses a
# trivial grid proposal so the wiring can be exercised end-to-end:
python -m ov_cud.run_image --image path.jpg --offline --grid-proposals

# Real run (requires SAM2 + DINOv2 + open_clip configured):
python -m ov_cud.run_image --image path.jpg \
    --sam2-config <cfg> --sam2-checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np

from .config import Config
from .data import Candidate
from .pipeline import build_default_pipeline


def _load_image(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def grid_proposal_fn(image: np.ndarray, rows: int = 4, cols: int = 4) -> List[Candidate]:
    """A trivial proposal generator (rectangular tiles) for offline wiring demos."""
    h, w = image.shape[:2]
    cands: List[Candidate] = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = int(r * h / rows), int((r + 1) * h / rows)
            x1, x2 = int(c * w / cols), int((c + 1) * w / cols)
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            cands.append(Candidate(mask=mask, bbox=(x1, y1, x2, y2),
                                   area=float(mask.sum()), source="grid",
                                   source_score=1.0))
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(description="OV-CUD stages 1-6 on one image")
    ap.add_argument("--image", required=True)
    ap.add_argument("--offline", action="store_true", help="force offline backends")
    ap.add_argument("--grid-proposals", action="store_true",
                    help="use a trivial grid proposal fn instead of SAM2")
    ap.add_argument("--sam2-config")
    ap.add_argument("--sam2-checkpoint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = Config(offline=args.offline, sam2_config=args.sam2_config,
                    sam2_checkpoint=args.sam2_checkpoint)
    image = _load_image(args.image)
    pipeline = build_default_pipeline(config)
    if args.grid_proposals:
        pipeline.proposal_fn = grid_proposal_fn

    result = pipeline.run(image)
    payload = result.to_json()
    text = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
