"""Single-image runner. CLI has no algorithm tuning parameters."""

import argparse
import json
import os

import numpy as np
from PIL import Image

from pf_cud.data import CountResult
from pf_cud.pipeline import PFCUDPipeline
from pf_cud.visualize.draw import draw_result


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def result_to_jsonable(result: CountResult) -> dict:
    out = []
    for rank, g in enumerate(result.groups):
        out.append(
            {
                "rank": rank + 1,
                "count": len(g.indices),
                "group_type": g.group_type,
                "confidence": g.confidence,
                "candidate_indices": g.indices,
                "score": g.score,
                "meta": _to_jsonable(g.meta),
            }
        )
    return {
        "image_shape": list(result.image_shape),
        "num_candidates": len(result.candidates),
        "groups": out,
        "meta": _to_jsonable(result.meta),
    }


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="PF-CUD single image counting (no algorithm parameters)."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_json", default="result.json")
    parser.add_argument("--out_vis", default="result.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    image = load_image(args.image)

    # sam_model 可以后续注入。第一版没有 SAM 也可以只跑 blob + graph。
    pipeline = PFCUDPipeline(sam_model=None)

    result = pipeline.run(image)

    jsonable = result_to_jsonable(result)

    out_json_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(jsonable, f, indent=2, ensure_ascii=False)

    vis = draw_result(image, result)
    out_vis_dir = os.path.dirname(os.path.abspath(args.out_vis))
    os.makedirs(out_vis_dir, exist_ok=True)
    Image.fromarray(vis).save(args.out_vis)

    print(json.dumps(jsonable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
