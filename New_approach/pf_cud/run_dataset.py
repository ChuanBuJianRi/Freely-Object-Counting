"""Dataset runner.

Reads a directory of images (and optionally a ground-truth JSON mapping
image filename -> list of per-group counts), runs PF-CUD on each image, and
reports counting metrics using Hungarian matching between predicted group
counts and ground-truth counts.

CLI has no algorithm tuning parameters.
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
from PIL import Image

from pf_cud.eval.match import match_counts
from pf_cud.eval.metrics import mae, nae, rmse, sre
from pf_cud.pipeline import PFCUDPipeline
from pf_cud.run_image import result_to_jsonable

_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def list_images(image_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in _IMG_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
        paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(set(paths))


def load_gt(gt_path: str) -> Dict[str, List[int]]:
    with open(gt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    gt: Dict[str, List[int]] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            gt[name] = [int(v) for v in value.values()]
        elif isinstance(value, (list, tuple)):
            gt[name] = [int(v) for v in value]
        else:
            gt[name] = [int(value)]
    return gt


def main():
    parser = argparse.ArgumentParser(
        description="PF-CUD dataset runner (no algorithm parameters)."
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--gt_json", default=None)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt = load_gt(args.gt_json) if args.gt_json else None
    pipeline = PFCUDPipeline(sam_model=None)

    images = list_images(args.image_dir)
    if not images:
        print(f"No images found in {args.image_dir}")
        return

    per_image_results = {}
    total_pred: List[int] = []
    total_gt: List[int] = []

    for path in images:
        name = os.path.basename(path)
        image = np.array(Image.open(path).convert("RGB"))
        result = pipeline.run(image)
        jsonable = result_to_jsonable(result)
        per_image_results[name] = jsonable

        if gt is not None and name in gt:
            pred_counts = [g["count"] for g in jsonable["groups"]]
            gt_counts = gt[name]
            matched, _, _ = match_counts(pred_counts, gt_counts)
            for pi, gj in matched:
                total_pred.append(pred_counts[pi])
                total_gt.append(gt_counts[gj])

    out_path = os.path.join(args.out_dir, "dataset_results.json")
    summary = {"results": per_image_results}

    if gt is not None and total_gt:
        summary["metrics"] = {
            "mae": mae(total_pred, total_gt),
            "rmse": rmse(total_pred, total_gt),
            "nae": nae(total_pred, total_gt),
            "sre": sre(total_pred, total_gt),
            "num_matched": len(total_gt),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(images)} images. Results saved to {out_path}")
    if "metrics" in summary:
        print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
