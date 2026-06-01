"""Evaluate PF-CUD on the FSC147 test split.

FSC147 is a class-agnostic counting benchmark. PF-CUD is prior-free (no exemplar
boxes, no text prompt), so it outputs a set of ranked counting hypotheses. We
report two predictors:

- top1   : count of the highest-ranked group (the method's actual prediction).
- oracle : count of the group closest to GT (upper bound given the candidate
           grouping, useful to separate ranking error from grouping error).
- sum    : total number of candidates (sanity reference).

CLI exposes only engineering options (dataset paths, subset size, device).
No algorithm tuning parameters.
"""

import argparse
import json
import os
import time
from typing import List

import numpy as np
from PIL import Image

from pf_cud.eval.metrics import mae, rmse
from pf_cud.pipeline import PFCUDPipeline


def load_split(dataset_root: str, split: str) -> List[str]:
    with open(os.path.join(dataset_root, "Train_Test_Val_FSC_147.json")) as f:
        splits = json.load(f)
    return splits[split]


def load_annotations(dataset_root: str) -> dict:
    with open(os.path.join(dataset_root, "annotation_FSC147_384.json")) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="PF-CUD FSC147 evaluation.")
    parser.add_argument(
        "--dataset_root",
        default="/home/czp/ws_yiyang/FreeCounting/ws_yiyang/datasets/FSC147",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--stride", type=int, default=1, help="subsample split")
    parser.add_argument("--out_json", default="outputs/fsc147_eval.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_edge", action="store_true")
    parser.add_argument("--no_visual", action="store_true", help="skip DINOv2 (Phase 1)")
    args = parser.parse_args()

    images = load_split(args.dataset_root, args.split)
    ann = load_annotations(args.dataset_root)
    img_dir = os.path.join(args.dataset_root, "images_384_VarV2")

    images = images[:: args.stride]
    if args.limit > 0:
        images = images[: args.limit]

    pipeline = PFCUDPipeline(
        sam_model=None, use_edge=args.use_edge, use_visual=not args.no_visual
    )

    gt_counts: List[int] = []
    top1_counts: List[int] = []
    oracle_counts: List[int] = []
    per_image = []

    t0 = time.time()
    for i, name in enumerate(images):
        gt = len(ann[name]["points"])
        path = os.path.join(img_dir, name)
        image = np.array(Image.open(path).convert("RGB"))

        result = pipeline.run(image)
        group_counts = [len(g.indices) for g in result.groups]

        if group_counts:
            top1 = group_counts[0]
            oracle = min(group_counts, key=lambda c: abs(c - gt))
        else:
            top1 = 0
            oracle = 0

        gt_counts.append(gt)
        top1_counts.append(top1)
        oracle_counts.append(oracle)
        per_image.append(
            {
                "image": name,
                "gt": gt,
                "top1": top1,
                "oracle": oracle,
                "num_candidates": len(result.candidates),
                "num_groups": len(result.groups),
            }
        )

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            print(
                f"[{i + 1}/{len(images)}] {name} gt={gt} top1={top1} "
                f"oracle={oracle} cand={len(result.candidates)} "
                f"groups={len(result.groups)} ({elapsed:.1f}s)",
                flush=True,
            )

    metrics = {
        "num_images": len(images),
        "top1_mae": mae(top1_counts, gt_counts),
        "top1_rmse": rmse(top1_counts, gt_counts),
        "oracle_mae": mae(oracle_counts, gt_counts),
        "oracle_rmse": rmse(oracle_counts, gt_counts),
        "elapsed_sec": time.time() - t0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_image": per_image}, f, indent=2)

    print("\n=== FSC147", args.split, "results ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
