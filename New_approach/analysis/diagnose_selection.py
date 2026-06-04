#!/usr/bin/env python3
"""Offline diagnosis: which *non-cheating* group-selection rule best recovers
the FSC-147 count from PF-CUD's candidate groups?

For each image we run the full PF-CUD pipeline once and dump, for every output
group, its size (count), per-cue raw scores, total mask-area fraction and the
ranking position. We then evaluate several selection strategies that DO NOT
look at GT:

  top1          : current method -> count of the rank-1 group
  largest       : count of the group with the most members
  total_fg      : sum of counts over non-background groups
  repeat_only   : pick the group maximizing repeatability (=count) directly
  area_dominant : pick the group with the largest total mask-area fraction
  count_otsu_hi : Otsu-split the group-count distribution, sum the high cluster

and compare them against:
  oracle        : count of the group closest to GT (cheating upper bound)

Usage:
  python diagnose_selection.py --limit 40 --stride 30 --out dump.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

PROJECT = Path("/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/New_approach")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pf_cud.pipeline import PFCUDPipeline  # noqa: E402

DATASET = "/home/gaoyiyang/ws_yiyang/datasets/FSC147"


def dump_image(pipeline, image, gt):
    result = pipeline.run(image)
    h, w = result.image_shape
    groups = []
    for rank, g in enumerate(result.groups):
        area = sum(int(result.candidates[i].mask.sum()) for i in g.indices) / float(h * w)
        groups.append(
            {
                "rank": rank,
                "count": len(g.indices),
                "area_frac": area,
                "group_type": g.group_type,
                "score": g.score,
                "raw": g.meta.get("raw_scores", {}),
            }
        )
    return {"gt": gt, "num_candidates": len(result.candidates), "groups": groups}


# --- selection strategies (none look at gt except oracle) ---
def sel_top1(groups):
    return groups[0]["count"] if groups else 0


def sel_largest(groups):
    return max((g["count"] for g in groups), default=0)


def sel_total_fg(groups):
    fg = [g["count"] for g in groups if g["group_type"] != "background_or_pattern"]
    return sum(fg) if fg else sum(g["count"] for g in groups)


def sel_repeat_only(groups):
    if not groups:
        return 0
    return max(groups, key=lambda g: g["count"])["count"]


def sel_area_dominant(groups):
    if not groups:
        return 0
    return max(groups, key=lambda g: g["area_frac"])["count"]


def sel_count_otsu_hi(groups):
    counts = np.array([g["count"] for g in groups], dtype=float)
    if counts.size == 0:
        return 0
    if counts.size == 1 or np.unique(counts).size == 1:
        return int(counts.sum())
    tau = threshold_otsu(counts)
    hi = counts[counts >= tau]
    return int(hi.sum()) if hi.size else int(counts.max())


def sel_oracle(groups, gt):
    if not groups:
        return 0
    return min((g["count"] for g in groups), key=lambda c: abs(c - gt))


STRATS = {
    "top1": lambda g, gt: sel_top1(g),
    "largest": lambda g, gt: sel_largest(g),
    "total_fg": lambda g, gt: sel_total_fg(g),
    "repeat_only": lambda g, gt: sel_repeat_only(g),
    "area_dominant": lambda g, gt: sel_area_dominant(g),
    "count_otsu_hi": lambda g, gt: sel_count_otsu_hi(g),
    "oracle": lambda g, gt: sel_oracle(g, gt),
}


def metrics(preds, gts):
    p, g = np.asarray(preds, float), np.asarray(gts, float)
    ae = np.abs(p - g)
    se = (p - g) ** 2
    return {
        "MAE": round(float(ae.mean()), 2),
        "MSE": round(float(se.mean()), 2),
        "RMSE": round(float(np.sqrt(se.mean())), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--no_visual", action="store_true")
    ap.add_argument("--out", default="dump.json")
    args = ap.parse_args()

    splits = json.load(open(os.path.join(DATASET, "Train_Test_Val_FSC_147.json")))
    ann = json.load(open(os.path.join(DATASET, "annotation_FSC147_384.json")))
    img_dir = os.path.join(DATASET, "images_384_VarV2")
    names = splits[args.split][args.offset :: args.stride][: args.limit]

    pipeline = PFCUDPipeline(sam_model=None, use_edge=False, use_visual=not args.no_visual)

    dump = []
    t0 = time.time()
    for i, name in enumerate(names):
        gt = len(ann[name]["points"])
        image = np.array(Image.open(os.path.join(img_dir, name)).convert("RGB"))
        rec = dump_image(pipeline, image, gt)
        rec["image"] = name
        dump.append(rec)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{i+1}/{len(names)}] {name} gt={gt} "
                  f"groups={len(rec['groups'])} ({time.time()-t0:.0f}s)", flush=True)

    json.dump(dump, open(args.out, "w"), indent=2)

    gts = [r["gt"] for r in dump]
    print(f"\n=== selection strategy comparison (n={len(dump)}, {args.split}) ===")
    print(f"{'strategy':>16} {'MAE':>8} {'MSE':>10} {'RMSE':>8}")
    results = {}
    for name, fn in STRATS.items():
        preds = [fn(r["groups"], r["gt"]) for r in dump]
        m = metrics(preds, gts)
        results[name] = m
        print(f"{name:>16} {m['MAE']:>8} {m['MSE']:>10} {m['RMSE']:>8}")
    json.dump({"n": len(dump), "results": results}, open(args.out + ".summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
