#!/usr/bin/env python3
"""Probe the scale (sigma) structure of blob candidates for FSC147.

Hypothesis (option C): the correct countable objects live at one/few specific
LoG scales; the over-complete giant group is a cross-scale mixture. If true,
binning raw blob centres by sigma should produce, at the "object" scale, a count
close to GT.

This dumps, per image, the list of blob (sigma, cx, cy) BEFORE dedup/grouping,
then reports for several parameter-free scale-selection rules how close the
chosen scale's blob count is to GT.

Run (CPU is fine; blob gen is CPU-bound):
  python analysis/probe_scale.py --build --stride 47 --limit 25
  python analysis/probe_scale.py --analyze
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path("/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/New_approach")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

DATASET = "/home/gaoyiyang/ws_yiyang/datasets/FSC147"
CACHE = str(PROJECT / "outputs" / "scale_probe_cache.json")


def build(stride, limit, offset):
    from PIL import Image
    from pf_cud.candidates.blob_candidates import BlobCandidateGenerator
    from pf_cud.features.visual import build_visual_extractor
    from pf_cud.features.fusion import fused_distance

    splits = json.load(open(os.path.join(DATASET, "Train_Test_Val_FSC_147.json")))
    ann = json.load(open(os.path.join(DATASET, "annotation_FSC147_384.json")))
    img_dir = os.path.join(DATASET, "images_384_VarV2")
    names = splits["test"][offset::stride][:limit]

    blob = BlobCandidateGenerator()
    ve = build_visual_extractor()
    out = []
    t0 = time.time()
    for i, name in enumerate(names):
        gt = len(ann[name]["points"])
        img = np.array(Image.open(os.path.join(img_dir, name)).convert("RGB"))
        cands = blob.generate(img)
        h, w = img.shape[:2]

        # Per-scale appearance consistency: attach visual features to all blobs,
        # then for each scale level compute the mean pairwise visual distance of
        # its members (low = members look like the same repeated thing).
        ve.attach(img, cands)
        sig = np.round(np.array([c.meta["sigma"] for c in cands], float), 4)
        levels = sorted(set(sig.tolist()))
        scale_cons = {}
        for lv in levels:
            members = [c for c, s in zip(cands, sig) if s == lv]
            if len(members) <= 1:
                scale_cons[lv] = 0.0
                continue
            d = fused_distance(members)
            iu = np.triu_indices(d.shape[0], k=1)
            md = float(d[iu].mean()) if iu[0].size else 0.0
            scale_cons[lv] = (1.0 / md) if md > 0 else float("inf")

        rec = {
            "image": name, "gt": gt, "h": h, "w": w,
            "sigmas": [float(c.meta["sigma"]) for c in cands],
            "resp": [float(c.meta["response"]) for c in cands],
            "cx": [float((c.bbox[0] + c.bbox[2]) / 2.0) for c in cands],
            "cy": [float((c.bbox[1] + c.bbox[3]) / 2.0) for c in cands],
            "scale_levels": [float(lv) for lv in levels],
            "scale_cons": [float(scale_cons[lv]) for lv in levels],
        }
        out.append(rec)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{i+1}/{len(names)}] {name} gt={gt} blobs={len(cands)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"))
    print(f"cached {len(out)} -> {CACHE}")


def analyze():
    data = json.load(open(CACHE))
    gts = np.array([d["gt"] for d in data], float)

    def mae(p):
        return float(np.abs(np.array(p, float) - gts).mean())

    def rmse(p):
        return float(np.sqrt(((np.array(p, float) - gts) ** 2).mean()))

    # Per-image: count blobs at each distinct sigma level.
    per_scale = []
    for d in data:
        s = np.array(d["sigmas"], float)
        levels = sorted(set(np.round(s, 4)))
        counts = {lv: int((np.round(s, 4) == lv).sum()) for lv in levels}
        per_scale.append(counts)

    # Show the GT vs per-scale counts for the first few images.
    print("per-image scale histogram (sigma: nblobs), gt | best-scale-count")
    for d, cs in list(zip(data, per_scale))[:25]:
        items = sorted(cs.items())
        best = min((v for v in cs.values()), key=lambda v: abs(v - d["gt"])) if cs else 0
        hist = " ".join(f"{k:.1f}:{v}" for k, v in items)
        print(f"  gt={d['gt']:>4} best={best:>4} | {hist}")

    # Selection rules over scales (parameter-free):
    # finest = smallest sigma's count; coarsest = largest sigma's count;
    # mode_scale = the scale with the most blobs; oracle_scale = closest to gt.
    finest, coarsest, mode_sc, oracle_sc = [], [], [], []
    for d, cs in zip(data, per_scale):
        if not cs:
            finest.append(0); coarsest.append(0); mode_sc.append(0); oracle_sc.append(0)
            continue
        lv = sorted(cs)
        finest.append(cs[lv[0]])
        coarsest.append(cs[lv[-1]])
        mode_sc.append(max(cs.values()))
        oracle_sc.append(min(cs.values(), key=lambda v: abs(v - d["gt"])))

    print("\n=== scale-selection rules ===")
    for nm, p in [("finest_scale", finest), ("coarsest_scale", coarsest),
                  ("mode_scale(most blobs)", mode_sc),
                  ("oracle_scale(cheat)", oracle_sc)]:
        print(f"{nm:>24} MAE={mae(p):8.2f} RMSE={rmse(p):8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--stride", type=int, default=47)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    if args.build:
        build(args.stride, args.limit, args.offset)
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()
