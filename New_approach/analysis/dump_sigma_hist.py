"""Dump per-image pre-dedup blob sigma histograms for the full FSC147 test set.

Only runs the (CPU-bound) BlobCandidateGenerator -- no DINOv2 -- so it is cheap
and shardable across processes. Output feeds offline scale-layer rule sweeps.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from PIL import Image
    from pf_cud.candidates.blob_candidates import BlobCandidateGenerator

    splits = json.load(open(os.path.join(DATASET, "Train_Test_Val_FSC_147.json")))
    ann = json.load(open(os.path.join(DATASET, "annotation_FSC147_384.json")))
    img_dir = os.path.join(DATASET, "images_384_VarV2")
    names = splits[args.split][args.offset::args.stride]
    if args.limit > 0:
        names = names[:args.limit]

    blob = BlobCandidateGenerator()
    out = []
    t0 = time.time()
    for i, name in enumerate(names):
        gt = len(ann[name]["points"])
        img = np.array(Image.open(os.path.join(img_dir, name)).convert("RGB"))
        cands = blob.generate(img)
        sig = [float(c.meta["sigma"]) for c in cands if "sigma" in c.meta]
        out.append({"image": name, "gt": gt, "sigmas": sig})
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[{i+1}/{len(names)}] {name} gt={gt} blobs={len(sig)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
