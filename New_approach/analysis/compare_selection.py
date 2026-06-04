#!/usr/bin/env python3
"""Compare count-selection methods on FSC-147 (no GT used except oracle).

For each image, run the candidate-generation + feature + grouping front-end once
and evaluate:
  top1     : current pipeline (rank-1 group size)            [existing method]
  largest  : size of the largest group                       [policy baseline]
  total_fg : sum of non-background group sizes               [policy baseline]
  mdl      : adaptive MDL model selection (this proposal)    [no policy]
  oracle   : group closest to GT                             [cheating upper bound]

Only `oracle` looks at GT. Everything else is unsupervised.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path("/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/New_approach")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pf_cud.candidates.blob_candidates import BlobCandidateGenerator  # noqa: E402
from pf_cud.candidates.merge_candidates import deduplicate_candidates  # noqa: E402
from pf_cud.features.color import attach_color_features  # noqa: E402
from pf_cud.features.shape import attach_shape_features  # noqa: E402
from pf_cud.features.spatial import attach_spatial_features  # noqa: E402
from pf_cud.features.visual import NullVisualExtractor, build_visual_extractor  # noqa: E402
from pf_cud.features.fusion import fused_distance  # noqa: E402
from pf_cud.graph.components import graph_to_groups  # noqa: E402
from pf_cud.graph.cut import otsu_cut_mst  # noqa: E402
from pf_cud.graph.mst import build_mst  # noqa: E402
from pf_cud.mdl.refine import mdl_merge_refinement  # noqa: E402
from pf_cud.ranking.hypothesis import rank_groups  # noqa: E402
from pf_cud.select.mdl_count import mdl_select_count  # noqa: E402

DATASET = "/home/gaoyiyang/ws_yiyang/datasets/FSC147"


def metrics(preds, gts):
    p, g = np.asarray(preds, float), np.asarray(gts, float)
    ae, se = np.abs(p - g), (p - g) ** 2
    return {"MAE": round(float(ae.mean()), 2),
            "MSE": round(float(se.mean()), 2),
            "RMSE": round(float(np.sqrt(se.mean())), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--no_visual", action="store_true")
    ap.add_argument("--out", default="select_compare.json")
    args = ap.parse_args()

    splits = json.load(open(os.path.join(DATASET, "Train_Test_Val_FSC_147.json")))
    ann = json.load(open(os.path.join(DATASET, "annotation_FSC147_384.json")))
    img_dir = os.path.join(DATASET, "images_384_VarV2")
    names = splits[args.split][args.offset :: args.stride][: args.limit]

    blob = BlobCandidateGenerator()
    ve = NullVisualExtractor() if args.no_visual else build_visual_extractor()

    rows = []
    t0 = time.time()
    for n_i, name in enumerate(names):
        gt = len(ann[name]["points"])
        img = np.array(Image.open(os.path.join(img_dir, name)).convert("RGB"))

        cand = deduplicate_candidates(blob.generate(img))
        if not cand:
            rows.append({"image": name, "gt": gt, "top1": 0, "largest": 0,
                         "total_fg": 0, "mdl": 0, "oracle": 0})
            continue
        ve.attach(img, cand)
        attach_shape_features(cand)
        attach_color_features(img, cand)
        attach_spatial_features(cand)

        # shared front-end: fused distance -> MST -> Otsu cut -> components -> MDL merge
        d = fused_distance(cand)
        groups = mdl_merge_refinement(cand, graph_to_groups(otsu_cut_mst(build_mst(d))))
        ranked = rank_groups(cand, list(groups))
        gcounts = [len(g.indices) for g in ranked]

        top1 = gcounts[0] if gcounts else 0
        largest = max(gcounts) if gcounts else 0
        fg = [len(g.indices) for g in ranked if g.group_type != "background_or_pattern"]
        total_fg = sum(fg) if fg else sum(gcounts)
        oracle = min(gcounts, key=lambda c: abs(c - gt)) if gcounts else 0
        mdl = mdl_select_count(cand)["count"]

        rows.append({"image": name, "gt": gt, "top1": top1, "largest": largest,
                     "total_fg": total_fg, "mdl": mdl, "oracle": oracle})
        if (n_i + 1) % 5 == 0 or n_i == 0:
            print(f"[{n_i+1}/{len(names)}] {name} gt={gt} top1={top1} "
                  f"largest={largest} mdl={mdl} oracle={oracle} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    json.dump(rows, open(args.out, "w"), indent=2)
    gts = [r["gt"] for r in rows]
    print(f"\n=== count-selection comparison (n={len(rows)}, {args.split}, "
          f"visual={'off' if args.no_visual else 'on'}) ===")
    print(f"{'method':>10} {'MAE':>8} {'MSE':>10} {'RMSE':>8}")
    for m in ["top1", "largest", "total_fg", "mdl", "oracle"]:
        r = metrics([x[m] for x in rows], gts)
        tag = "  <- cheating UB" if m == "oracle" else (
              "  <- proposal" if m == "mdl" else "")
        print(f"{m:>10} {r['MAE']:>8} {r['MSE']:>10} {r['RMSE']:>8}{tag}")


if __name__ == "__main__":
    main()
