"""对比实验：OCCAM-M 配方的 SAM2 candidate dot recall（FSC147 100+ 区间）

目的
----
诊断 1 发现旧缓存（transformers pipeline + sam2.1-hiera-tiny）在 GT 100+ 区间
dot recall 仅 ~0.68。本脚本用 OCCAM 论文配方重新生成候选，验证能否提升：

    - backbone : sam2.1-hiera-small
    - 种子点    : 8px spacing 的自定义 point_grids（对齐 OCCAM Table 1）
    - AMG 阈值  : pred_iou=0.7 / stability=0.8 / stability_offset=0.0 /
                  mask_threshold=0.0 / box_nms=0.7 / multimask=True / use_m2m=False
    - tiling    : crop_n_layers=1（图像分块再分割，利于密集小目标）
    - points_per_batch = 1000

只测 "GT dot 是否被任意候选 mask 覆盖" 的 dot recall，与旧缓存基线逐图对比。

运行（必须用配好 GPU+sam2 的 venv）：
    /home/czp/ws_yiyang/FreeCounting/venv/bin/python script/diag_occam_dot_recall.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = "/home/czp/ws_yiyang/FreeCounting/ws_yiyang/OCCAM/checkpoints/sam2.1_hiera_small.pt"
ANN_PATH = "/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json"
IMG_DIR = "/home/czp/official_code/dataset/FSC147/images_384_VarV2"
OLD_CACHE = "/home/czp/ws_yiyang/ovcud_cache/fsc147_test"


def build_occam_amg(device: str, spacing: int, points_per_batch: int, crop_n_layers: int):
    """构造 OCCAM-M 风格的 AMG。

    用 8px spacing 的归一化 point_grids 替代 points_per_side 均匀网格，
    精确对齐 OCCAM Table 1 的 "Seed-point Spacing = 8px"。
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)

    # 归一化网格：8px spacing 假定在标准 384 短边附近；这里用相对坐标，
    # AMG 会按每张图实际尺寸缩放回像素。以 384 为参考边长换算 step。
    ref = 384.0
    step = spacing / ref
    coords = np.arange(step / 2.0, 1.0, step, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    point_grids = [grid]
    # crop_n_layers>0 时，AMG 需要每层一个 grid；用同一密度复制即可
    for layer in range(1, crop_n_layers + 1):
        point_grids.append(grid)

    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=None,
        point_grids=point_grids,
        points_per_batch=points_per_batch,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        stability_score_offset=0.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=0.7,
        use_m2m=False,
        multimask_output=True,
        output_mode="binary_mask",
    )


def candidate_union(masks: List[np.ndarray], h: int, w: int) -> np.ndarray:
    union = np.zeros((h, w), dtype=bool)
    for m in masks:
        mm = np.asarray(m).astype(bool)
        if mm.shape == (h, w):
            union |= mm
    return union


def dot_recall(union: np.ndarray, points: np.ndarray) -> Tuple[int, int]:
    h, w = union.shape
    covered = n = 0
    for x, y in points:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue
        n += 1
        if union[yi, xi]:
            covered += 1
    return covered, n


def old_cache_recall(file_name: str, points: np.ndarray) -> Tuple[int, int, int]:
    """旧缓存基线：返回 (covered, n_dots, n_cand)。"""
    from pycocotools import mask as mask_utils

    pt = os.path.join(OLD_CACHE, file_name.replace(".jpg", ".pt"))
    if not os.path.exists(pt):
        return -1, -1, -1
    d = torch.load(pt, map_location="cpu", weights_only=False)
    h, w = int(d["height"]), int(d["width"])
    masks = [mask_utils.decode(r).astype(bool) for r in d["masks_rle"]]
    union = candidate_union(masks, h, w)
    cov, n = dot_recall(union, points)
    return cov, n, len(masks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="FSC147 文件名列表，如 4884.jpg")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spacing", type=int, default=8, help="种子点间距(px, 参考384边)")
    ap.add_argument("--points-per-batch", type=int, default=1000)
    ap.add_argument("--crop-n-layers", type=int, default=1, help="tiling 层数(0=整图)")
    ap.add_argument("--out", default="/home/czp/official_code/result/logs/occam_dot_recall.json")
    args = ap.parse_args()

    ann = json.load(open(ANN_PATH))
    print(f"[init] building OCCAM-M AMG (hiera-small) on {args.device} ...")
    t0 = time.time()
    amg = build_occam_amg(args.device, args.spacing, args.points_per_batch, args.crop_n_layers)
    print(f"[init] AMG ready in {time.time() - t0:.1f}s")

    results = []
    agg_new = [0, 0]   # covered, dots
    agg_old = [0, 0]
    for fn in args.images:
        entry = ann.get(fn)
        if entry is None or not entry.get("points"):
            print(f"[skip] {fn}: no points")
            continue
        points = np.asarray(entry["points"], dtype=np.float64)
        gt_count = len(points)

        img_path = os.path.join(IMG_DIR, fn)
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        t1 = time.time()
        raw = amg.generate(image)
        gen_s = time.time() - t1
        masks = [r["segmentation"] for r in raw]
        union = candidate_union(masks, h, w)
        cov_new, n_new = dot_recall(union, points)

        cov_old, n_old, n_cand_old = old_cache_recall(fn, points)

        r_new = cov_new / n_new if n_new else 0.0
        r_old = cov_old / n_old if (n_old and n_old > 0) else float("nan")
        agg_new[0] += cov_new
        agg_new[1] += n_new
        if n_old > 0:
            agg_old[0] += cov_old
            agg_old[1] += n_old

        row = {
            "file": fn,
            "gt_count": gt_count,
            "img_hw": [h, w],
            "new_n_cand": len(masks),
            "new_recall": round(r_new, 4),
            "old_n_cand": n_cand_old,
            "old_recall": round(r_old, 4) if n_old > 0 else None,
            "gen_sec": round(gen_s, 1),
        }
        results.append(row)
        print(
            f"[{fn}] gt={gt_count:>4}  new: {len(masks):>4}cand recall={r_new:.3f}  "
            f"| old: {n_cand_old:>4}cand recall={r_old:.3f}  ({gen_s:.1f}s)"
        )

    print()
    if agg_new[1]:
        print(f"=== micro dot recall (这些图合计) ===")
        print(f"  OCCAM-M (new) : {agg_new[0]}/{agg_new[1]} = {agg_new[0]/agg_new[1]:.4f}")
    if agg_old[1]:
        print(f"  旧缓存  (old) : {agg_old[0]}/{agg_old[1]} = {agg_old[0]/agg_old[1]:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(
        {
            "config": {
                "spacing": args.spacing,
                "points_per_batch": args.points_per_batch,
                "crop_n_layers": args.crop_n_layers,
                "backbone": "sam2.1-hiera-small",
            },
            "micro_new": agg_new[0] / agg_new[1] if agg_new[1] else None,
            "micro_old": agg_old[0] / agg_old[1] if agg_old[1] else None,
            "rows": results,
        },
        open(args.out, "w"),
        indent=2,
        ensure_ascii=False,
    )
    print(f"\n明细写入 {args.out}")


if __name__ == "__main__":
    main()
