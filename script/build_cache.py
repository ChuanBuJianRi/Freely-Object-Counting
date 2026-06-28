"""OCCAM-M 配方：预计算 SAM2 候选缓存（FSC147）。

为每张图像跑一次 OCCAM-M 风格的 SAM2 AutomaticMaskGenerator，把候选 mask 以
COCO RLE 形式落盘成每图一个 .pt，供 diag_*_dot_recall.py 与 frame_1 下游消费。

OCCAM-M 配方（与 diag_occam_dot_recall.py 完全一致）：
    - backbone : sam2.1-hiera-small（本机权重 sam2.1_hiera_small.pt）
    - config   : configs/sam2.1/sam2.1_hiera_s.yaml
    - 种子点   : 8px spacing 的自定义 point_grids（对齐 OCCAM Table 1）
    - tiling   : crop_n_layers=1
    - points_per_batch = 1000
    - AMG 阈值 : pred_iou=0.7 / stability=0.8 / stability_offset=0.0 /
                 mask_threshold=0.0 / box_nms=0.7 / crop_nms=0.7 /
                 multimask=True / use_m2m=False
    - 候选上限 : 无上限（max_candidates=0）

落盘字段对齐旧缓存：masks_rle / height / width / file_name / img_id / gt_count。
下游 frame_1 才会计算 z / bbox / purity 等，本脚本只产出候选 mask。

运行（必须用配好 GPU+sam2 的 venv）：
    /home/czp/ws_yiyang/FreeCounting/venv/bin/python script/build_cache.py \
        --out-dir /home/czp/ws_yiyang/ovcud_cache/fsc147_test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

import numpy as np
import torch
from PIL import Image

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = "/home/czp/ws_yiyang/FreeCounting/ws_yiyang/OCCAM/checkpoints/sam2.1_hiera_small.pt"
ANN_PATH = "/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json"
IMG_DIR = "/home/czp/official_code/dataset/FSC147/images_384_VarV2"


def build_occam_amg(device: str, spacing: int, points_per_batch: int, crop_n_layers: int):
    """构造 OCCAM-M 风格的 AMG（与 diag_occam_dot_recall.py 同款）。

    用 8px spacing 的归一化 point_grids 替代 points_per_side 均匀网格，
    精确对齐 OCCAM Table 1 的 "Seed-point Spacing = 8px"。
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)

    # 归一化网格：8px spacing 以 384 短边为参考换算 step，AMG 按每张图实际尺寸缩放。
    ref = 384.0
    step = spacing / ref
    coords = np.arange(step / 2.0, 1.0, step, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    point_grids = [grid]
    for _ in range(1, crop_n_layers + 1):
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


def encode_masks_rle(masks: List[np.ndarray]) -> List[dict]:
    """把二值 mask 列表编码为 COCO RLE（counts 转 str 以便 torch.save 可序列化）。"""
    from pycocotools import mask as mask_utils

    rles: List[dict] = []
    for m in masks:
        mm = np.asfortranarray(np.asarray(m).astype(np.uint8))
        rle = mask_utils.encode(mm)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        rles.append({"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts})
    return rles


def main() -> None:
    ap = argparse.ArgumentParser(description="OCCAM-M SAM2 候选缓存生成（FSC147）")
    ap.add_argument("--ann", default=ANN_PATH, help="FSC147 annotation json")
    ap.add_argument("--img-dir", default=IMG_DIR, help="FSC147 384 图像目录")
    ap.add_argument("--out-dir", required=True, help="缓存输出目录（每图一个 .pt）")
    ap.add_argument("--images", nargs="*", default=None, help="可选：仅处理这些文件名（默认全量）")
    ap.add_argument("--images-file", default=None, help="可选：从 json 列表文件读取要处理的文件名")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spacing", type=int, default=8, help="种子点间距(px, 参考384边)")
    ap.add_argument("--points-per-batch", type=int, default=1000)
    ap.add_argument("--crop-n-layers", type=int, default=1, help="tiling 层数(0=整图)")
    ap.add_argument("--max-candidates", type=int, default=0, help="候选上限，0=无上限")
    ap.add_argument("--overwrite", action="store_true", help="已存在的 .pt 也重算")
    ap.add_argument("--limit", type=int, default=-1, help="最多处理多少张（调试用）")
    args = ap.parse_args()

    ann = json.load(open(args.ann))
    if args.images_file:
        file_names = json.load(open(args.images_file))
    elif args.images:
        file_names = args.images
    else:
        file_names = sorted(ann.keys())
    if args.limit > 0:
        file_names = file_names[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[init] building OCCAM-M AMG (hiera-small) on {args.device} ...")
    t0 = time.time()
    amg = build_occam_amg(args.device, args.spacing, args.points_per_batch, args.crop_n_layers)
    print(f"[init] AMG ready in {time.time() - t0:.1f}s; {len(file_names)} images to process")

    n_done = n_skip = 0
    for fn in file_names:
        entry = ann.get(fn)
        if entry is None:
            print(f"[skip] {fn}: not in annotation")
            n_skip += 1
            continue

        img_id = os.path.splitext(fn)[0]
        out_pt = os.path.join(args.out_dir, f"{img_id}.pt")
        if os.path.exists(out_pt) and not args.overwrite:
            n_skip += 1
            continue

        img_path = os.path.join(args.img_dir, fn)
        if not os.path.exists(img_path):
            print(f"[skip] {fn}: image not found")
            n_skip += 1
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        t1 = time.time()
        raw = amg.generate(image)
        masks = [r["segmentation"] for r in raw]
        if args.max_candidates > 0:
            # 按 AMG 的 predicted_iou 降序保留上限内候选
            order = np.argsort([-float(r.get("predicted_iou", 0.0)) for r in raw])
            masks = [masks[i] for i in order[: args.max_candidates]]
        masks_rle = encode_masks_rle(masks)
        gen_s = time.time() - t1

        points = entry.get("points") or []
        torch.save(
            {
                "img_id": img_id,
                "file_name": fn,
                "gt_count": len(points),
                "masks_rle": masks_rle,
                "height": int(h),
                "width": int(w),
            },
            out_pt,
        )
        n_done += 1
        print(f"[{fn}] {len(masks_rle):>4} cand  ({gen_s:.1f}s)  -> {out_pt}")

    print(f"\n完成：写出 {n_done} 张，跳过 {n_skip} 张 -> {args.out_dir}")


if __name__ == "__main__":
    main()
