"""离线预计算 pipeline（OV-CUD §10 / 2_training_plan.md §1.2、§6）。

对每张训练图执行一次，缓存结果，避免每个 epoch 重复跑 SAM2 / DINOv2：

    1. SAM2 生成过完备候选 {M_i}        (proposals.SAM2Proposer)
    2. 轻量过滤明显噪声                  (内置于 SAM2Proposer，可选再过 filtering)
    3. 三路 crop + geometry             (candidates.build_three_crops / geometry_features)
    4. DINOv2 编码三路并拼接 z_i        (encoders.DINOv2RegionEncoder)
    5. 候选-GT 匹配 -> purity/coverage/valid/matched_class/...  (candidates.match_candidates)
    6. 缓存到磁盘 per-image .pt

输出的 .pt 字段与 training.train_category.CachedCandidateDataset 对齐：
    z              : [N, REGION_DIM] float
    matched_class  : [N] long   （background 候选置 0 但 valid=0，不参与正样本）
    purity         : [N] float
    coverage       : [N] float
    valid          : [N] long
    is_countable   : [N] float  （matched_class 是否 countable；COCO 全 thing -> 1）
    geometry       : [N, 8] float
    class_name     : str        （该图主导类别名，供 split_by_class 按类留出）

同时把文本原型（vocabulary bank）写到 prototypes_path，供训练加载。

用法：
    python -m frame_1.training.preprocess \
        --dataset_root /path/to/dataset --split val2017 \
        --cache_dir /path/to/cache --max_images 30 --device cpu

不带数据时无法运行（需真实 COCO + 模型权重）；可用 --dry_run 仅校验依赖导入。
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..candidates.crops import build_three_crops
from ..candidates.geometry import geometry_features
from ..candidates.matching import match_candidates, stack_match_labels
from ..config import PreprocessConfig, REGION_DIM, Sam2Config


def _dominant_class_name(
    matched_class: np.ndarray,
    valid: np.ndarray,
    class_names: List[str],
) -> str:
    """取该图有效候选中出现最多的类别名（供 split_by_class 按类留出）。"""
    cls = [int(c) for c, v in zip(matched_class.tolist(), valid.tolist()) if v > 0 and c >= 0]
    if not cls:
        return "__background__"
    top = Counter(cls).most_common(1)[0][0]
    if 0 <= top < len(class_names):
        return class_names[top]
    return "__unknown__"


def preprocess_dataset(cfg: PreprocessConfig, sam2_cfg: Optional[Sam2Config] = None) -> int:
    """跑完整预处理，返回成功缓存的图像数。"""
    # 延迟导入重依赖，便于 --dry_run 时不强制加载
    from ..data.coco_lvis import COCOInstanceData
    from ..encoders.dinov2_encoder import DINOv2RegionEncoder
    from ..encoders.text_encoder import TextPrototypeBuilder
    from ..proposals.sam2_proposal import SAM2Proposer

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = COCOInstanceData(cfg.ann_file, cfg.image_dir, max_images=cfg.max_images)
    print(f"[preprocess] dataset: {len(data)} images, {data.num_classes} classes")

    # 文本原型（vocabulary bank）——只需算一次
    proto_path = Path(cfg.prototypes_path)
    if not proto_path.exists():
        print(f"[preprocess] building text prototypes -> {proto_path}")
        tb = TextPrototypeBuilder(device=cfg.device)
        prototypes = tb.build(data.class_names)   # [C, proj_dim]
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prototypes, proto_path)
        del tb
    else:
        print(f"[preprocess] reuse existing prototypes: {proto_path}")

    sam2_cfg = sam2_cfg or Sam2Config(device=cfg.device)
    proposer = SAM2Proposer(config=sam2_cfg)
    encoder = DINOv2RegionEncoder(device=cfg.device)

    n_ok = 0
    for idx in range(len(data)):
        rec = data.get_record(idx)
        if rec is None:
            continue
        image = rec["image"]
        instances = rec["instances"]
        h, w = image.shape[:2]

        cands = proposer.generate(image, max_candidates=cfg.max_candidates_per_image)
        if not cands:
            print(f"[preprocess] image {rec['image_id']}: no candidates, skip")
            continue

        # 三路 crop + geometry
        masked_crops, box_crops, ctx_crops, geoms = [], [], [], []
        for c in cands:
            mc, bc, cc = build_three_crops(image, c.mask, c.bbox)
            masked_crops.append(mc)
            box_crops.append(bc)
            ctx_crops.append(cc)
            geoms.append(geometry_features(c.bbox, c.area, h, w))

        # DINOv2 三路拼接 z_i
        z = encoder.encode_views(masked_crops, box_crops, ctx_crops)   # [N, REGION_DIM]
        if z.shape[1] != REGION_DIM:
            print(f"[preprocess][warn] z dim {z.shape[1]} != REGION_DIM {REGION_DIM} "
                  f"(检查 DINOV2_DIM / NUM_CROP_VIEWS 配置)")

        # 候选-GT 匹配
        results = match_candidates([c.mask for c in cands], instances)
        labels = stack_match_labels(results)

        matched_class = labels["matched_class"].copy()
        valid = labels["valid"]
        # background 候选（matched_class=-1）置 0 占位，靠 valid=0 排除出正样本
        matched_class[matched_class < 0] = 0

        # is_countable：matched_class 是否 countable（COCO 全 thing）
        is_countable = np.array(
            [1.0 if (v > 0 and data.is_countable[mc]) else 0.0
             for mc, v in zip(matched_class.tolist(), valid.tolist())],
            dtype=np.float32,
        )

        class_name = _dominant_class_name(matched_class, valid, data.class_names)

        out = {
            "z": z.float(),
            "matched_class": torch.from_numpy(matched_class).long(),
            "purity": torch.from_numpy(labels["purity"]).float(),
            "coverage": torch.from_numpy(labels["coverage"]).float(),
            "valid": torch.from_numpy(valid).long(),
            "is_countable": torch.from_numpy(is_countable).float(),
            "matched_instance_id": torch.from_numpy(labels["matched_instance_id"]).long(),
            "geometry": torch.from_numpy(np.stack(geoms, axis=0)).float(),
            "class_name": class_name,
            "image_id": int(rec["image_id"]),
            "bbox": torch.tensor([list(c.bbox) for c in cands], dtype=torch.float32),
        }
        out_path = cache_dir / f"{rec['image_id']:012d}.pt"
        torch.save(out, out_path)
        n_ok += 1
        if n_ok % 5 == 0 or n_ok == 1:
            print(f"[preprocess] {n_ok} cached (last: {out_path.name}, "
                  f"{len(cands)} cands, valid={int(valid.sum())})")

    print(f"[preprocess] done: {n_ok} images cached under {cache_dir}")
    return n_ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default=PreprocessConfig.dataset_root)
    p.add_argument("--split", default=PreprocessConfig.split)
    p.add_argument("--cache_dir", default=PreprocessConfig.cache_dir)
    p.add_argument("--prototypes_path", default=PreprocessConfig.prototypes_path)
    p.add_argument("--max_images", type=int, default=PreprocessConfig.max_images)
    p.add_argument("--max_candidates_per_image", type=int,
                   default=PreprocessConfig.max_candidates_per_image)
    p.add_argument("--device", default=PreprocessConfig.device)
    p.add_argument("--dry_run", action="store_true",
                   help="只校验模块导入与配置，不加载模型/数据")
    args = p.parse_args()

    cfg = PreprocessConfig(
        dataset_root=args.dataset_root,
        split=args.split,
        cache_dir=args.cache_dir,
        prototypes_path=args.prototypes_path,
        max_images=args.max_images,
        max_candidates_per_image=args.max_candidates_per_image,
        device=args.device,
    )

    if args.dry_run:
        print("[preprocess][dry_run] config OK:")
        print(f"  ann_file   = {cfg.ann_file}")
        print(f"  image_dir  = {cfg.image_dir}")
        print(f"  cache_dir  = {cfg.cache_dir}")
        print(f"  REGION_DIM = {REGION_DIM}")
        print("[preprocess][dry_run] 链路模块均可导入，未执行实际计算。")
        return

    preprocess_dataset(cfg)


if __name__ == "__main__":
    main()
