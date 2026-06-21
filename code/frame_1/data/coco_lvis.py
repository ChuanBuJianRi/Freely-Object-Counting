"""COCO / LVIS 实例分割数据加载（OV-CUD §10.1 训练数据）。

提供：
    - 词表（contiguous class idx <-> category name），用于文本原型与类别标签
    - 逐图读取：图像 + GT 实例（mask / bbox / class_idx / instance_id）

只使用 instance mask + category 标签，不使用任何 count / density 监督，符合方法定位。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class COCOInstanceData:
    """封装 pycocotools，按 contiguous 类别索引提供词表与逐图 GT。"""

    def __init__(self, ann_file: str, image_dir: str, max_images: int = -1) -> None:
        from pycocotools.coco import COCO

        self.image_dir = image_dir
        self.coco = COCO(ann_file)

        cat_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(cat_ids)
        # contiguous: 0..C-1 对应 COCO category id
        self.cat_id_to_idx: Dict[int, int] = {cid: i for i, cid in enumerate(cat_ids)}
        self.idx_to_cat_id: Dict[int, int] = {i: cid for cid, i in self.cat_id_to_idx.items()}
        self.class_names: List[str] = [c["name"] for c in cats]
        self.num_classes = len(self.class_names)
        # COCO 全部为 thing 类（countable）
        self.is_countable = [True] * self.num_classes

        img_ids = sorted(self.coco.getImgIds())
        # 只保留含标注的图，保证有 GT 用于匹配
        img_ids = [i for i in img_ids if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0]
        if max_images is not None and max_images > 0:
            img_ids = img_ids[:max_images]
        self.img_ids = img_ids

    def __len__(self) -> int:
        return len(self.img_ids)

    def get_record(self, idx: int) -> Optional[Dict]:
        """返回 {image: HxWx3 uint8, instances: [...], image_id, file_name}。"""
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.image_dir, info["file_name"])
        if not os.path.exists(path):
            return None
        image = np.array(Image.open(path).convert("RGB"))
        h, w = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        instances = []
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            m = self.coco.annToMask(a)
            if m.shape != (h, w) or m.sum() == 0:
                continue
            x, y, bw, bh = a["bbox"]
            instances.append(
                {
                    "mask": m.astype(np.uint8),
                    "bbox": [x, y, x + bw, y + bh],
                    "class_idx": self.cat_id_to_idx[a["category_id"]],
                    "instance_id": a["id"],
                }
            )
        if not instances:
            return None
        return {
            "image": image,
            "instances": instances,
            "image_id": img_id,
            "file_name": info["file_name"],
        }


__all__ = ["COCOInstanceData"]
