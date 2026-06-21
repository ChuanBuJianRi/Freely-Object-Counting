"""OV-CUD 全局配置（对齐 1_revised(1).md 与 2_training_plan.md）。

集中管理模型名、特征维度、候选-GT 匹配阈值、crop 参数和数据路径，
供离线预处理（preprocess）与训练（train_category）共用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# --------------------------------------------------------------------------- #
# 冻结的骨干模型（均经 hf-mirror 下载并缓存）
# --------------------------------------------------------------------------- #
SAM2_MODEL = "facebook/sam2.1-hiera-tiny"      # 候选生成（CPU 友好的最小尺寸）
DINOV2_MODEL = "facebook/dinov2-small"          # 区域编码，hidden = 384
CLIP_MODEL = "openai/clip-vit-base-patch32"     # 文本原型，projection_dim = 512

DINOV2_DIM = 384        # 单路 DINOv2 特征维度
NUM_CROP_VIEWS = 3      # masked / box / context 三路
REGION_DIM = DINOV2_DIM * NUM_CROP_VIEWS        # 拼接后的区域特征维度 z_i = 1152
PROJ_DIM = 512          # 对齐空间维度（= CLIP projection_dim）

# --------------------------------------------------------------------------- #
# 候选-GT 匹配阈值（2_training_plan.md §1.3）
# --------------------------------------------------------------------------- #
TAU_PURITY = 0.5        # 低于此 purity -> background/noise，不作类别强正样本
TAU_PART = 0.5          # coverage 低且 purity 高 -> part candidate
TAU_IOU_MATCH = 0.5     # IoU 匹配阈值

# --------------------------------------------------------------------------- #
# 候选裁剪 / 几何
# --------------------------------------------------------------------------- #
CONTEXT_EXPAND = 0.5    # context crop 相对 bbox 的外扩比例（每边）
MIN_BOX_SIZE = 4        # 过滤过小 bbox（像素）

# --------------------------------------------------------------------------- #
# SAM2 自动 mask 生成 / 候选过滤（6.3 Candidate Filtering）
# --------------------------------------------------------------------------- #
SAM2_POINTS_PER_BATCH = 64
MIN_AREA_RATIO = 1e-4   # mask 面积 / 图像面积 下限
MAX_AREA_RATIO = 0.95   # 上限（过滤近全图 mask）
NEAR_DUP_IOU = 0.9      # 高 IoU 近重复候选去重阈值


# --------------------------------------------------------------------------- #
# 文本原型模板（prompt ensemble，系统内部固定模板，非用户 prompt）
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATES: List[str] = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of a single {}.",
    "an image of a {}.",
    "a cropped photo of a {}.",
]


@dataclass
class PreprocessConfig:
    """离线预计算配置。"""

    dataset_root: str = "/home/czp/ws_yiyang/OpenCount/dataset"
    split: str = "val2017"            # 子集来源（val 更小，适合 CPU 验证）
    cache_dir: str = "/home/czp/ws_yiyang/OpenCount/cache"
    prototypes_path: str = "/home/czp/ws_yiyang/OpenCount/cache/text_prototypes.pt"
    max_images: int = 30              # CPU 子集规模，-1 表示全量
    max_candidates_per_image: int = 64
    device: str = "cpu"
    seed: int = 0

    @property
    def ann_file(self) -> str:
        return os.path.join(
            self.dataset_root, "coco", "annotations", f"instances_{self.split}.json"
        )

    @property
    def image_dir(self) -> str:
        return os.path.join(self.dataset_root, "coco", "images", self.split)


__all__ = [
    "SAM2_MODEL",
    "DINOV2_MODEL",
    "CLIP_MODEL",
    "DINOV2_DIM",
    "REGION_DIM",
    "PROJ_DIM",
    "NUM_CROP_VIEWS",
    "TAU_PURITY",
    "TAU_PART",
    "TAU_IOU_MATCH",
    "CONTEXT_EXPAND",
    "MIN_BOX_SIZE",
    "SAM2_POINTS_PER_BATCH",
    "MIN_AREA_RATIO",
    "MAX_AREA_RATIO",
    "NEAR_DUP_IOU",
    "PROMPT_TEMPLATES",
    "PreprocessConfig",
]
