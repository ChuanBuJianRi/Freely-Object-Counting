"""OV-CUD 全局配置（对齐 1_revised(1).md 与 2_training_plan.md）。

集中管理模型名、特征维度、候选-GT 匹配阈值、crop 参数和数据路径，
供离线预处理（preprocess）与训练（train_category）共用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------- #
# 冻结的骨干模型（均经 hf-mirror 下载并缓存）
# --------------------------------------------------------------------------- #
SAM2_MODEL = "/home/czp/official_code/models/sam2.1-hiera-small"   # 候选生成（本地离线 small 权重）
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
SAM2_POINTS_PER_BATCH = 1000   # OCCAM-M 推荐值
MIN_AREA_RATIO = 1e-4   # mask 面积 / 图像面积 下限
MAX_AREA_RATIO = 0.95   # 上限（过滤近全图 mask）
NEAR_DUP_IOU = 0.9      # 高 IoU 近重复候选去重阈值

# SAM2 自动 mask 生成器（AMG）参数，对齐 transformers mask-generation pipeline。
# 取值参考 SAM/SAM2 官方默认；过完备采样下适当放宽质量阈值以保证 recall。
SAM2_POINTS_PER_SIDE = 16        # 每边网格采样点数（GPU 实测 16 时 0.5s/图，候选召回足够训练分类头；32 因 batch 显存路径反而慢且接近 24G 上限）
SAM2_PRED_IOU_THRESH = 0.88      # 预测 mask 质量过滤阈值
SAM2_STABILITY_SCORE_THRESH = 0.95   # 稳定性分数过滤阈值
SAM2_STABILITY_SCORE_OFFSET = 1.0    # 计算稳定性分数时的偏移量
SAM2_MASK_THRESHOLD = 0.0        # 将预测 mask 二值化的阈值
SAM2_CROP_N_LAYERS = 0           # >0 时在图像 crop 上多层重跑（提高小目标 recall）
SAM2_CROP_OVERLAP_RATIO = 512 / 1500   # crop 之间的重叠比例
SAM2_CROP_N_POINTS_DOWNSCALE = 1       # 每层 crop 采样点的降采样因子
SAM2_CROPS_NMS_THRESH = 0.7      # 跨 crop 去重的 box IoU NMS 阈值
DEDUP_BOX_IOU_PRESCREEN = 0.5    # 近重复去重时 bbox 预筛 IoU 阈值（低于则跳过 mask IoU 计算）


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
class Sam2Config:
    """SAM2 自动候选生成配置（封装 AMG 参数 + 候选过滤/去重阈值）。

    默认值取自上方模块级常量，便于在不同实验中整体覆盖。
    """

    model_name: str = SAM2_MODEL
    device: str = "cpu"

    # 自动 mask 生成器（AMG）采样 / 质量参数
    points_per_batch: int = SAM2_POINTS_PER_BATCH
    points_per_side: int = SAM2_POINTS_PER_SIDE
    pred_iou_thresh: float = SAM2_PRED_IOU_THRESH
    stability_score_thresh: float = SAM2_STABILITY_SCORE_THRESH
    stability_score_offset: float = SAM2_STABILITY_SCORE_OFFSET
    mask_threshold: float = SAM2_MASK_THRESHOLD
    crops_n_layers: int = SAM2_CROP_N_LAYERS
    crop_overlap_ratio: float = SAM2_CROP_OVERLAP_RATIO
    crop_n_points_downscale_factor: int = SAM2_CROP_N_POINTS_DOWNSCALE
    crops_nms_thresh: float = SAM2_CROPS_NMS_THRESH

    # 候选过滤 / 去重
    min_area_ratio: float = MIN_AREA_RATIO
    max_area_ratio: float = MAX_AREA_RATIO
    min_box_size: int = MIN_BOX_SIZE
    near_dup_iou: float = NEAR_DUP_IOU
    dedup_box_iou_prescreen: float = DEDUP_BOX_IOU_PRESCREEN

    def amg_kwargs(self) -> dict:
        """返回可直接传给 transformers mask-generation pipeline 的 AMG 参数字典。"""
        return {
            "points_per_batch": self.points_per_batch,
            "points_per_crop": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "stability_score_offset": self.stability_score_offset,
            "mask_threshold": self.mask_threshold,
            "crops_n_layers": self.crops_n_layers,
            "crop_overlap_ratio": self.crop_overlap_ratio,
            "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor,
            "crops_nms_thresh": self.crops_nms_thresh,
        }


@dataclass
class PreprocessConfig:
    """离线预计算配置。"""

    dataset_root: str = "/home/czp/official_code/dataset"
    split: str = "val2017"            # 子集来源（val 更小，适合 CPU 验证）
    cache_dir: str = "/home/czp/official_code/cache"
    prototypes_path: str = "/home/czp/official_code/cache/text_prototypes.pt"
    max_images: int = 30              # CPU 子集规模，-1 表示全量
    max_candidates_per_image: int = 0   # 0 = 无上限（据 dot recall 诊断结论放开）
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
    "SAM2_POINTS_PER_SIDE",
    "SAM2_PRED_IOU_THRESH",
    "SAM2_STABILITY_SCORE_THRESH",
    "SAM2_STABILITY_SCORE_OFFSET",
    "SAM2_MASK_THRESHOLD",
    "SAM2_CROP_N_LAYERS",
    "SAM2_CROP_OVERLAP_RATIO",
    "SAM2_CROP_N_POINTS_DOWNSCALE",
    "SAM2_CROPS_NMS_THRESH",
    "DEDUP_BOX_IOU_PRESCREEN",
    "PROMPT_TEMPLATES",
    "Sam2Config",
    "PreprocessConfig",
]
