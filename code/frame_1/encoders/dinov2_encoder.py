"""DINOv2 区域编码器（OV-CUD §7 Region Encoding）。

冻结 DINOv2，对每个候选的三路 crop（masked / box / context）分别编码，
取 CLS（pooler_output）作为区域向量，拼接成 z_i：

    z_i = [ DINOv2(masked_crop), DINOv2(box_crop), DINOv2(context_crop) ]   # 3 * 384 = 1152

DINOv2 只负责视觉表征，不输出类别。融合（这里为简单拼接）后供 Category Head 使用。
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image

from ..config import DINOV2_MODEL


class DINOv2RegionEncoder:
    """批量编码 PIL crop 列表，返回 [N, 384] 区域特征。"""

    def __init__(self, model_name: str = DINOV2_MODEL, device: str = "cpu") -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, crops: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """编码一组 crop，返回 [N, dim]（CLS / pooler_output）。"""
        if len(crops) == 0:
            return torch.zeros(0, self.dim)
        feats = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            if getattr(out, "pooler_output", None) is not None:
                f = out.pooler_output
            else:
                f = out.last_hidden_state[:, 0]   # CLS token 兜底
            feats.append(f.float().cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_views(
        self,
        masked_crops: List[Image.Image],
        box_crops: List[Image.Image],
        context_crops: List[Image.Image],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """三路 crop 分别编码后拼接，返回 [N, 3*dim]。三路长度必须一致。"""
        assert len(masked_crops) == len(box_crops) == len(context_crops)
        z_mask = self.encode(masked_crops, batch_size)
        z_box = self.encode(box_crops, batch_size)
        z_ctx = self.encode(context_crops, batch_size)
        return torch.cat([z_mask, z_box, z_ctx], dim=-1)   # [N, 3*dim]


__all__ = ["DINOv2RegionEncoder"]
