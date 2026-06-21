"""Category Prediction Head 的多种实现（OV-CUD Stage 1）。

对应训练规划文档 `2_training_plan.md` 的 §2.1，提供四种可切换的分类头评分方式：

    A. TextPrototypeCosineHead   —— 视觉 embedding 与文本原型 cosine（开放词表主干）
    B. LinearPrototypeHead       —— 每类可学习权重向量（闭集判别力强）
    C. CosineMarginHead          —— cosine + 角度 margin（ArcFace / CosFace 风格）
    D. HybridCategoryHead        —— A/C 开放词表分支 + B 闭集分支加权融合（推荐）

所有 head 共享一个 ProjectionHead，把 DINOv2 区域特征 z_i 映射到对齐空间 h_i。
此外提供辅助头 AuxiliaryHeads（is_countable 二分类 + 可选 group_type 粗类）。

约定：
    - 输入 z: [B, in_dim]   区域融合特征
    - 文本原型 text_prototypes: [num_classes, proj_dim]，已 L2 normalize，冻结
    - 主分类 logits: [B, num_classes]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 共享 Projection Head
# --------------------------------------------------------------------------- #
class ProjectionHead(nn.Module):
    """把区域特征 z_i 投影到对齐/判别空间 h_i。

    结构：MLP(in_dim -> hidden -> proj_dim)，可选 L2 normalize。
    cosine 类 head 需要 normalize=True；纯 linear head 可不 normalize。
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 512,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or proj_dim
        self.normalize = normalize

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [proj_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 中间层加激活/dropout，最后一层不加
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mlp(z)
        if self.normalize:
            h = F.normalize(h, dim=-1)
        return h


# --------------------------------------------------------------------------- #
# 方式 A：Text-Prototype Cosine Head
# --------------------------------------------------------------------------- #
class TextPrototypeCosineHead(nn.Module):
    """logit_{i,c} = cosine(h_i, t_c) / temperature。

    开放词表主干：新增类别只需向 text_prototypes 追加一行，无需改结构。
    temperature 可固定也可学习（log 空间参数化，clamp 到合理范围）。
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 512,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        proj_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.proj = ProjectionHead(in_dim, proj_dim, normalize=True, **(proj_kwargs or {}))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temperature", torch.tensor(math.log(temperature)))

    def get_temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.01, 0.5)

    def forward(self, z: torch.Tensor, text_prototypes: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)                                   # [B, proj_dim]
        t = F.normalize(text_prototypes, dim=-1)           # [C, proj_dim]
        logits = (h @ t.t()) / self.get_temperature()      # [B, C]
        return logits


# --------------------------------------------------------------------------- #
# 方式 B：Learnable Linear / Prototype Classifier
# --------------------------------------------------------------------------- #
class LinearPrototypeHead(nn.Module):
    """logit_{i,c} = W_c · h_i + b_c，每类一个可学习视觉原型。

    闭集判别力最强，但 num_classes 固定。新增类别需 expand_classes 后微调。
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        proj_dim: int = 512,
        bias: bool = True,
        normalize_proj: bool = False,
        proj_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.proj = ProjectionHead(
            in_dim, proj_dim, normalize=normalize_proj, **(proj_kwargs or {})
        )
        self.classifier = nn.Linear(proj_dim, num_classes, bias=bias)

    @property
    def num_classes(self) -> int:
        return self.classifier.out_features

    def forward(self, z: torch.Tensor, text_prototypes: Optional[torch.Tensor] = None) -> torch.Tensor:
        # text_prototypes 仅为统一接口而保留，此 head 不使用
        h = self.proj(z)
        return self.classifier(h)

    @torch.no_grad()
    def expand_classes(self, num_new: int) -> None:
        """为新增类别扩展分类器权重（新权重随机初始化，需后续微调）。"""
        old = self.classifier
        new = nn.Linear(old.in_features, old.out_features + num_new, bias=old.bias is not None)
        new.weight[: old.out_features] = old.weight
        if old.bias is not None:
            new.bias[: old.out_features] = old.bias
        self.classifier = new.to(old.weight.device)


# --------------------------------------------------------------------------- #
# 方式 C：Cosine + Margin（ArcFace / CosFace）
# --------------------------------------------------------------------------- #
class CosineMarginHead(nn.Module):
    """cosine + 角度/余弦 margin。

    训练时对正类施加 margin，推理时退化为普通 cosine（仍兼容开放词表）。
        - margin_type="arc"  : ArcFace，cos(theta + m)
        - margin_type="cos"  : CosFace，cos(theta) - m
    scale s 即 1/temperature。
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 512,
        scale: float = 30.0,
        margin: float = 0.2,
        margin_type: str = "arc",
        proj_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert margin_type in ("arc", "cos")
        self.proj = ProjectionHead(in_dim, proj_dim, normalize=True, **(proj_kwargs or {}))
        self.scale = scale
        self.margin = margin
        self.margin_type = margin_type

    def forward(
        self,
        z: torch.Tensor,
        text_prototypes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.proj(z)
        t = F.normalize(text_prototypes, dim=-1)
        cosine = (h @ t.t()).clamp(-1 + 1e-6, 1 - 1e-6)    # [B, C]

        # 推理（无 labels）：普通 cosine 打分
        if labels is None or not self.training:
            return self.scale * cosine

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        if self.margin_type == "cos":
            margin_logits = cosine - self.margin * one_hot
        else:  # arc
            theta = torch.acos(cosine)
            margin_logits = torch.cos(theta + self.margin * one_hot)
        return self.scale * margin_logits


# --------------------------------------------------------------------------- #
# 方式 D：Hybrid 双分支（推荐）
# --------------------------------------------------------------------------- #
class HybridCategoryHead(nn.Module):
    """开放词表分支(A 或 C) + 闭集判别分支(B) 加权融合。

        score = alpha * open_vocab_logits + (1 - alpha) * closed_set_logits

    闭集分支只覆盖训练集已知类别（前 num_closed_classes 列）；
    未见类别只有开放词表分支有分数，自动 fallback。
    alpha 可固定或可学习。
    """

    def __init__(
        self,
        in_dim: int,
        num_closed_classes: int,
        proj_dim: int = 512,
        alpha: float = 0.5,
        learnable_alpha: bool = True,
        open_branch: str = "cosine",   # "cosine" (A) 或 "margin" (C)
        open_kwargs: Optional[dict] = None,
        closed_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert open_branch in ("cosine", "margin")
        self.open_branch_type = open_branch
        self.num_closed_classes = num_closed_classes

        if open_branch == "cosine":
            self.open_head = TextPrototypeCosineHead(in_dim, proj_dim, **(open_kwargs or {}))
        else:
            self.open_head = CosineMarginHead(in_dim, proj_dim, **(open_kwargs or {}))

        self.closed_head = LinearPrototypeHead(
            in_dim, num_closed_classes, proj_dim, **(closed_kwargs or {})
        )

        init = math.log(alpha / (1 - alpha))  # logit 反函数，sigmoid(raw)=alpha
        if learnable_alpha:
            self.alpha_logit = nn.Parameter(torch.tensor(init))
        else:
            self.register_buffer("alpha_logit", torch.tensor(init))

    def get_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        z: torch.Tensor,
        text_prototypes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_total = text_prototypes.size(0)

        if self.open_branch_type == "margin":
            open_logits = self.open_head(z, text_prototypes, labels)
        else:
            open_logits = self.open_head(z, text_prototypes)        # [B, num_total]

        closed_logits = self.closed_head(z)                          # [B, num_closed]
        # 把闭集分支补零对齐到全词表宽度，未见类只有开放分支贡献
        if num_total > self.num_closed_classes:
            pad = z.new_zeros(z.size(0), num_total - self.num_closed_classes)
            closed_logits = torch.cat([closed_logits, pad], dim=1)

        alpha = self.get_alpha()
        return alpha * open_logits + (1 - alpha) * closed_logits


# --------------------------------------------------------------------------- #
# 辅助头：is_countable 二分类 + 可选 group_type 粗类
# --------------------------------------------------------------------------- #
class AuxiliaryHeads(nn.Module):
    """与主分类共享投影后的 h_i，输出辅助判断。

        is_countable : countable vs auxiliary/fallback 二分类 logit
        group_type   : countable / auxiliary / unknown / background 粗类 logits（可选）
    """

    def __init__(
        self,
        proj_dim: int,
        num_group_types: int = 0,
    ) -> None:
        super().__init__()
        self.is_countable = nn.Linear(proj_dim, 1)
        self.group_type = (
            nn.Linear(proj_dim, num_group_types) if num_group_types > 0 else None
        )

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {"is_countable_logit": self.is_countable(h).squeeze(-1)}
        if self.group_type is not None:
            out["group_type_logits"] = self.group_type(h)
        return out


# --------------------------------------------------------------------------- #
# 工厂：按名字构建分类头
# --------------------------------------------------------------------------- #
@dataclass
class CategoryHeadConfig:
    head_type: str = "hybrid"        # "cosine" | "linear" | "margin" | "hybrid"
    in_dim: int = 768
    proj_dim: int = 512
    num_classes: int = 80            # linear/hybrid 的闭集类别数
    extra: Optional[dict] = None     # 透传给具体 head 的额外参数


def build_category_head(cfg: CategoryHeadConfig) -> nn.Module:
    extra = cfg.extra or {}
    if cfg.head_type == "cosine":
        return TextPrototypeCosineHead(cfg.in_dim, cfg.proj_dim, **extra)
    if cfg.head_type == "linear":
        return LinearPrototypeHead(cfg.in_dim, cfg.num_classes, cfg.proj_dim, **extra)
    if cfg.head_type == "margin":
        return CosineMarginHead(cfg.in_dim, cfg.proj_dim, **extra)
    if cfg.head_type == "hybrid":
        return HybridCategoryHead(cfg.in_dim, cfg.num_classes, cfg.proj_dim, **extra)
    raise ValueError(f"unknown head_type: {cfg.head_type}")


__all__ = [
    "ProjectionHead",
    "TextPrototypeCosineHead",
    "LinearPrototypeHead",
    "CosineMarginHead",
    "HybridCategoryHead",
    "AuxiliaryHeads",
    "CategoryHeadConfig",
    "build_category_head",
]
