"""CLIP 文本原型生成（OV-CUD §5 Vocabulary Bank）。

用冻结的 CLIP text encoder，把每个类别名经多模板（prompt ensemble）编码并平均，
得到 L2 归一化的文本原型 t_c，作为开放词表分类头的固定锚点。

    t_c = normalize( mean_template normalize(CLIP_text("a photo of a {class}")) )

新增类别只需追加类名重新生成原型，无需训练（对应文档"可扩展词表"）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from ..config import CLIP_MODEL, PROMPT_TEMPLATES


class TextPrototypeBuilder:
    """加载冻结 CLIP text encoder，按词表生成文本原型矩阵。"""

    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        templates: Optional[Sequence[str]] = None,
        device: str = "cpu",
    ) -> None:
        from transformers import CLIPModel, CLIPTokenizer

        self.device = device
        self.templates = list(templates) if templates is not None else list(PROMPT_TEMPLATES)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def proj_dim(self) -> int:
        return int(self.model.config.projection_dim)

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        out = self.model.get_text_features(**tok)
        # transformers 5.x: get_text_features 返回 BaseModelOutputWithPooling
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        return torch.nn.functional.normalize(feat.float(), dim=-1)

    @torch.no_grad()
    def build(self, class_names: Sequence[str]) -> torch.Tensor:
        """返回 [num_classes, proj_dim] 的归一化文本原型。"""
        prototypes = []
        for name in class_names:
            prompts = [t.format(name) for t in self.templates]
            emb = self._encode_texts(prompts)          # [num_templates, D]
            proto = emb.mean(dim=0)                     # 模板平均
            proto = torch.nn.functional.normalize(proto, dim=-1)
            prototypes.append(proto)
        return torch.stack(prototypes, dim=0)           # [C, D]


__all__ = ["TextPrototypeBuilder"]
