"""Stage 1 训练脚本：训练 Category Prediction Head。

支持四种分类头逐一尝试（对应 2_training_plan.md §2.1）：

    python -m frame_1.training.train_category --head_type cosine
    python -m frame_1.training.train_category --head_type linear
    python -m frame_1.training.train_category --head_type margin
    python -m frame_1.training.train_category --head_type hybrid
    python -m frame_1.training.train_category --head_type all      # 依次跑全部

数据接口：CachedCandidateDataset 读取离线预计算缓存（DINOv2 特征 + 候选-GT 匹配标签）。
不带 --data_dir 时使用合成数据做 smoke test，验证四种 head 都能前向/反向跑通。

冻结：SAM2 / DINOv2 / Text Encoder（这里以离线特征 + 固定 text_prototypes 体现）。
只训练：projection head + 各评分头 + 辅助头。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import PROJ_DIM, REGION_DIM
from ..heads.category_head import (
    AuxiliaryHeads,
    CosineMarginHead,
    HybridCategoryHead,
    build_category_head,
    CategoryHeadConfig,
)
from .losses import CategoryLossConfig, category_loss

HEAD_TYPES = ["cosine", "linear", "margin", "hybrid"]

# 与候选缓存同目录但非候选样本的文件（不应被 glob 当成 per-image 缓存）
_NON_CANDIDATE_PT = {"text_prototypes.pt"}


def _list_cache_files(data_dir: str) -> list[Path]:
    """列出候选缓存 .pt，排除文本原型等非候选文件。"""
    return sorted(
        f for f in Path(data_dir).glob("*.pt") if f.name not in _NON_CANDIDATE_PT
    )


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class CachedCandidateDataset(Dataset):
    """从离线缓存读取候选级样本。

    期望每个 .pt 文件（一张图）含字段：
        z              : [N, in_dim]   融合后的区域特征
        matched_class  : [N]           候选-GT 匹配类别 id
        purity         : [N]
        valid          : [N]           0/1
        is_countable   : [N]           0/1（matched_class 是否 countable）
    若 RegionFuse 可训练，则缓存三路原始特征并在此融合（此处假设已融合）。
    """

    def __init__(self, data_dir: str, files: Optional[list[Path]] = None) -> None:
        if files is not None:
            self.files = list(files)
        else:
            self.files = _list_cache_files(data_dir)
        if not self.files:
            raise FileNotFoundError(f"no .pt cache found under {data_dir}")
        self._index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            n = int(torch.load(f, map_location="cpu")["z"].shape[0])
            self._index.extend((fi, i) for i in range(n))
        self._cache: dict[int, dict] = {}

    @staticmethod
    def split_by_image(
        data_dir: str, val_ratio: float = 0.1, seed: int = 0
    ) -> tuple["CachedCandidateDataset", "CachedCandidateDataset"]:
        """按「图」划分 train/val，避免同图候选同时落入两侧造成泄漏。"""
        files = _list_cache_files(data_dir)
        if not files:
            raise FileNotFoundError(f"no .pt cache found under {data_dir}")
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(files), generator=g).tolist()
        n_val = max(1, int(len(files) * val_ratio))
        val_files = [files[i] for i in perm[:n_val]]
        train_files = [files[i] for i in perm[n_val:]]
        return (
            CachedCandidateDataset(data_dir, files=train_files),
            CachedCandidateDataset(data_dir, files=val_files),
        )

    @staticmethod
    def split_by_class(
        data_dir: str, val_ratio: float = 0.2, seed: int = 0
    ) -> tuple["CachedCandidateDataset", "CachedCandidateDataset", list[str]]:
        """按「类别」划分：留出一部分类别（held-out 未见类）只进验证集。

        这是衡量/优化开放词表「跨类泛化」的正确划分——验证类完全不参与训练，
        模拟测试集的未见类设定，避免用已见类 top1 选出对未见类无效的模型。
        返回 (train_ds, heldout_val_ds, heldout_class_names)。
        """
        files = _list_cache_files(data_dir)
        if not files:
            raise FileNotFoundError(f"no .pt cache found under {data_dir}")
        file_cls: dict[Path, str] = {}
        classes: set[str] = set()
        for f in files:
            d = torch.load(f, map_location="cpu")
            cn = d.get("class_name", "__unknown__")
            file_cls[f] = cn
            classes.add(cn)
        cls_list = sorted(classes)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(cls_list), generator=g).tolist()
        n_val = max(1, int(len(cls_list) * val_ratio))
        heldout = {cls_list[i] for i in perm[:n_val]}
        train_files = [f for f in files if file_cls[f] not in heldout]
        val_files = [f for f in files if file_cls[f] in heldout]
        return (
            CachedCandidateDataset(data_dir, files=train_files),
            CachedCandidateDataset(data_dir, files=val_files),
            sorted(heldout),
        )

    def _load(self, fi: int) -> dict:
        if fi not in self._cache:
            self._cache[fi] = torch.load(self.files[fi], map_location="cpu")
        return self._cache[fi]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        fi, i = self._index[idx]
        d = self._load(fi)
        return {
            "z": d["z"][i].float(),
            "matched_class": int(d["matched_class"][i]),
            "purity": float(d["purity"][i]),
            "valid": float(d["valid"][i]),
            "is_countable": float(d.get("is_countable", torch.zeros(1))[i])
            if "is_countable" in d else 0.0,
        }


class SyntheticCandidateDataset(Dataset):
    """合成数据，用于无缓存时的 smoke test：可分的高斯簇 + 随机 purity/valid。"""

    def __init__(self, num_samples: int, in_dim: int, num_classes: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.num_classes = num_classes
        centers = torch.randn(num_classes, in_dim, generator=g) * 3.0
        labels = torch.randint(0, num_classes, (num_samples,), generator=g)
        self.z = centers[labels] + torch.randn(num_samples, in_dim, generator=g)
        self.labels = labels
        self.purity = torch.rand(num_samples, generator=g).clamp(0.2, 1.0)
        self.valid = (self.purity > 0.5).float()
        self.is_countable = (labels < num_classes // 2).float()

    def __len__(self) -> int:
        return self.z.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "z": self.z[idx],
            "matched_class": int(self.labels[idx]),
            "purity": float(self.purity[idx]),
            "valid": float(self.valid[idx]),
            "is_countable": float(self.is_countable[idx]),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "z": torch.stack([b["z"] for b in batch]),
        "matched_class": torch.tensor([b["matched_class"] for b in batch], dtype=torch.long),
        "purity": torch.tensor([b["purity"] for b in batch]),
        "valid": torch.tensor([b["valid"] for b in batch]),
        "is_countable": torch.tensor([b["is_countable"] for b in batch]),
    }


# --------------------------------------------------------------------------- #
# 文本原型
# --------------------------------------------------------------------------- #
def load_text_prototypes(path: Optional[str], num_classes: int, proj_dim: int) -> torch.Tensor:
    """加载冻结的文本原型 [num_classes, proj_dim]。无文件时随机生成（smoke test）。"""
    if path:
        t = torch.load(path, map_location="cpu")
        return torch.nn.functional.normalize(t.float(), dim=-1)
    g = torch.Generator().manual_seed(123)
    t = torch.randn(num_classes, proj_dim, generator=g)
    return torch.nn.functional.normalize(t, dim=-1)


# --------------------------------------------------------------------------- #
# 训练单个 head
# --------------------------------------------------------------------------- #
def get_projected_h(head: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """取投影后的 h_i 供辅助头与对齐损失共享。"""
    if isinstance(head, HybridCategoryHead):
        return head.open_head.proj(z)
    return head.proj(z)


def forward_logits(
    head: torch.nn.Module,
    z: torch.Tensor,
    text_prototypes: torch.Tensor,
    labels: Optional[torch.Tensor],
):
    """统一各 head 的前向，返回 (logits, h)。labels=None 时为推理路径。"""
    if isinstance(head, (CosineMarginHead, HybridCategoryHead)):
        logits = head(z, text_prototypes, labels)
    else:
        logits = head(z, text_prototypes)
    return logits, get_projected_h(head, z)


def compute_class_weight(
    loader: DataLoader, num_classes: int, device: str
) -> torch.Tensor:
    """按训练集有效样本(valid>0)的类别频率计算长尾平衡权重（inverse-freq，归一化到均值 1）。"""
    counts = torch.zeros(num_classes)
    for batch in loader:
        labels = batch["matched_class"]
        valid = batch["valid"] > 0
        for c in labels[valid].tolist():
            if 0 <= c < num_classes:
                counts[c] += 1
    freq = counts.clamp_min(1.0)
    w = freq.sum() / (num_classes * freq)   # inverse frequency
    w = w / w.mean()
    return w.to(device)


@torch.no_grad()
def evaluate(
    head: torch.nn.Module,
    aux: AuxiliaryHeads,
    loader: DataLoader,
    text_prototypes: torch.Tensor,
    device: str,
) -> dict:
    """在验证集上计算 top1 / countable_acc（仅统计 valid>0 的候选）。"""
    head.eval()
    aux.eval()
    n_valid = 0
    correct = 0
    cnt_correct = 0
    cnt_total = 0
    for batch in loader:
        z = batch["z"].to(device)
        labels = batch["matched_class"].to(device)
        valid = (batch["valid"] > 0).to(device)
        is_countable = batch["is_countable"].to(device)

        logits, h = forward_logits(head, z, text_prototypes, None)
        pred = logits.argmax(dim=-1)
        m = valid
        n_valid += int(m.sum())
        correct += int(((pred == labels) & m).sum())

        cnt_logit = aux(h)["is_countable_logit"]
        cnt_pred = (cnt_logit > 0).float()
        cnt_correct += int(((cnt_pred == is_countable) & m).sum())
        cnt_total += int(m.sum())

    return {
        "eval_top1": correct / max(n_valid, 1),
        "eval_countable_acc": cnt_correct / max(cnt_total, 1),
        "eval_n_valid": float(n_valid),
    }


def train_one_head(
    head_type: str,
    loader: DataLoader,
    text_prototypes: torch.Tensor,
    in_dim: int,
    num_classes: int,
    proj_dim: int,
    epochs: int,
    lr: float,
    device: str,
    *,
    val_loader: Optional[DataLoader] = None,
    loss_cfg: Optional[CategoryLossConfig] = None,
    class_weight: Optional[torch.Tensor] = None,
    weight_decay: float = 1e-4,
    dropout: float = 0.0,
    num_layers: int = 2,
    warmup_epochs: int = 0,
    patience: int = 0,
    save_path: Optional[str] = None,
) -> dict:
    proj_kwargs = {"dropout": dropout, "num_layers": num_layers}
    extra: dict = {}
    # 把 proj_kwargs 透传给各 head 内部的 ProjectionHead
    if head_type in ("cosine", "margin"):
        extra = {"proj_kwargs": proj_kwargs}
    elif head_type == "linear":
        extra = {"proj_kwargs": proj_kwargs}
    elif head_type == "hybrid":
        extra = {"open_kwargs": {"proj_kwargs": proj_kwargs},
                 "closed_kwargs": {"proj_kwargs": proj_kwargs}}

    cfg = CategoryHeadConfig(
        head_type=head_type, in_dim=in_dim, proj_dim=proj_dim,
        num_classes=num_classes, extra=extra,
    )
    head = build_category_head(cfg).to(device)
    aux = AuxiliaryHeads(proj_dim=proj_dim).to(device)
    loss_cfg = loss_cfg or CategoryLossConfig(lambda_count=0.5, lambda_align=0.1)

    params = list(head.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # warmup + cosine 退火调度（按 epoch）
    def lr_lambda(ep: int) -> float:
        if warmup_epochs > 0 and ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    text_prototypes = text_prototypes.to(device)
    cw = class_weight.to(device) if class_weight is not None else None

    last: dict = {}
    best_metric = -1.0
    best_state: Optional[dict] = None
    no_improve = 0

    for epoch in range(epochs):
        head.train()
        aux.train()
        agg: dict = {}
        n_batches = 0
        for batch in loader:
            z = batch["z"].to(device)
            labels = batch["matched_class"].to(device)
            w = (batch["purity"] * batch["valid"]).to(device)
            is_countable = batch["is_countable"].to(device)

            logits, h = forward_logits(head, z, text_prototypes, labels)
            aux_out = aux(h)

            total, parts = category_loss(
                logits, labels, w, loss_cfg,
                is_countable_logit=aux_out["is_countable_logit"],
                is_countable_target=is_countable,
                h=torch.nn.functional.normalize(h, dim=-1),
                text_prototypes=text_prototypes,
                class_weight=cw,
            )

            opt.zero_grad()
            total.backward()
            opt.step()

            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            n_batches += 1

        sched.step()
        last = {k: v / max(n_batches, 1) for k, v in agg.items()}

        msg = f"[{head_type}] epoch {epoch + 1}/{epochs}  lr={opt.param_groups[0]['lr']:.2e}  " + \
              "  ".join(f"{k}={v:.4f}" for k, v in last.items())

        if val_loader is not None:
            val = evaluate(head, aux, val_loader, text_prototypes, device)
            last.update(val)
            msg += "  " + "  ".join(f"{k}={v:.4f}" for k, v in val.items())

            cur = val["eval_top1"]
            if cur > best_metric:
                best_metric = cur
                no_improve = 0
                best_state = {
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "aux": {k: v.detach().cpu().clone() for k, v in aux.state_dict().items()},
                    "head_type": head_type,
                    "in_dim": in_dim,
                    "proj_dim": proj_dim,
                    "num_classes": num_classes,
                    "metrics": {**last},
                    "epoch": epoch + 1,
                }
                msg += "  *best*"
            else:
                no_improve += 1

        print(msg)

        if patience > 0 and val_loader is not None and no_improve >= patience:
            print(f"[{head_type}] early stop at epoch {epoch + 1} "
                  f"(no improve {no_improve} epochs, best eval_top1={best_metric:.4f})")
            break

    # 保存最优（有验证集时）或最后一个（无验证集时）
    if save_path is not None:
        if best_state is None:
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "aux": {k: v.detach().cpu().clone() for k, v in aux.state_dict().items()},
                "head_type": head_type,
                "in_dim": in_dim,
                "proj_dim": proj_dim,
                "num_classes": num_classes,
                "metrics": {**last},
                "epoch": epochs,
            }
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
        print(f"[{head_type}] saved checkpoint -> {save_path} "
              f"(best eval_top1={best_metric:.4f})" if best_metric >= 0
              else f"[{head_type}] saved checkpoint -> {save_path}")

    if best_metric >= 0:
        last["best_eval_top1"] = best_metric
    return last


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head_type", default="all",
                   choices=HEAD_TYPES + ["all"])
    p.add_argument("--data_dir", default=None, help="离线缓存目录；缺省用合成数据")
    p.add_argument("--text_prototypes", default=None, help="文本原型 .pt；缺省随机")
    p.add_argument("--in_dim", type=int, default=REGION_DIM,
                   help="融合区域特征维度，默认 = REGION_DIM(1152)")
    p.add_argument("--proj_dim", type=int, default=PROJ_DIM,
                   help="对齐空间维度，默认 = PROJ_DIM(512)")
    p.add_argument("--num_classes", type=int, default=80,
                   help="闭集类别数，COCO=80（合成 smoke test 也用此值）")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_synth", type=int, default=4096)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ---- 验证 / 早停 / 保存 ----
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="划分验证集比例；0 表示不划分（无验证评测）")
    p.add_argument("--val_dir", default=None,
                   help="独立验证缓存目录；指定后忽略 --val_ratio")
    p.add_argument("--val_by_class", action="store_true",
                   help="按类别留出 held-out 未见类做验证（衡量开放词表跨类泛化）")
    p.add_argument("--patience", type=int, default=0,
                   help=">0 时启用早停（连续 N 个 epoch 验证 top1 未提升）")
    p.add_argument("--save_dir", default=None,
                   help="保存最优 checkpoint 的目录；缺省不保存")
    p.add_argument("--seed", type=int, default=0)

    # ---- 优化 / 正则 ----
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="ProjectionHead dropout，缓解过拟合")
    p.add_argument("--num_layers", type=int, default=2, help="ProjectionHead MLP 层数")
    p.add_argument("--warmup_epochs", type=int, default=0)

    # ---- 损失 ----
    p.add_argument("--use_focal", action="store_true", help="用 focal loss 替代加权 CE")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--lambda_count", type=float, default=0.5)
    p.add_argument("--lambda_align", type=float, default=0.1)
    p.add_argument("--align_temperature", type=float, default=0.07)
    p.add_argument("--class_weight", action="store_true",
                   help="按训练集类别频率计算长尾平衡权重")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    val_dataset: Optional[Dataset] = None
    if args.data_dir:
        if args.val_dir:
            dataset: Dataset = CachedCandidateDataset(args.data_dir)
            val_dataset = CachedCandidateDataset(args.val_dir)
        elif args.val_by_class and args.val_ratio > 0:
            dataset, val_dataset, heldout = CachedCandidateDataset.split_by_class(
                args.data_dir, val_ratio=args.val_ratio, seed=args.seed
            )
            print(f"[info] 按类划分：held-out 未见类 {len(heldout)} 个 -> {heldout}")
        elif args.val_ratio > 0:
            dataset, val_dataset = CachedCandidateDataset.split_by_image(
                args.data_dir, val_ratio=args.val_ratio, seed=args.seed
            )
        else:
            dataset = CachedCandidateDataset(args.data_dir)
    else:
        print("[info] no --data_dir, using synthetic data for smoke test")
        dataset = SyntheticCandidateDataset(args.num_synth, args.in_dim, args.num_classes)
        if args.val_ratio > 0:
            val_dataset = SyntheticCandidateDataset(
                max(256, args.num_synth // 5), args.in_dim, args.num_classes, seed=999
            )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        if val_dataset is not None else None
    )
    text_prototypes = load_text_prototypes(args.text_prototypes, args.num_classes, args.proj_dim)

    class_weight = None
    if args.class_weight:
        class_weight = compute_class_weight(loader, args.num_classes, args.device)
        print(f"[info] class_weight computed (min={class_weight.min():.3f}, "
              f"max={class_weight.max():.3f})")

    loss_cfg = CategoryLossConfig(
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        lambda_count=args.lambda_count,
        lambda_align=args.lambda_align,
        align_temperature=args.align_temperature,
        label_smoothing=args.label_smoothing,
    )

    head_types = HEAD_TYPES if args.head_type == "all" else [args.head_type]
    summary = {}
    for ht in head_types:
        print(f"\n==== training head_type = {ht} ====")
        save_path = (
            str(Path(args.save_dir) / f"category_{ht}.pt") if args.save_dir else None
        )
        summary[ht] = train_one_head(
            ht, loader, text_prototypes,
            args.in_dim, args.num_classes, args.proj_dim,
            args.epochs, args.lr, args.device,
            val_loader=val_loader,
            loss_cfg=loss_cfg,
            class_weight=class_weight,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            num_layers=args.num_layers,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            save_path=save_path,
        )

    print("\n==== summary ====")
    for ht, m in summary.items():
        line = f"{ht:8s}  L_total={m.get('L_total', float('nan')):.4f}"
        if "best_eval_top1" in m:
            line += f"  best_eval_top1={m['best_eval_top1']:.4f}"
        print(line)


if __name__ == "__main__":
    main()
