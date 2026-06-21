# OpenCount / OV-CUD 管线说明

本文件说明 `official_code` 目录中整理收纳的管线代码与资源。
最新管线来源为 `ws_yiyang/OpenCount/frame_1`（开放词表无监督计数，OV-CUD），
已按目录用途归类复制进本工程。

## 目录结构

```
official_code/
├── agent说明.md          # 本说明
├── code/                 # 管线源码
│   └── frame_1/          # OV-CUD 核心代码包（从 ws_yiyang/OpenCount/frame_1 复制）
├── library/              # 设计与训练规划文档
│   ├── 1_revised(1).md       # 方法设计文档
│   └── 2_training_plan.md    # 训练规划文档
├── dataset/              # 数据集（已物理拷贝，约 46G）
│   ├── OpenCount_dataset/    # coco (39G) + lvis (1.6G)
│   └── FSC147/               # FSC147 计数评测集 (4.9G)
├── script/               # 运行/评测脚本（预留）
├── result/               # 实验结果产物
│   └── checkpoints/          # 训练好的权重（分类头/关系头/文本原型）
└── cache/                # 离线预计算缓存（预留）
```

> 注意：数据集已全部物理拷贝，本工程数据自包含：
> - `dataset/OpenCount_dataset/`：COCO 39G + LVIS 1.6G（训练用）
> - `dataset/FSC147/`：FSC147 计数评测集 4.9G（含 images_384_VarV2、gt_density_map、标注 json）

## 训练好的权重（result/checkpoints）

在 FSC147 上训练好的最新权重，来源 `ws_yiyang/ovcud_cache`。

| 文件 | 内容 | 关键元信息 |
| --- | --- | --- |
| `fsc147_hybrid.pt` | 分类头（hybrid，Stage 1） | `head_type=hybrid`，`in_dim=384`，`proj_dim=512`，`num_classes=147`；含 `head`/`aux` 权重与 `metrics`（eval_top1≈0.826，countable_acc=1.0） |
| `fsc147_relation.pt` | 关系头（Stage 2，epoch 20） | `feat_dim=1544`，`z_dim=384`，`hidden_dim=512`，`num_layers=3`；含 `relation_head` 权重与 `optimizer`，依赖上面的分类头（`category_ckpt`） |
| `text_prototypes_fsc147.pt` | FSC147 文本原型 | `[147, 512]`，已 L2 归一化，作为开放词表分类锚点 |
| `text_prototypes_fsc147_categories.json` | 原型类别名 | 与文本原型行一一对应的 147 个类别名 |

> 说明：关系头依赖分类头（其 `category_ckpt` 原指向 `ws_yiyang/ovcud_cache/ckpts/fsc147_hybrid.pt`），
> 加载时如需自包含，请把该路径指向本工程 `result/checkpoints/fsc147_hybrid.pt`。
> 关系头各 epoch 中间检查点（ep005/010/015/020）与 LVIS 关系头未拉取，如需可再补。

## 管线代码（code/frame_1）

OV-CUD 是一条「冻结骨干 + 轻量可训练头」的开放词表计数管线，整体流程：

```
图像
 → SAM2 自动候选生成 (proposals)
 → 候选裁剪三视图 + 几何特征 (candidates)
 → DINOv2 区域编码 / CLIP 文本原型 (encoders)
 → Category Prediction Head 分类 (heads)
 → 训练（候选-GT 匹配 + 加权损失）(training)
```

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| `config.py` | 全局配置 | 冻结骨干模型名、特征维度、候选-GT 匹配阈值、crop/几何参数、prompt 模板、`PreprocessConfig` 数据路径 |
| `proposals/sam2_proposal.py` | 候选生成（§6.1/§6.3） | 用冻结 SAM2 mask-generation pipeline 生成过完备候选，按面积比/最小框/近重复 IoU 过滤 |
| `candidates/crops.py` | 候选裁剪（§6.2） | 为每个候选构造 masked / box / context 三路 crop |
| `candidates/geometry.py` | 几何与集合运算 | 8 维几何特征、`mask_overlaps`(IoU/purity/coverage)、`mask_iou`、`containment` |
| `encoders/dinov2_encoder.py` | 区域编码（§7） | 冻结 DINOv2 编码三路 crop 并拼接成区域特征 `z_i`（3×384=1152） |
| `encoders/text_encoder.py` | 文本原型（§5） | 冻结 CLIP，prompt ensemble 生成 L2 归一化文本原型，支持词表扩展 |
| `heads/category_head.py` | 分类头（Stage 1 §2.1） | 四种可切换 head：A 文本原型 cosine / B 线性原型 / C cosine+margin / D 混合（推荐），加共享 `ProjectionHead` 与辅助头 `AuxiliaryHeads` |
| `training/losses.py` | 损失（§2.2） | `L_category = L_cls + λ_count·L_count + λ_align·L_align`，加权 CE/focal、BCE、InfoNCE 对齐 |
| `training/train_category.py` | Stage 1 训练 | 训练分类头，支持离线缓存数据集与合成数据 smoke test |
| `data/coco_lvis.py` | 数据加载（§10.1） | 封装 pycocotools，提供词表与逐图 GT 实例（仅用 mask+类别，不用 count 监督） |

### 关键设计要点

- **冻结骨干**：SAM2 / DINOv2 / CLIP 文本编码器全部冻结，仅训练 projection head、各评分头与辅助头。
- **开放词表**：文本原型作为固定锚点，新增类别只需追加类名重新生成原型，无需改结构（线性/混合头的闭集分支可 `expand_classes` 后微调）。
- **候选-GT 匹配**：用 purity / coverage / IoU 阈值（`TAU_PURITY`/`TAU_PART`/`TAU_IOU_MATCH`）确定正/部件/背景样本，低 purity 样本通过权重为 0 自动忽略。

## 运行方式

代码包目录名为 `frame_1`，从 `official_code/code/` 下以包方式运行。

Stage 1 训练（无数据时自动用合成数据做 smoke test）：

```bash
cd /home/czp/official_code/code
python -m frame_1.training.train_category --head_type all
# 单独某种头：cosine | linear | margin | hybrid
python -m frame_1.training.train_category --head_type hybrid
```

使用离线缓存与文本原型：

```bash
python -m frame_1.training.train_category \
  --head_type hybrid \
  --data_dir <离线缓存目录> \
  --text_prototypes <text_prototypes.pt>
```

> 说明：`config.py` 中的数据/缓存路径为绝对路径，默认仍指向 `ws_yiyang/OpenCount`。
> 数据集已拷贝到本工程 `dataset/OpenCount_dataset/`，如需本工程独立运行，
> 把 `PreprocessConfig.dataset_root` 改为
> `/home/czp/official_code/dataset/OpenCount_dataset`，并将 `cache_dir` 指向 `cache/`。

## 依赖

主要依赖：`torch`、`transformers`、`numpy`、`pillow`、`pycocotools`。
SAM2 / DINOv2 / CLIP 权重经 hf-mirror 下载缓存（见 `config.py` 模型名）。
