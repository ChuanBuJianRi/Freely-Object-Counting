# PF-CUD: Parameter-Free Counting Unit Discovery

在保持 **training-free / prior-free** 的前提下，进一步实现 **parameter-free** 的通用计数框架。
这里的 parameter-free 指：推理时不需要用户针对图像、数据集、类别手动调整任何阈值或超参数。
算法内部仅使用模型默认设置、图像尺寸推导量、统计分布自动估计量（Otsu、MST、MDL、rank normalization）。

## Pipeline

```
Input Image
 -> 1. Over-complete Candidate Generation (SAM + blob + edge)
 -> 2. Candidate Canonicalization & Deduplication (MST + Otsu, no IoU threshold)
 -> 3. Feature Extraction (visual + shape + color + spatial)
 -> 4. Rank-Normalized Distance Fusion (no manual weights)
 -> 5. MST Graph Construction (no kNN, no epsilon)
 -> 6. Otsu-based Graph Cutting (no delta)
 -> 7. Connected Components as Counting Groups
 -> 8. MDL-based Group Refinement (merge, optional split)
 -> 9. Hypothesis Ranking (object / pattern / background)
 -> 10. Output Counts (count + type + confidence + visualization)
```

## 安装

```bash
pip install -r requirements.txt
```

`torch` / `torchvision` 用于视觉特征（DINOv2，回退 ResNet50）。
在受限环境中，视觉特征会自动回退为空特征（`NullVisualExtractor`），
此时仍可运行 blob + shape/color/spatial 的 Phase 1 流程。

## 单图运行

```bash
python -m pf_cud.run_image \
  --image examples/test.jpg \
  --out_json outputs/result.json \
  --out_vis outputs/result.png
```

注意：该命令没有任何需要调的算法参数。

## 数据集评估

```bash
python -m pf_cud.run_dataset \
  --image_dir examples \
  --gt_json examples/gt.json \
  --out_dir outputs
```

`gt.json` 可以是 `{"img.png": {"apple": 12, "orange": 8}}` 或 `{"img.png": [12, 8]}`。
预测 group counts 与 ground-truth 通过 Hungarian matching 对齐后计算 MAE/RMSE/NAE/SRE。

## 编程接口

```python
import numpy as np
from PIL import Image
from pf_cud.pipeline import PFCUDPipeline

image = np.array(Image.open("examples/test.jpg").convert("RGB"))
pipeline = PFCUDPipeline(sam_model=None)   # 注入 SAM/SAM2 automatic mask generator 可选
result = pipeline.run(image)

for rank, group in enumerate(result.groups):
    print(rank + 1, group.group_type, group.count, group.confidence)
```

## 目录结构

```
pf_cud/
  config.py            # 仅工程配置（设备/模型名/路径），无算法阈值
  data.py              # Candidate / CountGroup / CountResult
  candidates/          # sam / blob / edge / merge(dedup)
  features/            # utils / visual / shape / color / spatial / fusion
  graph/               # mst / cut / components
  mdl/                 # score / refine(merge + split)
  ranking/             # hypothesis
  eval/                # metrics / match
  visualize/           # draw
  pipeline.py
  run_image.py
  run_dataset.py
```

## Parameter-free 约束

- CLI 不出现任何算法阈值相关参数（epsilon / delta / k / iou_thresh / score_thresh / num_scales / finch_thresh）。
- 所有阈值来自数据分布（Otsu）。
- 图结构使用参数自由结构（MST）。
- 特征融合使用 rank aggregation，无手动权重。
- group refine 使用 MDL score 比较，无阈值。
- 所有尺度由图像尺寸或模型默认规则自动生成。
