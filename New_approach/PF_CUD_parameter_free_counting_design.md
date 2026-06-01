# PF-CUD：Parameter-Free Counting Unit Discovery

> 目标：在保持 **training-free / prior-free** 的前提下，进一步实现 **parameter-free** 的通用计数框架。  
> 这里的 parameter-free 指：**推理时不需要用户针对图像、数据集、类别手动调整任何阈值或超参数**。  
> 算法内部可以使用模型默认设置、图像尺寸推导量、统计分布自动估计量，例如 Otsu、MST、MDL、rank normalization。

---

## 0. 背景与核心动机

OCCAM 的主流程可以概括为：

```text
RGB Image
 ↓
SAM2 dense point prompts
 ↓
binary masks
 ↓
mask filtering
 ↓
crop object boxes
 ↓
ResNet50 feature vectors
 ↓
threshold-based FINCH
 ↓
clusters and counts
```

这个框架很强，因为它不需要训练、不需要 exemplar、不需要 text prompt，也能做 multi-class counting。

但它还有一个关键问题：  
它不是完全 parameter-free。它里面仍然有一些经验设定，例如 seed point spacing、IoU duplicate threshold、bounding box resize size、FINCH distance thresholds 等。

所以我们要跳出 OCCAM 的方向不是简单替换 FINCH，而是重新定义任务：

> 不再把 counting 看成 mask clustering，  
> 而是看成 **counting unit discovery**：自动发现图像中哪些区域构成“可数单元”。

---

## 1. 方法总览

新方法名可以暂定为：

```text
PF-CUD: Parameter-Free Counting Unit Discovery
```

完整 pipeline：

```text
Input Image
 ↓
1. Over-complete Candidate Generation
   生成尽可能完整的候选区域
 ↓
2. Candidate Canonicalization
   统一候选区域表示
 ↓
3. Feature Extraction
   提取视觉、形状、颜色、空间特征
 ↓
4. Rank-Normalized Distance Fusion
   不使用人工权重，自动融合距离
 ↓
5. MST Graph Construction
   不使用 kNN，不使用 epsilon
 ↓
6. Otsu-based Graph Cutting
   不使用 delta，不使用人工阈值
 ↓
7. Connected Components as Counting Groups
   得到初始可数单元组
 ↓
8. MDL-based Group Refinement
   用最小描述长度自动决定合并/拆分
 ↓
9. Hypothesis Ranking
   自动区分主体物体、重复花纹、背景纹理
 ↓
10. Output Counts
   输出每组数量、置信度、类型、可视化结果
```

---

## 2. 项目目录建议

建议把代码写成下面结构：

```text
pf_cud/
  __init__.py

  config.py
  data.py

  candidates/
    __init__.py
    sam_candidates.py
    blob_candidates.py
    edge_candidates.py
    merge_candidates.py

  features/
    __init__.py
    visual.py
    shape.py
    color.py
    spatial.py
    fusion.py

  graph/
    __init__.py
    mst.py
    cut.py
    components.py

  mdl/
    __init__.py
    score.py
    refine.py

  ranking/
    __init__.py
    saliency.py
    hypothesis.py

  eval/
    __init__.py
    metrics.py
    match.py

  visualize/
    __init__.py
    draw.py

  pipeline.py
  run_image.py
  run_dataset.py
```

最小可运行版本可以先只实现：

```text
data.py
candidates/sam_candidates.py
features/visual.py
features/shape.py
features/color.py
features/fusion.py
graph/mst.py
graph/cut.py
graph/components.py
mdl/score.py
ranking/hypothesis.py
pipeline.py
run_image.py
```

---

## 3. 核心数据结构

文件：`pf_cud/data.py`

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class Candidate:
    """一个候选可数区域。"""
    mask: np.ndarray                 # bool array, shape = [H, W]
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    source: str                      # "sam", "blob", "edge", etc.
    score: Optional[float] = None
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CountGroup:
    """一个最终或中间的可数单元组。"""
    indices: List[int]               # candidate indices
    group_type: Optional[str] = None  # object / pattern / background / unknown
    count: Optional[int] = None
    confidence: Optional[float] = None
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CountResult:
    """整张图片的输出结果。"""
    groups: List[CountGroup]
    candidates: List[Candidate]
    image_shape: tuple[int, int]
    meta: Dict[str, Any] = field(default_factory=dict)
```

注意：  
所有参数都不要写在 `config.py` 里让用户调。  
`config.py` 只保存模型名字、设备选择、路径，不保存阈值。

---

## 4. Step 1：Over-complete Candidate Generation

### 4.1 设计原则

不要只依赖 SAM2。  
因为 SAM2 偏向“物体级 mask”，但你的目标还包括：

```text
物体
小斑点
重复花纹
符号
局部部件
背景纹理
```

所以候选应该来自多个来源：

```text
SAM masks       → 适合完整物体
blob candidates → 适合斑点、小圆点、小目标
edge components → 适合闭合图案、符号、规则形状
```

第一版可以先实现 SAM + blob。  
等主流程跑通后再加 edge components。

---

### 4.2 SAM candidate generator

文件：`pf_cud/candidates/sam_candidates.py`

这里不要暴露 `points_per_side`、`pred_iou_thresh`、`stability_score_thresh` 给用户。  
如果用 SAM/SAM2 的 automatic mask generator，就使用官方默认或模型封装默认值。

```python
import numpy as np
from typing import List
from pf_cud.data import Candidate


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


class SAMCandidateGenerator:
    """
    SAM/SAM2 候选生成器。

    设计要求：
    - 不暴露可调阈值。
    - 使用官方 automatic mask generator 的默认配置。
    - 如果后续发现过多/过少，不通过手动调参解决，而通过后续 MST/Otsu/MDL 自动筛选。
    """

    def __init__(self, sam_model):
        self.sam_model = sam_model

    def generate(self, image_rgb: np.ndarray) -> List[Candidate]:
        """
        image_rgb: uint8 RGB image, shape [H, W, 3]
        """
        raw_masks = self.sam_model.generate(image_rgb)

        candidates: List[Candidate] = []
        for m in raw_masks:
            # 不同 SAM 实现可能字段不同，这里用常见的 segmentation 字段。
            mask = m["segmentation"].astype(bool)

            if mask.sum() == 0:
                continue

            bbox = mask_to_bbox(mask)
            candidates.append(
                Candidate(
                    mask=mask,
                    bbox=bbox,
                    source="sam",
                    score=m.get("predicted_iou", None),
                    meta={"raw_sam": m}
                )
            )

        return candidates
```

如果暂时没有 SAM2，可以用 SAM1 或者任何 segmentation proposal model 替代。  
你的方法创新不绑定某一个 mask generator，核心在后面的 parameter-free grouping。

---

### 4.3 Blob candidate generator

Blob detector 往往需要 threshold。  
为了保持 parameter-free，不能让用户设 threshold。  
做法是：

1. 把图片转灰度。
2. 构建 scale-space。
3. 找局部极大响应点。
4. 对响应值使用 Otsu 自动分割。
5. 用响应点生成圆形或椭圆形 mask。

文件：`pf_cud/candidates/blob_candidates.py`

```python
import numpy as np
from typing import List
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_laplace, maximum_filter
from pf_cud.data import Candidate
from pf_cud.candidates.sam_candidates import mask_to_bbox


def image_adaptive_sigmas(h: int, w: int) -> list[float]:
    """
    不让用户调尺度。
    根据图像尺寸自动生成尺度。
    这里的数值不是任务参数，而是图像金字塔的固定构造规则。
    """
    short = min(h, w)
    # 从很小 blob 到中等 blob，自动覆盖。
    # 使用 logspace，让尺度覆盖更均匀。
    min_sigma = max(1.0, short / 512.0)
    max_sigma = max(min_sigma * 2.0, short / 32.0)
    num = int(np.ceil(np.log2(max_sigma / min_sigma + 1))) + 4
    return np.geomspace(min_sigma, max_sigma, num=num).tolist()


def disk_mask(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


class BlobCandidateGenerator:
    """
    参数自由 blob 候选生成器。
    用 Otsu 自动决定哪些 LoG 响应值得保留。
    """

    def generate(self, image_rgb: np.ndarray) -> List[Candidate]:
        h, w = image_rgb.shape[:2]
        gray = rgb2gray(image_rgb)

        sigmas = image_adaptive_sigmas(h, w)

        responses = []
        for sigma in sigmas:
            # LoG 响应乘 sigma^2 做尺度归一化。
            resp = -gaussian_laplace(gray, sigma=sigma) * (sigma ** 2)
            responses.append(resp)

        responses = np.stack(responses, axis=0)  # [S, H, W]

        # 在 scale-space 中找局部最大。
        local_max = responses == maximum_filter(responses, size=(3, 3, 3))
        positive = responses > 0

        all_values = responses[positive]
        if all_values.size == 0:
            return []

        # Otsu 自动响应阈值。
        try:
            tau = threshold_otsu(all_values)
        except ValueError:
            return []

        keep = local_max & (responses >= tau)

        candidates = []
        scale_ids, ys, xs = np.where(keep)

        for sid, y, x in zip(scale_ids, ys, xs):
            sigma = sigmas[int(sid)]
            radius = np.sqrt(2) * sigma
            mask = disk_mask(h, w, y, x, radius)
            bbox = mask_to_bbox(mask)
            candidates.append(
                Candidate(
                    mask=mask,
                    bbox=bbox,
                    source="blob",
                    score=float(responses[sid, y, x]),
                    meta={"sigma": float(sigma), "response": float(responses[sid, y, x])}
                )
            )

        return candidates
```

这里仍然有内部尺度构造公式，但用户不需要调。  
如果论文中写作，可以表述为：

```text
The blob scales are derived from image resolution rather than manually tuned.
```

---

## 5. Step 2：Candidate Canonicalization and Deduplication

### 5.1 为什么需要去重

多来源候选会产生大量重复区域：

```text
SAM mask A 与 SAM mask B 重复
SAM mask 与 blob mask 重复
blob mask 之间重复
```

OCCAM 用固定 IoU threshold 去重。  
我们不能用固定 threshold。  
所以改成：

```text
计算所有候选之间的 IoU distance = 1 - IoU
在候选重复图上使用 MST + Otsu 自动切分
每个重复 group 只保留代表候选
```

### 5.2 代码实现

文件：`pf_cud/candidates/merge_candidates.py`

```python
import numpy as np
from typing import List
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from skimage.filters import threshold_otsu
from pf_cud.data import Candidate


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def pairwise_iou_distance(candidates: List[Candidate]) -> np.ndarray:
    n = len(candidates)
    d = np.ones((n, n), dtype=np.float32)

    for i in range(n):
        d[i, i] = 0.0
        for j in range(i + 1, n):
            iou = compute_iou(candidates[i].mask, candidates[j].mask)
            dist = 1.0 - iou
            d[i, j] = dist
            d[j, i] = dist

    return d


def auto_partition_by_mst(distance_matrix: np.ndarray) -> list[list[int]]:
    """
    对任意距离矩阵执行：
    MST -> Otsu cut -> connected components

    不需要 k、epsilon、delta。
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    mst = minimum_spanning_tree(distance_matrix).toarray()
    mst = mst + mst.T

    edge_values = mst[mst > 0]
    if len(edge_values) == 0:
        return [[i] for i in range(n)]

    if len(np.unique(edge_values)) == 1:
        # 没有自然断点。
        return [list(range(n))]

    tau = threshold_otsu(edge_values)

    # 切掉大于 tau 的边。
    kept = (mst > 0) & (mst <= tau)
    graph = kept.astype(np.int32)

    n_components, labels = connected_components(graph, directed=False)

    groups = []
    for c in range(n_components):
        groups.append(np.where(labels == c)[0].tolist())

    return groups


def choose_representative(candidates: List[Candidate], indices: list[int]) -> Candidate:
    """
    重复候选中选择代表。
    不用人工规则阈值。
    代表候选选择：
    - 优先选择 mask area 接近 group median area 的候选；
    - 如果并列，优先 SAM；
    - 如果再并列，选择 score 更高者。
    """
    areas = np.array([candidates[i].mask.sum() for i in indices], dtype=np.float64)
    med = np.median(areas)
    area_rank = np.abs(areas - med)

    source_bonus = np.array([
        0.0 if candidates[i].source == "sam" else 1.0
        for i in indices
    ])

    scores = np.array([
        -(candidates[i].score if candidates[i].score is not None else 0.0)
        for i in indices
    ])

    order = np.lexsort((scores, source_bonus, area_rank))
    return candidates[indices[int(order[0])]]


def deduplicate_candidates(candidates: List[Candidate]) -> List[Candidate]:
    if len(candidates) <= 1:
        return candidates

    d = pairwise_iou_distance(candidates)

    # 注意：重复应该对应 IoU 高，也就是 distance 小。
    # 直接对所有候选 MST 切分会把不同物体也连起来。
    # 这里采用 overlap 子图：只处理存在 IoU 的候选。
    # 是否 overlap 不是阈值，是 IoU > 0 的几何事实。
    n = len(candidates)
    visited = np.zeros(n, dtype=bool)
    final = []

    for start in range(n):
        if visited[start]:
            continue

        # 找所有与 start 有直接或间接 overlap 的候选。
        stack = [start]
        comp = []
        visited[start] = True

        while stack:
            i = stack.pop()
            comp.append(i)
            for j in range(n):
                if not visited[j] and d[i, j] < 1.0:
                    visited[j] = True
                    stack.append(j)

        if len(comp) == 1:
            final.append(candidates[comp[0]])
            continue

        sub_d = d[np.ix_(comp, comp)]
        duplicate_groups_local = auto_partition_by_mst(sub_d)

        for g in duplicate_groups_local:
            global_indices = [comp[idx] for idx in g]
            rep = choose_representative(candidates, global_indices)
            rep.meta["merged_from"] = global_indices
            final.append(rep)

    return final
```

---

## 6. Step 3：Feature Extraction

每个候选区域提取四类特征：

```text
visual feature  → DINOv2 / ResNet / CLIP
shape feature   → area, aspect ratio, compactness, Hu moments
color feature   → masked color mean/std/hist
spatial feature → center, normalized position, size
```

不要手动设置权重。  
后面用 rank-normalized fusion 自动融合。

---

### 6.1 Crop and mask utility

文件：`pf_cud/features/utils.py`

```python
import numpy as np
from PIL import Image


def crop_candidate(image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # 背景置零，减少背景干扰。
    crop[~crop_mask] = 0

    return Image.fromarray(crop)


def safe_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm
```

---

### 6.2 Visual feature

文件：`pf_cud/features/visual.py`

建议第一版用 DINOv2。  
如果环境不支持 torch hub，可以先用 torchvision ResNet50。

```python
import torch
import numpy as np
from torchvision import transforms
from typing import List
from pf_cud.data import Candidate
from pf_cud.features.utils import crop_candidate, safe_normalize


class DINOv2Extractor:
    """
    视觉特征提取器。
    不暴露输入尺寸作为调参项，直接使用 backbone 默认习惯输入。
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    @torch.no_grad()
    def extract_one(self, image_rgb: np.ndarray, cand: Candidate) -> np.ndarray:
        crop = crop_candidate(image_rgb, cand.mask, cand.bbox)
        x = self.transform(crop).unsqueeze(0).to(self.device)

        feat = self.model(x)
        if isinstance(feat, dict):
            feat = feat.get("x_norm_clstoken", list(feat.values())[0])

        feat = feat.squeeze(0).detach().cpu().numpy()
        return safe_normalize(feat)

    def attach(self, image_rgb: np.ndarray, candidates: List[Candidate]) -> None:
        for cand in candidates:
            cand.features["visual"] = self.extract_one(image_rgb, cand)
```

---

### 6.3 Shape feature

文件：`pf_cud/features/shape.py`

```python
import numpy as np
from skimage.measure import perimeter, moments_hu
from typing import List
from pf_cud.data import Candidate
from pf_cud.features.utils import safe_normalize


def shape_feature(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox

    area = float(mask.sum())
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))

    area_norm = area / float(h * w)
    aspect = box_w / box_h
    extent = area / (box_w * box_h)

    per = float(perimeter(mask))
    compactness = (4.0 * np.pi * area) / (per ** 2 + 1e-8)

    hu = moments_hu(mask.astype(float))
    hu = np.sign(hu) * np.log1p(np.abs(hu))

    vec = np.array([
        area_norm,
        np.log1p(aspect),
        extent,
        compactness,
        box_w / w,
        box_h / h,
        *hu.tolist()
    ], dtype=np.float64)

    return safe_normalize(vec)


def attach_shape_features(candidates: List[Candidate]) -> None:
    for cand in candidates:
        cand.features["shape"] = shape_feature(cand.mask, cand.bbox)
```

---

### 6.4 Color feature

文件：`pf_cud/features/color.py`

```python
import numpy as np
from typing import List
from skimage.color import rgb2lab
from pf_cud.data import Candidate
from pf_cud.features.utils import safe_normalize


def color_feature(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    不使用可调 histogram bins。
    使用 Lab 颜色的一阶/二阶统计量和分位数。
    """
    lab = rgb2lab(image_rgb)
    pixels = lab[mask]

    if pixels.size == 0:
        return np.zeros(15, dtype=np.float64)

    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    q25 = np.quantile(pixels, 0.25, axis=0)
    q50 = np.quantile(pixels, 0.50, axis=0)
    q75 = np.quantile(pixels, 0.75, axis=0)

    vec = np.concatenate([mean, std, q25, q50, q75]).astype(np.float64)
    return safe_normalize(vec)


def attach_color_features(image_rgb: np.ndarray, candidates: List[Candidate]) -> None:
    for cand in candidates:
        cand.features["color"] = color_feature(image_rgb, cand.mask)
```

---

### 6.5 Spatial feature

文件：`pf_cud/features/spatial.py`

```python
import numpy as np
from typing import List
from pf_cud.data import Candidate
from pf_cud.features.utils import safe_normalize


def spatial_feature(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    # 中心性，不是阈值，只是几何属性。
    center_dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    vec = np.array([cx, cy, bw, bh, center_dist], dtype=np.float64)
    return safe_normalize(vec)


def attach_spatial_features(candidates: List[Candidate]) -> None:
    for cand in candidates:
        cand.features["spatial"] = spatial_feature(cand.mask, cand.bbox)
```

---

## 7. Step 4：Rank-Normalized Distance Fusion

### 7.1 为什么不用人工权重

传统做法可能会写：

```python
D = alpha * D_visual + beta * D_shape + gamma * D_color
```

但这需要调 `alpha, beta, gamma`。

我们改成：

```text
每一种距离矩阵先转成 rank matrix
最后对 rank 取平均
```

这样不同特征的数值尺度不会影响结果，也不需要权重。

---

### 7.2 代码实现

文件：`pf_cud/features/fusion.py`

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import List
from pf_cud.data import Candidate


def pairwise_feature_distance(candidates: List[Candidate], key: str, metric: str = "euclidean") -> np.ndarray:
    feats = np.stack([cand.features[key] for cand in candidates], axis=0)
    if len(feats) <= 1:
        return np.zeros((len(feats), len(feats)), dtype=np.float64)

    d = squareform(pdist(feats, metric=metric))
    return d.astype(np.float64)


def rank_normalize_distance(d: np.ndarray) -> np.ndarray:
    """
    把距离矩阵转换成 [0, 1] rank。
    不使用 min-max，因为 min/max 可能受 outlier 影响。
    """
    n = d.shape[0]
    if n <= 1:
        return d.copy()

    tri = np.triu_indices(n, k=1)
    values = d[tri]

    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    if len(values) > 1:
        ranks = ranks / (len(values) - 1)
    else:
        ranks = np.zeros_like(ranks, dtype=np.float64)

    out = np.zeros_like(d, dtype=np.float64)
    out[tri] = ranks
    out = out + out.T

    return out


def fused_distance(candidates: List[Candidate]) -> np.ndarray:
    """
    无权重距离融合。
    每种 feature 等价参与，但不是通过原始数值平均，而是 rank 平均。
    """
    keys = ["visual", "shape", "color", "spatial"]

    rank_mats = []
    for key in keys:
        d = pairwise_feature_distance(candidates, key)
        rank_mats.append(rank_normalize_distance(d))

    fused = np.mean(np.stack(rank_mats, axis=0), axis=0)

    # 保证对角线为 0。
    np.fill_diagonal(fused, 0.0)
    return fused
```

---

## 8. Step 5：MST Graph Construction

### 8.1 为什么用 MST

不要用：

```text
kNN graph     → 需要 k
epsilon graph → 需要 epsilon
radius graph  → 需要半径
```

改用：

```text
Minimum Spanning Tree
```

优点：

```text
不需要 k
不需要 epsilon
所有候选都会进入图
边由数据结构自动决定
```

### 8.2 代码实现

文件：`pf_cud/graph/mst.py`

```python
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def build_mst(distance_matrix: np.ndarray) -> np.ndarray:
    """
    返回对称 MST adjacency matrix。
    matrix[i, j] = edge weight or 0.
    """
    if distance_matrix.shape[0] <= 1:
        return np.zeros_like(distance_matrix)

    mst = minimum_spanning_tree(distance_matrix).toarray()
    mst = mst + mst.T
    return mst.astype(np.float64)
```

---

## 9. Step 6：Otsu-based Graph Cutting

### 9.1 为什么用 Otsu

MST 上边权通常会出现：

```text
短边：类内连接
长边：类间连接
```

Otsu 可以自动找到边权分布的自然分割点。

```text
不需要 delta
不需要 threshold
不需要 tuning
```

### 9.2 代码实现

文件：`pf_cud/graph/cut.py`

```python
import numpy as np
from skimage.filters import threshold_otsu


def otsu_cut_mst(mst: np.ndarray) -> np.ndarray:
    """
    输入 MST adjacency matrix。
    输出 cut 后的 adjacency matrix。
    """
    graph = mst.copy()
    edge_values = graph[graph > 0]

    if len(edge_values) == 0:
        return graph

    unique = np.unique(edge_values)

    if len(unique) == 1:
        # 没有自然断点，保留原 MST。
        return graph

    tau = threshold_otsu(edge_values)

    # 大于 tau 的边被认为是跨类长边。
    graph[graph > tau] = 0.0

    return graph
```

---

## 10. Step 7：Connected Components 得到初始组

文件：`pf_cud/graph/components.py`

```python
import numpy as np
from scipy.sparse.csgraph import connected_components
from typing import List
from pf_cud.data import CountGroup


def graph_to_groups(graph: np.ndarray) -> List[CountGroup]:
    n = graph.shape[0]
    if n == 0:
        return []

    if n == 1:
        return [CountGroup(indices=[0], count=1)]

    adjacency = (graph > 0).astype(np.int32)

    n_components, labels = connected_components(adjacency, directed=False)

    groups = []
    for c in range(n_components):
        inds = np.where(labels == c)[0].tolist()
        groups.append(CountGroup(indices=inds, count=len(inds)))

    return groups
```

---

## 11. Step 8：MDL-based Group Refinement

### 11.1 为什么需要 MDL

MST + Otsu 会给出初始 group，但可能出现：

```text
同类被拆开
不同类被合并
花纹和物体混在一起
背景纹理形成 group
```

不能用手动阈值修正。  
所以用 MDL。

核心思想：

> 一个好的 counting group 应该可以被“一个 prototype + 多个 instance variation”压缩表示。

如果一组候选真的属于同一类，它们的 feature residual 应该小。  
如果合并两个 group 后 residual 大幅增加，那么不该合并。  
如果拆开后总描述长度变短，那么应该拆开。

---

### 11.2 Group MDL score

文件：`pf_cud/mdl/score.py`

```python
import numpy as np
from typing import List
from pf_cud.data import Candidate, CountGroup


def stack_group_features(candidates: List[Candidate], group: CountGroup, key: str) -> np.ndarray:
    return np.stack([candidates[i].features[key] for i in group.indices], axis=0)


def gaussian_residual_code_length(x: np.ndarray) -> float:
    """
    用高斯残差近似描述长度。
    不需要手动阈值。
    """
    if len(x) <= 1:
        return 0.0

    mu = x.mean(axis=0, keepdims=True)
    residual = x - mu

    # variance 由数据自己估计。
    var = np.mean(residual ** 2) + 1e-12

    # negative log likelihood up to constant。
    n, d = x.shape
    return 0.5 * n * d * np.log(var) + 0.5 * np.sum(residual ** 2) / var


def prototype_cost(x: np.ndarray) -> float:
    """
    prototype 的复杂度。
    维度越高成本越高。
    这里不是调参，而是编码长度的自然项。
    """
    if x.ndim != 2:
        return 0.0
    _, d = x.shape
    return float(d * np.log(2.0 + x.shape[0]))


def group_mdl(candidates: List[Candidate], group: CountGroup) -> float:
    """
    一个 group 的总描述长度。
    visual/shape/color/spatial 全部参与。
    不使用人工权重，直接求和。
    """
    if len(group.indices) == 0:
        return 0.0

    total = 0.0
    for key in ["visual", "shape", "color", "spatial"]:
        x = stack_group_features(candidates, group, key)
        total += prototype_cost(x)
        total += gaussian_residual_code_length(x)

    # group membership 的编码成本。
    # 数量越多，编码成本自然增加。
    total += len(group.indices) * np.log(2.0 + len(candidates))

    return float(total)


def total_mdl(candidates: List[Candidate], groups: List[CountGroup]) -> float:
    return float(sum(group_mdl(candidates, g) for g in groups))
```

---

### 11.3 MDL merge refinement

基本策略：

```text
对任意两个 group：
    计算 merge 前 MDL
    计算 merge 后 MDL
    如果 merge 后更短，则合并
重复直到没有合并能降低 MDL
```

没有阈值。  
只有比较：

```text
new_mdl < old_mdl
```

文件：`pf_cud/mdl/refine.py`

```python
from typing import List
from pf_cud.data import Candidate, CountGroup
from pf_cud.mdl.score import group_mdl


def merge_two_groups(a: CountGroup, b: CountGroup) -> CountGroup:
    return CountGroup(indices=sorted(a.indices + b.indices), count=len(a.indices) + len(b.indices))


def mdl_merge_refinement(candidates: List[Candidate], groups: List[CountGroup]) -> List[CountGroup]:
    """
    贪心 MDL merge。
    不需要 merge threshold。
    """
    groups = groups[:]
    changed = True

    while changed:
        changed = False
        best_pair = None
        best_gain = 0.0

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                old_score = group_mdl(candidates, groups[i]) + group_mdl(candidates, groups[j])
                merged = merge_two_groups(groups[i], groups[j])
                new_score = group_mdl(candidates, merged)
                gain = old_score - new_score

                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j, merged)

        if best_pair is not None:
            i, j, merged = best_pair
            new_groups = []
            for idx, g in enumerate(groups):
                if idx not in (i, j):
                    new_groups.append(g)
            new_groups.append(merged)
            groups = new_groups
            changed = True

    for g in groups:
        g.count = len(g.indices)

    return groups
```

第一版可以只做 merge，不做 split。  
因为 MST + Otsu 已经倾向于切分，MDL merge 可以修复过度切分。

后续增强版本可以加入 split refinement：

```text
对每个 group 内部再跑 MST + Otsu
如果拆分后 total MDL 更低，则接受拆分
```

---

## 12. Step 9：Hypothesis Ranking

### 12.1 为什么需要 ranking

Prior-free counting 的最大问题是：

```text
图里到底应该数什么？
```

例如：

```text
苹果 vs 砖块
眼镜 vs 镜片
草莓 vs 盘子花纹
水印 vs 真实物体
```

所以不要只输出一个 count。  
应该输出多个 hypothesis：

```text
Group 1: likely main object, count = 12
Group 2: likely pattern, count = 36
Group 3: likely background texture, count = 58
```

---

### 12.2 Group attributes

每个 group 计算：

```text
repeatability      数量越多越可能是可数重复单元
visual_consistency 组内视觉残差越小越好
shape_consistency  组内形状残差越小越好
centrality         是否靠近图像主体区域
area_balance       面积是否稳定
backgroundness     是否像背景纹理
```

不能写人工权重。  
用 rank aggregation。

文件：`pf_cud/ranking/hypothesis.py`

```python
import numpy as np
from typing import List
from pf_cud.data import Candidate, CountGroup


def group_centers(candidates: List[Candidate], group: CountGroup) -> np.ndarray:
    centers = []
    h, w = candidates[0].mask.shape[:2]
    for idx in group.indices:
        x1, y1, x2, y2 = candidates[idx].bbox
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        centers.append([cx, cy])
    return np.array(centers, dtype=np.float64)


def group_areas(candidates: List[Candidate], group: CountGroup) -> np.ndarray:
    h, w = candidates[0].mask.shape[:2]
    return np.array([
        candidates[idx].mask.sum() / float(h * w)
        for idx in group.indices
    ], dtype=np.float64)


def feature_residual(candidates: List[Candidate], group: CountGroup, key: str) -> float:
    if len(group.indices) <= 1:
        return float("inf")

    x = np.stack([candidates[i].features[key] for i in group.indices], axis=0)
    mu = x.mean(axis=0, keepdims=True)
    return float(np.mean((x - mu) ** 2))


def compute_group_raw_scores(candidates: List[Candidate], group: CountGroup) -> dict[str, float]:
    centers = group_centers(candidates, group)
    areas = group_areas(candidates, group)

    count = len(group.indices)

    # 数量越多，重复性越强。
    repeatability = np.log1p(count)

    # 越靠中心越像主体。这里不是阈值，只是几何先验。
    center_dists = np.sqrt(((centers - 0.5) ** 2).sum(axis=1))
    centrality = -float(np.mean(center_dists))

    # 面积变化越小，越像同一类。
    area_consistency = -float(np.std(areas) / (np.mean(areas) + 1e-12))

    visual_consistency = -feature_residual(candidates, group, "visual")
    shape_consistency = -feature_residual(candidates, group, "shape")
    color_consistency = -feature_residual(candidates, group, "color")

    # 背景纹理通常数量极多、面积极小、空间铺满全图。
    # 这里先只计算属性，不做硬阈值。
    spatial_spread = float(np.linalg.det(np.cov(centers.T) + np.eye(2) * 1e-6)) if len(centers) > 1 else 0.0
    backgroundness = spatial_spread + repeatability - abs(centrality)

    return {
        "repeatability": repeatability,
        "centrality": centrality,
        "area_consistency": area_consistency,
        "visual_consistency": visual_consistency,
        "shape_consistency": shape_consistency,
        "color_consistency": color_consistency,
        "backgroundness": backgroundness,
    }


def rank_values(values: list[float], higher_is_better: bool = True) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    order = np.argsort(arr)
    if higher_is_better:
        order = order[::-1]

    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)

    if len(arr) > 1:
        ranks = 1.0 - ranks / (len(arr) - 1)
    else:
        ranks = np.ones_like(ranks, dtype=np.float64)

    return ranks


def rank_groups(candidates: List[Candidate], groups: List[CountGroup]) -> List[CountGroup]:
    if not groups:
        return groups

    raw = [compute_group_raw_scores(candidates, g) for g in groups]

    keys_for_main_object = [
        "repeatability",
        "centrality",
        "area_consistency",
        "visual_consistency",
        "shape_consistency",
        "color_consistency",
    ]

    rank_mats = []
    for key in keys_for_main_object:
        vals = [r[key] for r in raw]
        rank_mats.append(rank_values(vals, higher_is_better=True))

    main_scores = np.mean(np.stack(rank_mats, axis=0), axis=0)

    # pattern/background classification 也用 rank，不用阈值。
    bg_rank = rank_values([r["backgroundness"] for r in raw], higher_is_better=True)

    for i, g in enumerate(groups):
        g.score = float(main_scores[i])
        g.confidence = float(main_scores[i])

        # 类型不要用硬 threshold，而用相对排名。
        if bg_rank[i] == bg_rank.max() and len(groups) > 1:
            g.group_type = "background_or_pattern"
        else:
            g.group_type = "object_or_counting_unit"

        g.meta["raw_scores"] = raw[i]
        g.meta["main_rank_score"] = float(main_scores[i])
        g.meta["background_rank_score"] = float(bg_rank[i])

    groups = sorted(groups, key=lambda g: g.score if g.score is not None else -1, reverse=True)
    return groups
```

注意：  
这里的 `group_type` 不要过度自信。  
在 prior-free 设定下，最诚实的输出是：

```text
object_or_counting_unit
background_or_pattern
unknown
```

而不是强行说“这是苹果”。

---

## 13. Step 10：完整 Pipeline

文件：`pf_cud/pipeline.py`

```python
import numpy as np
from pf_cud.data import CountResult
from pf_cud.candidates.sam_candidates import SAMCandidateGenerator
from pf_cud.candidates.blob_candidates import BlobCandidateGenerator
from pf_cud.candidates.merge_candidates import deduplicate_candidates
from pf_cud.features.visual import DINOv2Extractor
from pf_cud.features.shape import attach_shape_features
from pf_cud.features.color import attach_color_features
from pf_cud.features.spatial import attach_spatial_features
from pf_cud.features.fusion import fused_distance
from pf_cud.graph.mst import build_mst
from pf_cud.graph.cut import otsu_cut_mst
from pf_cud.graph.components import graph_to_groups
from pf_cud.mdl.refine import mdl_merge_refinement
from pf_cud.ranking.hypothesis import rank_groups


class PFCUDPipeline:
    """
    Parameter-Free Counting Unit Discovery pipeline.

    用户只需要输入 image。
    不需要设置 epsilon、delta、k、IoU threshold、FINCH threshold。
    """

    def __init__(self, sam_model=None, visual_extractor=None):
        self.sam_generator = SAMCandidateGenerator(sam_model) if sam_model is not None else None
        self.blob_generator = BlobCandidateGenerator()
        self.visual_extractor = visual_extractor or DINOv2Extractor()

    def generate_candidates(self, image_rgb: np.ndarray):
        candidates = []

        if self.sam_generator is not None:
            candidates.extend(self.sam_generator.generate(image_rgb))

        candidates.extend(self.blob_generator.generate(image_rgb))

        candidates = deduplicate_candidates(candidates)
        return candidates

    def attach_features(self, image_rgb: np.ndarray, candidates):
        self.visual_extractor.attach(image_rgb, candidates)
        attach_shape_features(candidates)
        attach_color_features(image_rgb, candidates)
        attach_spatial_features(candidates)

    def run(self, image_rgb: np.ndarray) -> CountResult:
        candidates = self.generate_candidates(image_rgb)

        if len(candidates) == 0:
            return CountResult(
                groups=[],
                candidates=[],
                image_shape=image_rgb.shape[:2],
                meta={"status": "no_candidates"}
            )

        self.attach_features(image_rgb, candidates)

        d = fused_distance(candidates)
        mst = build_mst(d)
        cut_graph = otsu_cut_mst(mst)

        groups = graph_to_groups(cut_graph)
        groups = mdl_merge_refinement(candidates, groups)
        groups = rank_groups(candidates, groups)

        return CountResult(
            groups=groups,
            candidates=candidates,
            image_shape=image_rgb.shape[:2],
            meta={
                "num_candidates": len(candidates),
                "num_groups": len(groups),
            }
        )
```

---

## 14. 单图运行脚本

文件：`pf_cud/run_image.py`

```python
import argparse
import json
import numpy as np
from PIL import Image

from pf_cud.pipeline import PFCUDPipeline
from pf_cud.visualize.draw import draw_result


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def result_to_jsonable(result):
    out = []
    for rank, g in enumerate(result.groups):
        out.append({
            "rank": rank + 1,
            "count": len(g.indices),
            "group_type": g.group_type,
            "confidence": g.confidence,
            "candidate_indices": g.indices,
            "score": g.score,
            "meta": g.meta,
        })
    return {
        "image_shape": result.image_shape,
        "num_candidates": len(result.candidates),
        "groups": out,
        "meta": result.meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_json", default="result.json")
    parser.add_argument("--out_vis", default="result.png")
    args = parser.parse_args()

    image = load_image(args.image)

    # sam_model 可以后续注入。
    # 第一版没有 SAM 也可以只跑 blob + graph。
    pipeline = PFCUDPipeline(sam_model=None)

    result = pipeline.run(image)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result_to_jsonable(result), f, indent=2, ensure_ascii=False)

    vis = draw_result(image, result)
    Image.fromarray(vis).save(args.out_vis)

    print(json.dumps(result_to_jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

运行：

```bash
python -m pf_cud.run_image \
  --image examples/test.jpg \
  --out_json outputs/test_result.json \
  --out_vis outputs/test_result.png
```

注意：  
这个命令没有任何需要调的参数。

---

## 15. 可视化代码

文件：`pf_cud/visualize/draw.py`

```python
import numpy as np
from PIL import Image, ImageDraw
from pf_cud.data import CountResult


def draw_result(image_rgb: np.ndarray, result: CountResult) -> np.ndarray:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 固定颜色表不是算法参数，只是可视化。
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]

    for rank, group in enumerate(result.groups):
        color = colors[rank % len(colors)]

        for idx in group.indices:
            cand = result.candidates[idx]
            x1, y1, x2, y2 = cand.bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"#{rank + 1}: count={len(group.indices)}, {group.group_type}"
        draw.text((8, 8 + rank * 16), label, fill=color)

    return np.array(img)
```

---

## 16. Dataset Evaluation

### 16.1 Count metrics

文件：`pf_cud/eval/metrics.py`

```python
import numpy as np


def mae(pred_counts: list[int], gt_counts: list[int]) -> float:
    return float(np.mean([abs(p - g) for p, g in zip(pred_counts, gt_counts)]))


def rmse(pred_counts: list[int], gt_counts: list[int]) -> float:
    return float(np.sqrt(np.mean([(p - g) ** 2 for p, g in zip(pred_counts, gt_counts)])))


def nae(pred_counts: list[int], gt_counts: list[int]) -> float:
    vals = []
    for p, g in zip(pred_counts, gt_counts):
        vals.append(abs(p - g) / max(1, g))
    return float(np.mean(vals))


def sre(pred_counts: list[int], gt_counts: list[int]) -> float:
    vals = []
    for p, g in zip(pred_counts, gt_counts):
        vals.append(((p - g) ** 2) / max(1, g))
    return float(np.mean(vals))
```

### 16.2 多类别 matching

如果 ground truth 是多组 count，比如：

```json
{
  "apple": 12,
  "orange": 8,
  "strawberry": 15
}
```

而你的模型输出的是没有类别名的 groups：

```json
[
  {"count": 11},
  {"count": 16},
  {"count": 7}
]
```

就需要 matching。  
可以用 Hungarian matching 最小化 count difference。

文件：`pf_cud/eval/match.py`

```python
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_counts(pred_counts: list[int], gt_counts: list[int]):
    if len(pred_counts) == 0 or len(gt_counts) == 0:
        return [], list(range(len(pred_counts))), list(range(len(gt_counts)))

    cost = np.zeros((len(pred_counts), len(gt_counts)), dtype=np.float64)

    for i, p in enumerate(pred_counts):
        for j, g in enumerate(gt_counts):
            cost[i, j] = abs(p - g)

    row, col = linear_sum_assignment(cost)

    matched = list(zip(row.tolist(), col.tolist()))
    unmatched_pred = sorted(set(range(len(pred_counts))) - set(row.tolist()))
    unmatched_gt = sorted(set(range(len(gt_counts))) - set(col.tolist()))

    return matched, unmatched_pred, unmatched_gt
```

---

## 17. 如何保证“没有参数需要调整”

你的代码规范应该写成：

```text
1. CLI 不允许出现与算法阈值相关的参数。
2. 所有阈值必须来自数据分布，例如 Otsu。
3. 所有图结构必须来自参数自由结构，例如 MST。
4. 所有 feature fusion 不允许手动权重，使用 rank aggregation。
5. 所有 group refine 不允许阈值，使用 MDL score comparison。
6. 所有尺度必须由图像尺寸或模型默认规则自动生成。
```

### 禁止出现的接口

不要写：

```python
parser.add_argument("--epsilon")
parser.add_argument("--delta")
parser.add_argument("--k")
parser.add_argument("--iou_thresh")
parser.add_argument("--score_thresh")
parser.add_argument("--num_scales")
parser.add_argument("--finch_thresh")
```

### 允许出现的接口

可以写：

```python
parser.add_argument("--image")
parser.add_argument("--out_json")
parser.add_argument("--out_vis")
parser.add_argument("--device")
parser.add_argument("--model_path")
```

因为这些不是算法调参，只是工程配置。

---

## 18. 第一版最小实现路线

不要一开始就实现所有东西。  
建议分 4 个阶段。

---

### Phase 1：跑通 parameter-free graph counting

实现：

```text
blob candidates
shape/color/spatial features
rank fusion
MST
Otsu cut
connected components
visualization
```

暂时不加 DINO，不加 SAM。

目标：

```text
能在斑点、圆点、小颗粒图像上工作。
```

---

### Phase 2：加入 DINOv2 visual feature

实现：

```text
DINOv2 feature extractor
visual + shape + color + spatial rank fusion
```

目标：

```text
能处理外观相似但颜色/形状有变化的物体。
```

---

### Phase 3：加入 SAM/SAM2 candidates

实现：

```text
SAM candidate generator
candidate deduplication by MST + Otsu
```

目标：

```text
能处理真实物体，例如水果、车、动物、工具。
```

---

### Phase 4：加入 MDL refinement and hypothesis ranking

实现：

```text
MDL merge
optional MDL split
group ranking
object/pattern/background type
```

目标：

```text
能输出多个 counting hypotheses，
并把主体物体排在前面。
```

---

## 19. 方法创新点写法

可以这样写在论文或者 proposal 里：

```text
We formulate prior-free object counting as a parameter-free counting unit discovery problem.
Instead of clustering instance masks using empirically chosen thresholds, we construct an
over-complete set of candidate counting units and infer their grouping through a self-calibrated
graph process. Pairwise similarities from visual, geometric, color, and spatial cues are fused
by rank aggregation, avoiding manually tuned feature weights. A minimum spanning tree is then
built over the fused distances, eliminating the need for k-nearest-neighbor or epsilon graphs.
Graph cutting is performed by Otsu thresholding over MST edge weights, and group refinement is
guided by a minimum-description-length objective. The final output consists of ranked counting
hypotheses, allowing the method to separate dominant objects, repeated patterns, and background
textures without class labels, exemplars, prompts, training, or dataset-specific hyperparameters.
```

中文版本：

```text
我们将 prior-free object counting 重新定义为 parameter-free counting unit discovery。
不同于使用经验阈值对 instance masks 聚类，我们首先生成过完备候选可数单元，
然后通过自校准图推理自动发现重复结构。视觉、形状、颜色和空间特征通过
rank aggregation 融合，避免人工设置特征权重。随后在融合距离上构建 MST，
避免 kNN 或 epsilon graph 的参数选择。图切分通过 MST 边权的 Otsu 自适应阈值完成，
并使用 MDL 目标自动进行 group refinement。最终输出 ranked counting hypotheses，
从而在不使用类别标签、exemplar、prompt、训练或数据集特定超参数的情况下，
区分主体物体、重复花纹和背景纹理。
```

---

## 20. 与 OCCAM 的对比表

| Component | OCCAM | PF-CUD |
|---|---|---|
| Candidate source | mainly SAM2 masks | SAM/SAM2 + blob + edge candidates |
| Mask filtering | empirical rules | MST/Otsu deduplication |
| Feature | ResNet50 | DINOv2 + shape + color + spatial |
| Feature fusion | single embedding | rank-normalized multi-cue fusion |
| Graph | FINCH-like nearest-neighbor hierarchy | MST |
| Cut criterion | empirical thresholds | Otsu on MST edge distribution |
| Cluster refinement | threshold-based stopping | MDL score comparison |
| Main object selection | not explicit | ranked counting hypotheses |
| Parameters to tune | yes | no user-tuned inference parameters |
| Output | cluster counts | count + group type + confidence |

---

## 21. Potential Failure Cases

即使 parameter-free，方法仍然会失败。  
需要提前写清楚。

### 21.1 极端遮挡

同类物体外观差异太大，可能被拆开。

解决方向：

```text
引入 viewpoint-invariant visual feature
用 MDL merge 修复过度拆分
```

### 21.2 背景纹理比主体更重复

例如砖墙、棋盘格、布料纹理。  
模型可能把背景 pattern 排得很高。

解决方向：

```text
输出多 hypothesis
不要只输出一个答案
加入 saliency / foregroundness rank
```

### 21.3 候选生成阶段漏掉目标

如果 SAM 和 blob 都没有生成目标区域，后面无法恢复。

解决方向：

```text
加入 edge candidates
加入 DINO attention proposals
加入 connected component proposals
```

### 21.4 Prior-free 本身的语义歧义

没有 prompt 时，图像中“应该数什么”本来就不唯一。  
所以输出应该是：

```text
ranked hypotheses
```

而不是强行一个 single answer。

---

## 22. 后续可以增强的模块

### 22.1 Edge candidates

用 Canny + closed contour 生成候选。  
但 Canny 也有阈值。  
要保持 parameter-free，可以用：

```text
Canny thresholds from image gradient quantiles or Otsu
closed contours from connected components
```

### 22.2 DINO attention candidates

用 DINO patch attention 产生 salient regions。  
优势：

```text
更能找到主体物体
对背景纹理有一定抑制
```

### 22.3 MDL split

目前只写了 MDL merge。  
后续可以加入：

```text
对每个 group 内部再用 MST/Otsu split
如果 split 后 total MDL 更低，则接受
```

### 22.4 Multi-level output

输出三层：

```text
main_object_count
pattern_count
all_repetition_groups
```

例如：

```json
{
  "main_object_count": 12,
  "groups": [
    {"type": "object_or_counting_unit", "count": 12, "confidence": 0.91},
    {"type": "background_or_pattern", "count": 36, "confidence": 0.73}
  ]
}
```

---

## 23. 最小可交付版本 Checklist

### 代码文件

```text
pf_cud/data.py
pf_cud/candidates/blob_candidates.py
pf_cud/candidates/merge_candidates.py
pf_cud/features/shape.py
pf_cud/features/color.py
pf_cud/features/spatial.py
pf_cud/features/fusion.py
pf_cud/graph/mst.py
pf_cud/graph/cut.py
pf_cud/graph/components.py
pf_cud/mdl/score.py
pf_cud/mdl/refine.py
pf_cud/ranking/hypothesis.py
pf_cud/visualize/draw.py
pf_cud/pipeline.py
pf_cud/run_image.py
```

### 功能

```text
[ ] 输入一张图片
[ ] 自动生成 candidates
[ ] 自动提取 features
[ ] 自动构建 MST
[ ] 自动 Otsu cut
[ ] 自动 connected components
[ ] 自动 MDL merge
[ ] 自动 rank groups
[ ] 输出 JSON
[ ] 输出 visualization
[ ] CLI 无算法参数
```

### 命令

```bash
python -m pf_cud.run_image \
  --image examples/test.jpg \
  --out_json outputs/result.json \
  --out_vis outputs/result.png
```

---

## 24. 最终总结

这个版本的核心思想是：

> 把 OCCAM 的 “SAM2 + threshold FINCH”  
> 升级为  
> “over-complete candidates + rank-fused similarity + MST/Otsu graph cutting + MDL hypothesis selection”。

它的优势不是某个模块更复杂，而是整体任务定义变了：

```text
OCCAM:
  classify masks into clusters

PF-CUD:
  discover countable repeated units without tuned parameters
```

最重要的是：

```text
不需要 epsilon
不需要 delta
不需要 k
不需要 IoU threshold
不需要 FINCH threshold
不需要用户判断物体还是花纹
```

最终输出不是一个死板 count，而是多个 ranked counting hypotheses。  
这更符合 prior-free counting 的真实情况，也更适合你想解决的斑点、花纹、物体、背景纹理混合场景。
