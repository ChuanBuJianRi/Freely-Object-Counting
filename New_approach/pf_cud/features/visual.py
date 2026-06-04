"""Visual feature extraction.

Default backbone is DINOv2 (torch hub). If torch hub is unavailable, a
torchvision ResNet50 fallback is used. Input size is not exposed as a tuning
knob; we use each backbone's conventional input size.
"""

from typing import List, Optional

import numpy as np
import torch
from torchvision import transforms

from pf_cud.data import Candidate
from pf_cud.features.utils import crop_candidate, safe_normalize

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _default_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class _VisualBackbone:
    """Common interface for visual extractors."""

    device: str
    transform: transforms.Compose
    batch_size: int = 128

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def extract_one(self, image_rgb: np.ndarray, cand: Candidate) -> np.ndarray:
        crop = crop_candidate(image_rgb, cand.mask, cand.bbox)
        x = self.transform(crop).unsqueeze(0).to(self.device)
        feat = self._forward(x)
        feat = feat.squeeze(0).detach().cpu().numpy().ravel()
        return safe_normalize(feat)

    @torch.no_grad()
    def attach(self, image_rgb: np.ndarray, candidates: List[Candidate]) -> None:
        if not candidates:
            return
        # Build all crops on CPU (PIL crop is cheap; the slow part used to be the
        # per-crop CPU Resize/Normalize). We move the whole batch of uint8 crops
        # to the GPU and do Resize/CenterCrop/Normalize there, which is much
        # faster and keeps the GPU busy. Numerically equivalent to the previous
        # torchvision transform pipeline (bilinear resize + imagenet normalize).
        import torch.nn.functional as F
        from torchvision.transforms.functional import resize as tv_resize

        device = self.device
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        size = 224

        crops = [crop_candidate(image_rgb, c.mask, c.bbox) for c in candidates]

        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start : start + self.batch_size]
            batch = []
            for img in chunk:
                arr = np.asarray(img, dtype=np.uint8)
                t = torch.from_numpy(arr).to(device).permute(2, 0, 1).float().div_(255.0)
                # Resize shorter side to ``size`` then center-crop to size x size.
                t = tv_resize(t, [size], antialias=True)
                _, hh, ww = t.shape
                top = max(0, (hh - size) // 2)
                left = max(0, (ww - size) // 2)
                t = t[:, top : top + size, left : left + size]
                if t.shape[1] != size or t.shape[2] != size:
                    t = F.interpolate(
                        t.unsqueeze(0), size=(size, size), mode="bilinear",
                        align_corners=False, antialias=True,
                    ).squeeze(0)
                batch.append(t)
            x = torch.stack(batch, dim=0)
            x = (x - mean) / std
            feats = self._forward(x).detach().cpu().numpy()
            feats = feats.reshape(feats.shape[0], -1)
            for k, cand in enumerate(candidates[start : start + self.batch_size]):
                cand.features["visual"] = safe_normalize(feats[k])


class DINOv2Extractor(_VisualBackbone):
    """视觉特征提取器，默认 DINOv2，必要时回退 ResNet50。"""

    def __init__(self, model_name: str = "dinov2_vits14", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = _default_transform(224)

        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model(x)
        if isinstance(feat, dict):
            feat = feat.get("x_norm_clstoken", list(feat.values())[0])
        return feat


class ResNet50Extractor(_VisualBackbone):
    """torchvision ResNet50 backbone (penultimate global pooled features)."""

    def __init__(self, device: Optional[str] = None):
        from torchvision.models import ResNet50_Weights, resnet50

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = _default_transform(224)

        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        self.model = net.eval().to(self.device)

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NullVisualExtractor(_VisualBackbone):
    """No-network fallback producing a constant visual feature.

    Useful in restricted environments (Phase 1: blob + shape/color/spatial),
    where the design explicitly allows running without DINO/SAM. The constant
    feature means visual distances are all zero and contribute neutrally to the
    rank-fused distance.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"

    def extract_one(self, image_rgb: np.ndarray, cand: Candidate) -> np.ndarray:
        return np.zeros(1, dtype=np.float64)

    def attach(self, image_rgb: np.ndarray, candidates: List[Candidate]) -> None:
        for cand in candidates:
            cand.features["visual"] = np.zeros(1, dtype=np.float64)


def _network_available(timeout: float = 2.0) -> bool:
    """Quick reachability probe so offline runs fail fast instead of hanging."""
    import socket

    for host, port in (("github.com", 443), ("download.pytorch.org", 443)):
        try:
            socket.setdefaulttimeout(timeout)
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def build_visual_extractor(
    model_name: str = "dinov2_vits14", device: Optional[str] = None
) -> _VisualBackbone:
    """Best-effort visual extractor with graceful degradation.

    Tries DINOv2 -> ResNet50 -> Null. Selection is engineering robustness, not
    an algorithm tuning parameter. When no network is available the loaders
    would otherwise download weights and hang, so we skip straight to the
    no-network fallback (Phase 1: blob + shape/color/spatial still run).
    """
    if _network_available():
        try:
            return DINOv2Extractor(model_name=model_name, device=device)
        except Exception:
            pass
        try:
            return ResNet50Extractor(device=device)
        except Exception:
            pass
    return NullVisualExtractor(device=device)
