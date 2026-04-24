"""
HoVer-Net style nucleus detection: U-Net backbone + hover map regression + watershed.

Hover maps (horizontal/vertical distance from nucleus centroid) enable robust
instance segmentation without NMS — each nucleus gets an exact pixel-level mask.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Import watershed dependencies at module level
try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    _HAS_WATERSHED = True
except ImportError:
    _HAS_WATERSHED = False
    logger.warning(
        "skimage not installed — using scipy.ndimage.label fallback. "
        "Install scikit-image for better instance segmentation: pip install scikit-image"
    )


@dataclass
class DetectionResult:
    """Nucleus detection output for a single image patch."""

    centroids: np.ndarray  # (N, 2) float32, (row, col) in patch pixels
    masks: np.ndarray  # (N, H, W) bool instance masks
    probabilities: np.ndarray  # (N,) detection confidence
    cell_types: Optional[np.ndarray] = None  # (N,) int class indices
    hover_maps: Optional[np.ndarray] = None  # (2, H, W) HoV distance maps

    @property
    def count(self) -> int:
        return len(self.centroids)

    def filter_by_confidence(self, threshold: float = 0.5) -> "DetectionResult":
        keep = self.probabilities >= threshold
        return DetectionResult(
            centroids=self.centroids[keep],
            masks=self.masks[keep],
            probabilities=self.probabilities[keep],
            cell_types=self.cell_types[keep] if self.cell_types is not None else None,
        )


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class HoverNet(nn.Module):
    """
    Simplified HoVer-Net: shared U-Net encoder, three decoder heads.
      np_head — nuclear pixel binary map (2-class)
      hv_head — horizontal/vertical centroid distance maps (tanh)
      tp_head — nuclear type softmax (optional)
    """

    def __init__(self, in_channels: int = 3, num_types: int = 0, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.inc = _DoubleConv(in_channels, f)
        self.d1 = _Down(f, f * 2)
        self.d2 = _Down(f * 2, f * 4)
        self.d3 = _Down(f * 4, f * 8)
        self.d4 = _Down(f * 8, f * 16)

        self.np_u1 = _Up(f * 16, f * 8)
        self.np_u2 = _Up(f * 8, f * 4)
        self.np_u3 = _Up(f * 4, f * 2)
        self.np_u4 = _Up(f * 2, f)
        self.np_head = nn.Conv2d(f, 2, 1)

        self.hv_u1 = _Up(f * 16, f * 8)
        self.hv_u2 = _Up(f * 8, f * 4)
        self.hv_u3 = _Up(f * 4, f * 2)
        self.hv_u4 = _Up(f * 2, f)
        self.hv_head = nn.Conv2d(f, 2, 1)

        self.num_types = num_types
        if num_types > 0:
            self.tp_u1 = _Up(f * 16, f * 8)
            self.tp_u2 = _Up(f * 8, f * 4)
            self.tp_u3 = _Up(f * 4, f * 2)
            self.tp_u4 = _Up(f * 2, f)
            self.tp_head = nn.Conv2d(f, num_types, 1)

    def forward(self, x: torch.Tensor) -> dict:
        x0 = self.inc(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        np_x = self.np_u4(self.np_u3(self.np_u2(self.np_u1(x4, x3), x2), x1), x0)
        hv_x = self.hv_u4(self.hv_u3(self.hv_u2(self.hv_u1(x4, x3), x2), x1), x0)

        out = {"np": self.np_head(np_x), "hv": torch.tanh(self.hv_head(hv_x))}

        if self.num_types > 0:
            tp_x = self.tp_u4(self.tp_u3(self.tp_u2(self.tp_u1(x4, x3), x2), x1), x0)
            out["tp"] = self.tp_head(tp_x)

        return out


def _hover_to_instances(
    np_prob: np.ndarray,
    hv_map: np.ndarray,
    fg_threshold: float = 0.5,
    min_area: int = 10,
) -> np.ndarray:
    """
    Convert nuclear pixel prob + HoV maps → integer instance map via watershed.

    Falls back to simple connected components if skimage unavailable.
    Note: Fallback produces different results (no watershed refinement).
    """
    fg = np_prob > fg_threshold

    if not _HAS_WATERSHED:
        # Fallback: simple connected components (no watershed refinement)
        from scipy import ndimage

        instance_map, _ = ndimage.label(fg)
        # Remove small objects
        for nucleus_label in np.unique(instance_map):
            if nucleus_label != 0 and (instance_map == nucleus_label).sum() < min_area:
                instance_map[instance_map == nucleus_label] = 0
        return instance_map

    # Full watershed-based instance segmentation
    h_grad = np.abs(np.gradient(hv_map[0])[1])
    v_grad = np.abs(np.gradient(hv_map[1])[0])
    energy = -(h_grad + v_grad)

    coords = peak_local_max(energy, min_distance=5, labels=fg)
    seeds = np.zeros_like(fg, dtype=int)
    if coords.shape[0] > 0:
        seeds[coords[:, 0], coords[:, 1]] = np.arange(1, coords.shape[0] + 1)

    instance_map = watershed(-energy, seeds, mask=fg)

    # Remove small objects
    for nucleus_label in np.unique(instance_map):
        if nucleus_label != 0 and (instance_map == nucleus_label).sum() < min_area:
            instance_map[instance_map == nucleus_label] = 0

    return instance_map


class NucleusDetector:
    """
    End-to-end nucleus detector: RGB patch → DetectionResult.

        detector = NucleusDetector(num_types=5)
        result = detector.detect(patch_rgb_uint8)
    """

    def __init__(
        self,
        num_types: int = 0,
        base_filters: int = 64,
        fg_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.fg_threshold = fg_threshold
        self.confidence_threshold = confidence_threshold
        self.num_types = num_types

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = HoverNet(in_channels=3, num_types=num_types, base_filters=base_filters)
        self.model.to(self.device)

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info("Loaded NucleusDetector checkpoint: %s", checkpoint_path)

        self.model.eval()

    @torch.no_grad()
    def detect(self, patch: np.ndarray) -> DetectionResult:
        """
        Args:
            patch: (H, W, 3) uint8 RGB

        Returns:
            DetectionResult
        """
        x = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = ((x - mean) / std).unsqueeze(0).to(self.device)

        out = self.model(x)
        np_prob = F.softmax(out["np"], dim=1)[0, 1].cpu().numpy()
        hv_map = out["hv"][0].cpu().numpy()

        instance_map = _hover_to_instances(np_prob, hv_map, self.fg_threshold)

        labels = [label_id for label_id in np.unique(instance_map) if label_id != 0]
        if not labels:
            h, w = patch.shape[:2]
            return DetectionResult(
                centroids=np.zeros((0, 2), dtype=np.float32),
                masks=np.zeros((0, h, w), dtype=bool),
                probabilities=np.zeros(0, dtype=np.float32),
            )

        tp_probs = None
        if self.num_types > 0 and "tp" in out:
            tp_probs = F.softmax(out["tp"], dim=1)[0].cpu().numpy()

        centroids, masks, probs, ctypes = [], [], [], []
        for nucleus_label in labels:
            m = instance_map == nucleus_label
            prob = float(np_prob[m].mean())
            if prob < self.confidence_threshold:
                continue
            ys, xs = np.where(m)
            centroids.append([ys.mean(), xs.mean()])
            masks.append(m)
            probs.append(prob)
            if tp_probs is not None:
                ctypes.append(int(tp_probs[:, m].mean(axis=1).argmax()))

        if not centroids:
            h, w = patch.shape[:2]
            return DetectionResult(
                centroids=np.zeros((0, 2), dtype=np.float32),
                masks=np.zeros((0, h, w), dtype=bool),
                probabilities=np.zeros(0, dtype=np.float32),
            )

        return DetectionResult(
            centroids=np.array(centroids, dtype=np.float32),
            masks=np.array(masks, dtype=bool),
            probabilities=np.array(probs, dtype=np.float32),
            cell_types=np.array(ctypes, dtype=np.int32) if ctypes else None,
            hover_maps=hv_map,
        )
