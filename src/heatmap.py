"""
Ground truth heatmap generation and peak extraction for CenterNet.

Generates Gaussian-splat heatmaps at stride-2 resolution with
class-specific sigma values calibrated to bead size.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# Class index mapping
CLASS_IDX = {"6nm": 0, "12nm": 1}
CLASS_NAMES = ["6nm", "12nm"]
STRIDE = 2


def generate_heatmap_gt(
    coords_6nm: np.ndarray,
    coords_12nm: np.ndarray,
    image_h: int,
    image_w: int,
    sigmas: Optional[Dict[str, float]] = None,
    stride: int = STRIDE,
    confidence_6nm: Optional[np.ndarray] = None,
    confidence_12nm: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate CenterNet ground truth heatmaps and offset maps.

    Args:
        coords_6nm: (N, 2) array of (x, y) in ORIGINAL pixel space
        coords_12nm: (M, 2) array of (x, y) in ORIGINAL pixel space
        image_h: original image height
        image_w: original image width
        sigmas: per-class Gaussian sigma in feature space
        stride: output stride (default 2)
        confidence_6nm: optional per-particle confidence weights
        confidence_12nm: optional per-particle confidence weights

    Returns:
        heatmap: (2, H//stride, W//stride) float32 in [0, 1]
        offsets: (2, H//stride, W//stride) float32 sub-pixel offsets
        offset_mask: (H//stride, W//stride) bool — True at particle centers
        conf_map: (2, H//stride, W//stride) float32 confidence weights
    """
    if sigmas is None:
        sigmas = {"6nm": 1.0, "12nm": 1.5}

    h_feat = image_h // stride
    w_feat = image_w // stride

    heatmap = np.zeros((2, h_feat, w_feat), dtype=np.float32)
    offsets = np.zeros((2, h_feat, w_feat), dtype=np.float32)
    offset_mask = np.zeros((h_feat, w_feat), dtype=bool)
    conf_map = np.ones((2, h_feat, w_feat), dtype=np.float32)

    # Prepare coordinate lists with class labels and confidences
    all_entries = []
    if len(coords_6nm) > 0:
        confs = confidence_6nm if confidence_6nm is not None else np.ones(len(coords_6nm))
        for i, (x, y) in enumerate(coords_6nm):
            all_entries.append((x, y, "6nm", confs[i]))
    if len(coords_12nm) > 0:
        confs = confidence_12nm if confidence_12nm is not None else np.ones(len(coords_12nm))
        for i, (x, y) in enumerate(coords_12nm):
            all_entries.append((x, y, "12nm", confs[i]))

    for x, y, cls, conf in all_entries:
        cidx = CLASS_IDX[cls]
        sigma = sigmas[cls]

        # Feature-space center (float)
        cx_f = x / stride
        cy_f = y / stride

        # Integer grid center
        cx_i = int(round(cx_f))
        cy_i = int(round(cy_f))

        # Sub-pixel offset
        off_x = cx_f - cx_i
        off_y = cy_f - cy_i

        # Gaussian radius: truncate at 3 sigma
        r = max(int(3 * sigma + 1), 2)

        # Bounds-clipped grid
        y0 = max(0, cy_i - r)
        y1 = min(h_feat, cy_i + r + 1)
        x0 = max(0, cx_i - r)
        x1 = min(w_feat, cx_i + r + 1)

        if y0 >= y1 or x0 >= x1:
            continue

        yy, xx = np.meshgrid(
            np.arange(y0, y1),
            np.arange(x0, x1),
            indexing="ij",
        )

        # Gaussian centered at INTEGER center (not float)
        # The integer center MUST be exactly 1.0 — the CornerNet focal loss
        # uses pos_mask = (gt == 1.0) and treats everything else as negative.
        # Centering the Gaussian at the float position produces peaks of 0.78-0.93
        # which the loss sees as negatives → zero positive training signal.
        gaussian = np.exp(
            -((xx - cx_i) ** 2 + (yy - cy_i) ** 2) / (2 * sigma ** 2)
        )

        # Scale by confidence (for pseudo-label weighting)
        gaussian = gaussian * conf

        # Element-wise max (handles overlapping particles correctly)
        heatmap[cidx, y0:y1, x0:x1] = np.maximum(
            heatmap[cidx, y0:y1, x0:x1], gaussian
        )

        # Offset and confidence only at the integer center pixel
        if 0 <= cy_i < h_feat and 0 <= cx_i < w_feat:
            offsets[0, cy_i, cx_i] = off_x
            offsets[1, cy_i, cx_i] = off_y
            offset_mask[cy_i, cx_i] = True
            conf_map[cidx, cy_i, cx_i] = conf

    return heatmap, offsets, offset_mask, conf_map


def extract_peaks(
    heatmap: torch.Tensor,
    offset_map: torch.Tensor,
    stride: int = STRIDE,
    conf_threshold: float = 0.3,
    nms_kernel_sizes: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """
    Extract detections from predicted heatmap via max-pool NMS.

    Args:
        heatmap: (2, H/stride, W/stride) sigmoid-activated
        offset_map: (2, H/stride, W/stride) raw offset predictions
        stride: output stride
        conf_threshold: minimum confidence to keep
        nms_kernel_sizes: per-class NMS kernel sizes

    Returns:
        List of {'x': float, 'y': float, 'class': str, 'conf': float}
    """
    if nms_kernel_sizes is None:
        nms_kernel_sizes = {"6nm": 3, "12nm": 5}

    detections = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        hm_cls = heatmap[cls_idx].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        kernel = nms_kernel_sizes[cls_name]

        # Max-pool NMS
        hmax = F.max_pool2d(
            hm_cls, kernel_size=kernel, stride=1, padding=kernel // 2
        )
        peaks = (hmax.squeeze() == heatmap[cls_idx]) & (
            heatmap[cls_idx] > conf_threshold
        )

        ys, xs = torch.where(peaks)
        for y_idx, x_idx in zip(ys, xs):
            y_i = y_idx.item()
            x_i = x_idx.item()
            conf = heatmap[cls_idx, y_i, x_i].item()
            dx = offset_map[0, y_i, x_i].item()
            dy = offset_map[1, y_i, x_i].item()

            # Back to input space with sub-pixel offset
            det_x = (x_i + dx) * stride
            det_y = (y_i + dy) * stride

            detections.append({
                "x": det_x,
                "y": det_y,
                "class": cls_name,
                "conf": conf,
            })

    return detections
