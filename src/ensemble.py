"""
Test-time augmentation (D4 dihedral group) and model ensemble averaging.

D4 TTA: 4 rotations x 2 reflections = 8 geometric views
+ 2 intensity variants = 10 total forward passes.
Gold beads are rotationally invariant — D4 TTA is maximally effective.
Expected F1 gain: +1-3% over single forward pass.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional

from src.model import ImmunogoldCenterNet


def d4_tta_predict(
    model: ImmunogoldCenterNet,
    image: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Test-time augmentation over D4 dihedral group + intensity variants.

    Args:
        model: trained CenterNet model
        image: (H, W) uint8 preprocessed image
        device: torch device

    Returns:
        averaged_heatmap: (2, H/2, W/2) numpy array
        averaged_offsets: (2, H/2, W/2) numpy array
    """
    model.eval()
    heatmaps = []
    offsets_list = []

    # Ensure image dimensions are divisible by 32 for the encoder
    h, w = image.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    def _forward(img_np):
        """Run model on numpy image, return heatmap and offsets."""
        # Pad to multiple of 32
        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w)), mode="reflect")

        tensor = (
            torch.from_numpy(img_np)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)  # (1, 1, H, W)
            / 255.0
        ).to(device)

        with torch.no_grad():
            hm, off = model(tensor)

        hm = hm.squeeze(0).cpu().numpy()   # (2, H/2, W/2)
        off = off.squeeze(0).cpu().numpy()  # (2, H/2, W/2)

        # Remove padding from output
        hm_h = h // 2
        hm_w = w // 2
        return hm[:, :hm_h, :hm_w], off[:, :hm_h, :hm_w]

    # D4 group: 4 rotations x 2 reflections = 8 geometric views
    for k in range(4):
        for flip in [False, True]:
            aug = np.rot90(image, k).copy()
            if flip:
                aug = np.fliplr(aug).copy()

            hm, off = _forward(aug)

            # Inverse transforms on heatmap and offsets
            if flip:
                hm = np.flip(hm, axis=2).copy()   # flip W axis
                off = np.flip(off, axis=2).copy()
                off[0] = -off[0]  # negate x offset for horizontal flip

            if k > 0:
                hm = np.rot90(hm, -k, axes=(1, 2)).copy()
                off = np.rot90(off, -k, axes=(1, 2)).copy()
                # Rotate offset vectors
                if k == 1:  # 90° CCW undo
                    off = np.stack([-off[1], off[0]], axis=0)
                elif k == 2:  # 180°
                    off = np.stack([-off[0], -off[1]], axis=0)
                elif k == 3:  # 270° CCW undo
                    off = np.stack([off[1], -off[0]], axis=0)

            heatmaps.append(hm)
            offsets_list.append(off)

    # 2 intensity variants
    for factor in [0.9, 1.1]:
        aug = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        hm, off = _forward(aug)
        heatmaps.append(hm)
        offsets_list.append(off)

    # Average all views
    avg_heatmap = np.mean(heatmaps, axis=0)
    avg_offsets = np.mean(offsets_list, axis=0)

    return avg_heatmap, avg_offsets


def ensemble_predict(
    models: List[ImmunogoldCenterNet],
    image: np.ndarray,
    device: torch.device = torch.device("cpu"),
    use_tta: bool = True,
) -> tuple:
    """
    Ensemble prediction: average heatmaps from N models.

    Args:
        models: list of trained models (e.g., 5 seeds x 3 snapshots = 15)
        image: (H, W) uint8 preprocessed image
        device: torch device
        use_tta: whether to apply D4 TTA per model

    Returns:
        averaged_heatmap: (2, H/2, W/2) numpy array
        averaged_offsets: (2, H/2, W/2) numpy array
    """
    all_heatmaps = []
    all_offsets = []

    for model in models:
        model.eval()
        model.to(device)

        if use_tta:
            hm, off = d4_tta_predict(model, image, device)
        else:
            h, w = image.shape[:2]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32
            img_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")

            tensor = (
                torch.from_numpy(img_padded)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                / 255.0
            ).to(device)

            with torch.no_grad():
                hm_t, off_t = model(tensor)

            hm = hm_t.squeeze(0).cpu().numpy()[:, : h // 2, : w // 2]
            off = off_t.squeeze(0).cpu().numpy()[:, : h // 2, : w // 2]

        all_heatmaps.append(hm)
        all_offsets.append(off)

    return np.mean(all_heatmaps, axis=0), np.mean(all_offsets, axis=0)


def sliding_window_inference(
    model: ImmunogoldCenterNet,
    image: np.ndarray,
    patch_size: int = 512,
    overlap: int = 128,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Full-image inference via sliding window with overlap stitching.

    Tiles the image into overlapping patches, runs the model on each,
    and stitches heatmaps using max in overlap regions.

    Args:
        model: trained model
        image: (H, W) uint8 preprocessed image
        patch_size: tile size
        overlap: overlap between tiles
        device: torch device

    Returns:
        heatmap: (2, H/2, W/2) numpy array
        offsets: (2, H/2, W/2) numpy array
    """
    model.eval()
    h, w = image.shape[:2]
    stride_step = patch_size - overlap

    # Output dimensions at model stride
    out_h = h // 2
    out_w = w // 2
    out_patch = patch_size // 2

    heatmap = np.zeros((2, out_h, out_w), dtype=np.float32)
    offsets = np.zeros((2, out_h, out_w), dtype=np.float32)
    count = np.zeros((out_h, out_w), dtype=np.float32)

    for y0 in range(0, h - patch_size + 1, stride_step):
        for x0 in range(0, w - patch_size + 1, stride_step):
            patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
            tensor = (
                torch.from_numpy(patch)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                / 255.0
            ).to(device)

            with torch.no_grad():
                hm, off = model(tensor)

            hm_np = hm.squeeze(0).cpu().numpy()
            off_np = off.squeeze(0).cpu().numpy()

            # Output coordinates
            oy0 = y0 // 2
            ox0 = x0 // 2

            # Max-stitch heatmap, average-stitch offsets
            heatmap[:, oy0 : oy0 + out_patch, ox0 : ox0 + out_patch] = np.maximum(
                heatmap[:, oy0 : oy0 + out_patch, ox0 : ox0 + out_patch],
                hm_np,
            )
            offsets[:, oy0 : oy0 + out_patch, ox0 : ox0 + out_patch] += off_np
            count[oy0 : oy0 + out_patch, ox0 : ox0 + out_patch] += 1

    # Average offsets where counted
    count = np.maximum(count, 1)
    offsets /= count[np.newaxis, :, :]

    return heatmap, offsets
