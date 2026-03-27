"""
Visualization utilities for QC at every pipeline stage.

Generates overlay images showing predictions on raw EM images:
- Cyan circles for 6nm particles
- Yellow circles for 12nm particles
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional


# Color scheme
COLORS = {
    "6nm": (0, 255, 255),     # cyan
    "12nm": (255, 255, 0),    # yellow
    "6nm_pred": (0, 200, 200),
    "12nm_pred": (200, 200, 0),
}

RADII = {"6nm": 6, "12nm": 12}


def overlay_annotations(
    image: np.ndarray,
    annotations: Dict[str, np.ndarray],
    title: str = "",
    save_path: Optional[Path] = None,
    predictions: Optional[List[dict]] = None,
    figsize: tuple = (12, 12),
) -> plt.Figure:
    """
    Overlay ground truth annotations (and optional predictions) on image.

    Args:
        image: (H, W) grayscale image
        annotations: {'6nm': Nx2, '12nm': Mx2} pixel coordinates
        title: figure title
        save_path: if provided, save figure here
        predictions: optional list of {'x', 'y', 'class', 'conf'}
        figsize: figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image, cmap="gray")

    # Ground truth circles (solid)
    for cls, coords in annotations.items():
        if len(coords) == 0:
            continue
        color_rgb = np.array(COLORS[cls]) / 255.0
        radius = RADII[cls]
        for x, y in coords:
            circle = plt.Circle(
                (x, y), radius, fill=False,
                edgecolor=color_rgb, linewidth=1.5,
            )
            ax.add_patch(circle)

    # Predictions (dashed)
    if predictions:
        for det in predictions:
            cls = det["class"]
            color_rgb = np.array(COLORS.get(f"{cls}_pred", COLORS[cls])) / 255.0
            radius = RADII[cls]
            circle = plt.Circle(
                (det["x"], det["y"]), radius, fill=False,
                edgecolor=color_rgb, linewidth=1.0, linestyle="--",
            )
            ax.add_patch(circle)
            # Confidence label
            ax.text(
                det["x"] + radius + 2, det["y"],
                f'{det["conf"]:.2f}',
                color=color_rgb, fontsize=6,
            )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="cyan", label=f'6nm GT ({len(annotations.get("6nm", []))})', linewidth=1.5),
        mpatches.Patch(facecolor="none", edgecolor="yellow", label=f'12nm GT ({len(annotations.get("12nm", []))})', linewidth=1.5),
    ]
    if predictions:
        n_pred_6 = sum(1 for d in predictions if d["class"] == "6nm")
        n_pred_12 = sum(1 for d in predictions if d["class"] == "12nm")
        legend_elements.extend([
            mpatches.Patch(facecolor="none", edgecolor="darkcyan", label=f"6nm pred ({n_pred_6})", linewidth=1.0),
            mpatches.Patch(facecolor="none", edgecolor="goldenrod", label=f"12nm pred ({n_pred_12})", linewidth=1.0),
        ])
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.axis("off")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Overlay predicted heatmap on image for QC.

    Args:
        image: (H, W) grayscale
        heatmap: (2, H/2, W/2) predicted heatmap
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Raw Image")
    axes[0].axis("off")

    # Upsample heatmap to image size for overlay
    h, w = image.shape[:2]

    for idx, (cls, color) in enumerate([("6nm", "hot"), ("12nm", "cool")]):
        hm = heatmap[idx]
        # Resize to image dims
        from skimage.transform import resize
        hm_up = resize(hm, (h, w), order=1)

        axes[idx + 1].imshow(image, cmap="gray")
        axes[idx + 1].imshow(hm_up, cmap=color, alpha=0.5, vmin=0, vmax=1)
        axes[idx + 1].set_title(f"{cls} heatmap")
        axes[idx + 1].axis("off")

    fig.suptitle(title, fontsize=12)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_training_curves(
    metrics: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot training loss and F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(metrics["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, metrics["train_loss"], label="Train Loss")
    if "val_loss" in metrics:
        ax1.plot(epochs, metrics["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1
    if "val_f1_6nm" in metrics:
        ax2.plot(epochs, metrics["val_f1_6nm"], label="6nm F1")
    if "val_f1_12nm" in metrics:
        ax2.plot(epochs, metrics["val_f1_12nm"], label="12nm F1")
    if "val_f1_mean" in metrics:
        ax2.plot(epochs, metrics["val_f1_mean"], label="Mean F1", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_precision_recall_curve(
    detections: List[dict],
    gt_coords: np.ndarray,
    match_radius: float,
    cls_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot precision-recall curve for one class."""
    sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)

    tp_list = []
    matched_gt = set()

    for det in sorted_dets:
        det_coord = np.array([det["x"], det["y"]])
        if len(gt_coords) > 0:
            dists = np.sqrt(np.sum((gt_coords - det_coord) ** 2, axis=1))
            min_idx = np.argmin(dists)
            if dists[min_idx] <= match_radius and min_idx not in matched_gt:
                tp_list.append(1)
                matched_gt.add(min_idx)
            else:
                tp_list.append(0)
        else:
            tp_list.append(0)

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - t for t in tp_list])
    n_gt = max(len(gt_coords), 1)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_gt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {cls_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
