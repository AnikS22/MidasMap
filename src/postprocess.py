"""
Post-processing: structural mask filtering, cross-class NMS, threshold sweep.
"""

import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import dilation, disk
from typing import Dict, List, Optional


def apply_structural_mask_filter(
    detections: List[dict],
    mask: np.ndarray,
    margin_px: int = 5,
) -> List[dict]:
    """
    Remove detections outside biological tissue regions.

    Args:
        detections: list of {'x', 'y', 'class', 'conf'}
        mask: boolean array (H, W) where True = tissue region
        margin_px: dilate mask by this many pixels

    Returns:
        Filtered detection list.
    """
    if mask is None:
        return detections

    # Dilate mask to allow particles at region boundaries
    tissue = dilation(mask, disk(margin_px))

    filtered = []
    for det in detections:
        xi, yi = int(round(det["x"])), int(round(det["y"]))
        if (0 <= yi < tissue.shape[0] and
            0 <= xi < tissue.shape[1] and
            tissue[yi, xi]):
            filtered.append(det)
    return filtered


def cross_class_nms(
    detections: List[dict],
    distance_threshold: float = 8.0,
) -> List[dict]:
    """
    When 6nm and 12nm detections overlap, keep the higher-confidence one.

    This handles cases where both heads fire on the same particle.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = [True] * len(dets)

    coords = np.array([[d["x"], d["y"]] for d in dets])

    for i in range(len(dets)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(dets)):
            if not keep[j]:
                continue
            # Only suppress across classes
            if dets[i]["class"] == dets[j]["class"]:
                continue
            dist = np.sqrt(
                (coords[i, 0] - coords[j, 0]) ** 2
                + (coords[i, 1] - coords[j, 1]) ** 2
            )
            if dist < distance_threshold:
                keep[j] = False  # Lower confidence suppressed

    return [d for d, k in zip(dets, keep) if k]


def sweep_confidence_threshold(
    detections: List[dict],
    gt_coords: Dict[str, np.ndarray],
    match_radii: Dict[str, float],
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> Dict[str, float]:
    """
    Sweep confidence thresholds to find optimal per-class thresholds.

    Args:
        detections: all detections (before thresholding)
        gt_coords: {'6nm': Nx2, '12nm': Mx2} ground truth
        match_radii: per-class match radii in pixels
        start, stop, step: sweep range

    Returns:
        Dict with best threshold per class and overall.
    """
    from src.evaluate import match_detections_to_gt, compute_f1

    best_thresholds = {}
    thresholds = np.arange(start, stop, step)

    for cls in ["6nm", "12nm"]:
        best_f1 = -1
        best_thr = 0.3

        for thr in thresholds:
            cls_dets = [d for d in detections if d["class"] == cls and d["conf"] >= thr]
            if not cls_dets and len(gt_coords[cls]) == 0:
                continue

            pred_coords = np.array([[d["x"], d["y"]] for d in cls_dets]).reshape(-1, 2)
            gt = gt_coords[cls]

            if len(pred_coords) == 0:
                tp, fp, fn = 0, 0, len(gt)
            elif len(gt) == 0:
                tp, fp, fn = 0, len(pred_coords), 0
            else:
                tp, fp, fn = _simple_match(pred_coords, gt, match_radii[cls])

            f1, _, _ = compute_f1(tp, fp, fn)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        best_thresholds[cls] = best_thr

    return best_thresholds


def _simple_match(
    pred: np.ndarray, gt: np.ndarray, radius: float
) -> tuple:
    """Quick matching for threshold sweep (greedy, not Hungarian)."""
    from scipy.spatial.distance import cdist

    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt)

    dists = cdist(pred, gt)
    tp = 0
    matched_gt = set()

    # Greedy: match closest pairs first
    for i in range(len(pred)):
        min_j = np.argmin(dists[i])
        if dists[i, min_j] <= radius and min_j not in matched_gt:
            tp += 1
            matched_gt.add(min_j)
            dists[:, min_j] = np.inf

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn
