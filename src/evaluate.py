"""
Evaluation: Hungarian matching, per-class metrics, LOOCV runner.

Uses scipy linear_sum_assignment for optimal bipartite matching between
predictions and ground truth with class-specific match radii.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple


def compute_f1(tp: int, fp: int, fn: int, eps: float = 1e-6) -> Tuple[float, float, float]:
    """Compute F1, precision, recall from TP/FP/FN counts."""
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1, precision, recall


def match_detections_to_gt(
    detections: List[dict],
    gt_coords_6nm: np.ndarray,
    gt_coords_12nm: np.ndarray,
    match_radii: Optional[Dict[str, float]] = None,
) -> Dict[str, dict]:
    """
    Hungarian matching between predictions and ground truth.

    A detection matches GT only if:
    1. Euclidean distance < match_radius[class]
    2. Predicted class == GT class

    Args:
        detections: list of {'x', 'y', 'class', 'conf'}
        gt_coords_6nm: (N, 2) array of (x, y) GT for 6nm
        gt_coords_12nm: (M, 2) array of (x, y) GT for 12nm
        match_radii: per-class match radius in pixels

    Returns:
        Dict with per-class and overall TP/FP/FN/F1/precision/recall.
    """
    if match_radii is None:
        match_radii = {"6nm": 9.0, "12nm": 15.0}

    gt_by_class = {"6nm": gt_coords_6nm, "12nm": gt_coords_12nm}
    results = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in ["6nm", "12nm"]:
        cls_dets = [d for d in detections if d["class"] == cls]
        gt = gt_by_class[cls]
        radius = match_radii[cls]

        if len(cls_dets) == 0 and len(gt) == 0:
            results[cls] = {
                "tp": 0, "fp": 0, "fn": 0,
                "f1": 1.0, "precision": 1.0, "recall": 1.0,
            }
            continue

        if len(cls_dets) == 0:
            results[cls] = {
                "tp": 0, "fp": 0, "fn": len(gt),
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
            }
            total_fn += len(gt)
            continue

        if len(gt) == 0:
            results[cls] = {
                "tp": 0, "fp": len(cls_dets), "fn": 0,
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
            }
            total_fp += len(cls_dets)
            continue

        # Build cost matrix
        pred_coords = np.array([[d["x"], d["y"]] for d in cls_dets])
        cost = cdist(pred_coords, gt)

        # Set costs beyond radius to a large value (forbid match)
        cost_masked = cost.copy()
        cost_masked[cost_masked > radius] = 1e6

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_masked)

        # Count valid matches (within radius)
        tp = sum(
            1 for r, c in zip(row_ind, col_ind)
            if cost_masked[r, c] <= radius
        )
        fp = len(cls_dets) - tp
        fn = len(gt) - tp

        f1, prec, rec = compute_f1(tp, fp, fn)

        results[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "f1": f1, "precision": prec, "recall": rec,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Overall
    f1_overall, prec_overall, rec_overall = compute_f1(total_tp, total_fp, total_fn)
    results["overall"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "f1": f1_overall, "precision": prec_overall, "recall": rec_overall,
    }

    # Mean F1 across classes
    class_f1s = [results[c]["f1"] for c in ["6nm", "12nm"] if results[c]["fn"] + results[c]["tp"] > 0]
    results["mean_f1"] = np.mean(class_f1s) if class_f1s else 0.0

    return results


def evaluate_fold(
    detections: List[dict],
    gt_annotations: Dict[str, np.ndarray],
    match_radii: Optional[Dict[str, float]] = None,
    has_6nm: bool = True,
) -> Dict[str, dict]:
    """
    Evaluate detections for a single LOOCV fold.

    Args:
        detections: model predictions
        gt_annotations: {'6nm': Nx2, '12nm': Mx2}
        match_radii: per-class match radii
        has_6nm: whether this fold has 6nm GT (False for S7, S15)

    Returns:
        Evaluation metrics dict.
    """
    gt_6nm = gt_annotations.get("6nm", np.empty((0, 2)))
    gt_12nm = gt_annotations.get("12nm", np.empty((0, 2)))

    results = match_detections_to_gt(detections, gt_6nm, gt_12nm, match_radii)

    if not has_6nm:
        results["6nm"]["note"] = "N/A (missing annotations)"

    return results


def compute_average_precision(
    detections: List[dict],
    gt_coords: np.ndarray,
    match_radius: float,
) -> float:
    """
    Compute Average Precision (AP) for a single class.

    Follows PASCAL VOC style: sort by confidence, compute precision-recall
    curve, then compute area under curve.
    """
    if len(gt_coords) == 0:
        return 0.0 if detections else 1.0

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)

    tp_list = []
    fp_list = []
    matched_gt = set()

    for det in sorted_dets:
        det_coord = np.array([det["x"], det["y"]])
        dists = np.sqrt(np.sum((gt_coords - det_coord) ** 2, axis=1))
        min_idx = np.argmin(dists)

        if dists[min_idx] <= match_radius and min_idx not in matched_gt:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt.add(min_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_coords)

    # Compute AP using all-point interpolation
    ap = 0.0
    for i in range(len(precision)):
        if i == 0:
            ap += precision[i] * recall[i]
        else:
            ap += precision[i] * (recall[i] - recall[i - 1])

    return ap
