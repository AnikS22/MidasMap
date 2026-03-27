"""
Loss functions for CenterNet immunogold detection.

Implements CornerNet penalty-reduced focal loss for sparse heatmaps
and smooth L1 offset regression loss.
"""

import torch
import torch.nn.functional as F


def cornernet_focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    alpha: int = 2,
    beta: int = 4,
    conf_weights: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    CornerNet penalty-reduced focal loss for sparse heatmaps.

    The positive:negative pixel ratio is ~1:23,000 per channel.
    Standard BCE would learn to predict all zeros. This loss
    penalizes confident wrong predictions and rewards uncertain
    correct ones via the (1-p)^alpha and p^alpha terms.

    Args:
        pred: (B, C, H, W) sigmoid-activated predictions in [0, 1]
        gt: (B, C, H, W) Gaussian heatmap targets in [0, 1]
        alpha: focal exponent for prediction confidence (default 2)
        beta: penalty reduction exponent near GT peaks (default 4)
        conf_weights: optional (B, C, H, W) per-pixel confidence weights
                      for pseudo-label weighting
        eps: numerical stability

    Returns:
        Scalar loss, normalized by number of positive locations.
    """
    pos_mask = (gt == 1).float()
    neg_mask = (gt < 1).float()

    # Penalty reduction: pixels near particle centers get lower negative penalty
    # (1 - gt)^beta → 0 near peaks, → 1 far from peaks
    neg_weights = torch.pow(1 - gt, beta)

    # Positive loss: encourage high confidence at GT peaks
    pos_loss = torch.log(pred.clamp(min=eps)) * torch.pow(1 - pred, alpha) * pos_mask

    # Negative loss: penalize high confidence away from GT peaks
    neg_loss = (
        torch.log((1 - pred).clamp(min=eps))
        * torch.pow(pred, alpha)
        * neg_weights
        * neg_mask
    )

    # Apply confidence weighting if provided (for pseudo-label support)
    if conf_weights is not None:
        pos_loss = pos_loss * conf_weights
        # Negative loss near pseudo-labels also scaled
        neg_loss = neg_loss * conf_weights

    num_pos = pos_mask.sum().clamp(min=1)
    loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos

    return loss


def offset_loss(
    pred_offsets: torch.Tensor,
    gt_offsets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Smooth L1 loss on sub-pixel offsets at annotated particle locations only.

    Args:
        pred_offsets: (B, 2, H, W) predicted offsets
        gt_offsets: (B, 2, H, W) ground truth offsets
        mask: (B, H, W) boolean — True at particle integer centers

    Returns:
        Scalar loss.
    """
    # Expand mask to match offset dimensions
    mask_expanded = mask.unsqueeze(1).expand_as(pred_offsets)

    if mask_expanded.sum() == 0:
        return torch.tensor(0.0, device=pred_offsets.device, requires_grad=True)

    loss = F.smooth_l1_loss(
        pred_offsets[mask_expanded],
        gt_offsets[mask_expanded],
        reduction="mean",
    )
    return loss


def total_loss(
    heatmap_pred: torch.Tensor,
    heatmap_gt: torch.Tensor,
    offset_pred: torch.Tensor,
    offset_gt: torch.Tensor,
    offset_mask: torch.Tensor,
    lambda_offset: float = 1.0,
    focal_alpha: int = 2,
    focal_beta: int = 4,
    conf_weights: torch.Tensor = None,
) -> tuple:
    """
    Combined heatmap focal loss + offset regression loss.

    Args:
        heatmap_pred: (B, 2, H, W) sigmoid predictions
        heatmap_gt: (B, 2, H, W) Gaussian GT
        offset_pred: (B, 2, H, W) predicted offsets
        offset_gt: (B, 2, H, W) GT offsets
        offset_mask: (B, H, W) boolean mask
        lambda_offset: weight for offset loss (default 1.0)
        focal_alpha: focal loss alpha
        focal_beta: focal loss beta
        conf_weights: optional per-pixel confidence weights

    Returns:
        (total_loss, heatmap_loss_value, offset_loss_value)
    """
    l_hm = cornernet_focal_loss(
        heatmap_pred, heatmap_gt,
        alpha=focal_alpha, beta=focal_beta,
        conf_weights=conf_weights,
    )
    l_off = offset_loss(offset_pred, offset_gt, offset_mask)

    total = l_hm + lambda_offset * l_off

    return total, l_hm.item(), l_off.item()
