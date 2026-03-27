"""Unit tests for loss functions."""

import pytest
import torch

from src.loss import cornernet_focal_loss, offset_loss, total_loss


class TestCornerNetFocalLoss:
    def test_perfect_prediction_zero_loss(self):
        """Perfect predictions should produce near-zero loss."""
        gt = torch.zeros(1, 2, 64, 64)
        gt[0, 0, 32, 32] = 1.0  # one particle

        # Near-perfect prediction
        pred = torch.zeros(1, 2, 64, 64) + 1e-6
        pred[0, 0, 32, 32] = 1.0 - 1e-6

        loss = cornernet_focal_loss(pred, gt)
        assert loss.item() < 0.1

    def test_all_zeros_prediction_nonzero_loss(self):
        """Predicting all zeros when particles exist should give positive loss."""
        gt = torch.zeros(1, 2, 64, 64)
        gt[0, 0, 32, 32] = 1.0

        pred = torch.zeros(1, 2, 64, 64) + 1e-6
        loss = cornernet_focal_loss(pred, gt)
        assert loss.item() > 0

    def test_high_false_positive_penalized(self):
        """Predicting high confidence where GT is zero should be penalized."""
        gt = torch.zeros(1, 2, 64, 64)
        pred_low_fp = torch.zeros(1, 2, 64, 64) + 0.01
        pred_high_fp = torch.zeros(1, 2, 64, 64) + 0.9

        loss_low = cornernet_focal_loss(pred_low_fp, gt)
        loss_high = cornernet_focal_loss(pred_high_fp, gt)

        assert loss_high.item() > loss_low.item()

    def test_near_peak_reduced_penalty(self):
        """Pixels near GT peaks should have reduced negative penalty via beta term."""
        gt = torch.zeros(1, 2, 64, 64)
        gt[0, 0, 32, 32] = 1.0
        gt[0, 0, 31, 32] = 0.8  # nearby pixel with Gaussian falloff

        # Moderate prediction near peak should have low loss
        pred = torch.zeros(1, 2, 64, 64) + 0.01
        pred[0, 0, 31, 32] = 0.5

        loss = cornernet_focal_loss(pred, gt)
        # Should be a reasonable value, not extremely high
        assert loss.item() < 10

    def test_confidence_weighting(self):
        """Confidence weights should scale the loss."""
        gt = torch.zeros(1, 2, 64, 64)
        gt[0, 0, 32, 32] = 1.0
        pred = torch.zeros(1, 2, 64, 64) + 0.5

        weights_full = torch.ones(1, 2, 64, 64)
        weights_half = torch.ones(1, 2, 64, 64) * 0.5

        loss_full = cornernet_focal_loss(pred, gt, conf_weights=weights_full)
        loss_half = cornernet_focal_loss(pred, gt, conf_weights=weights_half)

        # Half weights should produce lower loss
        assert loss_half.item() < loss_full.item()


class TestOffsetLoss:
    def test_zero_when_no_particles(self):
        """Offset loss should be zero when mask is empty."""
        pred = torch.randn(1, 2, 64, 64)
        gt = torch.zeros(1, 2, 64, 64)
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)

        loss = offset_loss(pred, gt, mask)
        assert loss.item() == 0.0

    def test_nonzero_with_particles(self):
        """Offset loss should be nonzero when predictions differ from GT."""
        pred = torch.randn(1, 2, 64, 64)
        gt = torch.zeros(1, 2, 64, 64)
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        mask[0, 32, 32] = True

        loss = offset_loss(pred, gt, mask)
        assert loss.item() > 0


class TestTotalLoss:
    def test_returns_three_values(self):
        """total_loss should return (total, hm_loss, off_loss)."""
        hm_pred = torch.sigmoid(torch.randn(1, 2, 64, 64))
        hm_gt = torch.zeros(1, 2, 64, 64)
        off_pred = torch.randn(1, 2, 64, 64)
        off_gt = torch.zeros(1, 2, 64, 64)
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)

        total, hm_val, off_val = total_loss(
            hm_pred, hm_gt, off_pred, off_gt, mask,
        )

        assert isinstance(total, torch.Tensor)
        assert isinstance(hm_val, float)
        assert isinstance(off_val, float)
        assert total.requires_grad
