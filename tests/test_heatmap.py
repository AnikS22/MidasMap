"""Unit tests for heatmap GT generation and peak extraction."""

import numpy as np
import pytest
import torch

from src.heatmap import generate_heatmap_gt, extract_peaks


class TestHeatmapGeneration:
    def test_single_particle_peak(self):
        """A single particle should produce a Gaussian peak at the correct location."""
        coords_6nm = np.array([[100.0, 200.0]])
        coords_12nm = np.empty((0, 2))

        hm, off, mask, conf = generate_heatmap_gt(
            coords_6nm, coords_12nm, 512, 512, stride=2,
        )

        assert hm.shape == (2, 256, 256)
        assert off.shape == (2, 256, 256)
        assert mask.shape == (256, 256)

        # Peak should be at (50, 100) in stride-2 space
        peak_y, peak_x = np.unravel_index(hm[0].argmax(), hm[0].shape)
        assert abs(peak_x - 50) <= 1
        assert abs(peak_y - 100) <= 1

        # Peak value should be 1.0 (confidence=1.0 default)
        assert hm[0].max() == pytest.approx(1.0, abs=0.01)

        # 12nm channel should be empty
        assert hm[1].max() == 0.0

    def test_two_classes(self):
        """Both classes should produce peaks in their respective channels."""
        coords_6nm = np.array([[100.0, 100.0]])
        coords_12nm = np.array([[200.0, 200.0]])

        hm, _, _, _ = generate_heatmap_gt(
            coords_6nm, coords_12nm, 512, 512, stride=2,
        )

        assert hm[0].max() > 0.9  # 6nm channel has peak
        assert hm[1].max() > 0.9  # 12nm channel has peak

    def test_offset_values(self):
        """Offsets should encode sub-pixel correction."""
        # Place particle at (101.5, 200.5) → stride-2 center at (50.75, 100.25)
        # Integer center: (51, 100) → offset: (-0.25, 0.25)
        coords_6nm = np.array([[101.5, 200.5]])
        coords_12nm = np.empty((0, 2))

        _, off, mask, _ = generate_heatmap_gt(
            coords_6nm, coords_12nm, 512, 512, stride=2,
        )

        # Mask should have exactly one True pixel
        assert mask.sum() == 1

    def test_empty_annotations(self):
        """Empty annotations should produce zero heatmap."""
        hm, off, mask, conf = generate_heatmap_gt(
            np.empty((0, 2)), np.empty((0, 2)), 512, 512,
        )
        assert hm.max() == 0.0
        assert mask.sum() == 0

    def test_confidence_weighting(self):
        """Confidence < 1 should scale peak value."""
        coords = np.array([[100.0, 100.0]])
        confidences = np.array([0.5])

        hm, _, _, _ = generate_heatmap_gt(
            coords, np.empty((0, 2)), 512, 512,
            confidence_6nm=confidences,
        )

        assert hm[0].max() == pytest.approx(0.5, abs=0.05)

    def test_overlapping_particles_use_max(self):
        """Overlapping Gaussians should use element-wise max, not sum."""
        coords = np.array([[100.0, 100.0], [104.0, 100.0]])  # close together
        hm, _, _, _ = generate_heatmap_gt(
            coords, np.empty((0, 2)), 512, 512, stride=2,
        )
        # Max should be 1.0, not >1.0
        assert hm[0].max() <= 1.0


class TestPeakExtraction:
    def test_single_peak(self):
        """Extract a single peak from synthetic heatmap."""
        heatmap = torch.zeros(2, 256, 256)
        heatmap[0, 100, 50] = 0.9  # 6nm peak

        offset_map = torch.zeros(2, 256, 256)
        offset_map[0, 100, 50] = 0.3  # dx
        offset_map[1, 100, 50] = 0.1  # dy

        dets = extract_peaks(heatmap, offset_map, stride=2, conf_threshold=0.5)

        assert len(dets) == 1
        assert dets[0]["class"] == "6nm"
        assert dets[0]["conf"] == pytest.approx(0.9, abs=0.01)
        # x = (50 + 0.3) * 2 = 100.6
        assert dets[0]["x"] == pytest.approx(100.6, abs=0.1)
        # y = (100 + 0.1) * 2 = 200.2
        assert dets[0]["y"] == pytest.approx(200.2, abs=0.1)

    def test_nms_suppresses_neighbors(self):
        """NMS should suppress weaker neighboring peaks."""
        heatmap = torch.zeros(2, 256, 256)
        heatmap[0, 100, 50] = 0.9  # strong
        heatmap[0, 101, 50] = 0.7  # weaker neighbor (within NMS kernel)

        dets = extract_peaks(
            heatmap, torch.zeros(2, 256, 256),
            stride=2, conf_threshold=0.5,
            nms_kernel_sizes={"6nm": 5, "12nm": 5},
        )

        # Only the stronger peak should survive
        assert len([d for d in dets if d["class"] == "6nm"]) == 1

    def test_below_threshold_filtered(self):
        """Peaks below threshold should not be extracted."""
        heatmap = torch.zeros(2, 256, 256)
        heatmap[0, 100, 50] = 0.2  # below 0.3 threshold

        dets = extract_peaks(heatmap, torch.zeros(2, 256, 256), conf_threshold=0.3)
        assert len(dets) == 0
