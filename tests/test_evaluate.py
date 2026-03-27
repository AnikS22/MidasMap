"""Unit tests for evaluation matching and metrics."""

import numpy as np
import pytest

from src.evaluate import match_detections_to_gt, compute_f1, compute_average_precision


class TestComputeF1:
    def test_perfect_score(self):
        f1, p, r = compute_f1(10, 0, 0)
        assert f1 == pytest.approx(1.0, abs=0.001)
        assert p == pytest.approx(1.0, abs=0.001)
        assert r == pytest.approx(1.0, abs=0.001)

    def test_zero_detections(self):
        f1, p, r = compute_f1(0, 0, 10)
        assert f1 == pytest.approx(0.0, abs=0.01)
        assert r == pytest.approx(0.0, abs=0.01)

    def test_all_false_positives(self):
        f1, p, r = compute_f1(0, 10, 0)
        assert p == pytest.approx(0.0, abs=0.01)


class TestMatchDetections:
    def test_perfect_matching(self):
        """Detections at exact GT locations should all match."""
        gt_6nm = np.array([[100.0, 100.0], [200.0, 200.0]])
        gt_12nm = np.array([[300.0, 300.0]])

        dets = [
            {"x": 100.0, "y": 100.0, "class": "6nm", "conf": 0.9},
            {"x": 200.0, "y": 200.0, "class": "6nm", "conf": 0.8},
            {"x": 300.0, "y": 300.0, "class": "12nm", "conf": 0.7},
        ]

        results = match_detections_to_gt(dets, gt_6nm, gt_12nm)
        assert results["6nm"]["tp"] == 2
        assert results["6nm"]["fp"] == 0
        assert results["6nm"]["fn"] == 0
        assert results["12nm"]["tp"] == 1

    def test_wrong_class_no_match(self):
        """Detection near GT but wrong class should not match."""
        gt_6nm = np.array([[100.0, 100.0]])
        gt_12nm = np.empty((0, 2))

        dets = [
            {"x": 100.0, "y": 100.0, "class": "12nm", "conf": 0.9},
        ]

        results = match_detections_to_gt(dets, gt_6nm, gt_12nm)
        assert results["6nm"]["fn"] == 1  # missed
        assert results["12nm"]["fp"] == 1  # false positive

    def test_beyond_radius_no_match(self):
        """Detection beyond match radius should not match."""
        gt_6nm = np.array([[100.0, 100.0]])
        gt_12nm = np.empty((0, 2))

        dets = [
            {"x": 120.0, "y": 100.0, "class": "6nm", "conf": 0.9},  # 20px away > 9px radius
        ]

        results = match_detections_to_gt(
            dets, gt_6nm, gt_12nm, match_radii={"6nm": 9.0, "12nm": 15.0}
        )
        assert results["6nm"]["tp"] == 0
        assert results["6nm"]["fp"] == 1
        assert results["6nm"]["fn"] == 1

    def test_within_radius_matches(self):
        """Detection within match radius should match."""
        gt_6nm = np.array([[100.0, 100.0]])
        gt_12nm = np.empty((0, 2))

        dets = [
            {"x": 105.0, "y": 100.0, "class": "6nm", "conf": 0.9},  # 5px away < 9px
        ]

        results = match_detections_to_gt(
            dets, gt_6nm, gt_12nm, match_radii={"6nm": 9.0, "12nm": 15.0}
        )
        assert results["6nm"]["tp"] == 1

    def test_no_detections(self):
        """No detections: all GT are false negatives."""
        gt_6nm = np.array([[100.0, 100.0], [200.0, 200.0]])
        results = match_detections_to_gt([], gt_6nm, np.empty((0, 2)))
        assert results["6nm"]["fn"] == 2
        assert results["6nm"]["f1"] == pytest.approx(0.0, abs=0.01)

    def test_no_ground_truth(self):
        """No GT: all detections are false positives."""
        dets = [{"x": 100.0, "y": 100.0, "class": "6nm", "conf": 0.9}]
        results = match_detections_to_gt(dets, np.empty((0, 2)), np.empty((0, 2)))
        assert results["6nm"]["fp"] == 1


class TestAveragePrecision:
    def test_perfect_ap(self):
        """All detections match in rank order → AP = 1.0."""
        gt = np.array([[100.0, 100.0], [200.0, 200.0]])
        dets = [
            {"x": 100.0, "y": 100.0, "class": "6nm", "conf": 0.9},
            {"x": 200.0, "y": 200.0, "class": "6nm", "conf": 0.8},
        ]
        ap = compute_average_precision(dets, gt, match_radius=9.0)
        assert ap == pytest.approx(1.0, abs=0.01)
