"""
Leave-One-Image-Out Cross-Validation (LOOCV) evaluation runner.

For each fold:
    test:  held-out image
    val:   next image (for threshold tuning)
    train: remaining images

CRITICAL: Image-level splits ONLY. Patch-level splits inflate F1 by 5-15%.

Usage:
    python evaluate_loocv.py --config config/config.yaml
    python evaluate_loocv.py --config config/config.yaml --ensemble-dir checkpoints/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluate import match_detections_to_gt, compute_f1
from src.heatmap import extract_peaks
from src.model import ImmunogoldCenterNet
from src.postprocess import (
    apply_structural_mask_filter,
    cross_class_nms,
    sweep_confidence_threshold,
)
from src.preprocessing import discover_synapse_data, load_synapse
from src.ensemble import ensemble_predict, sliding_window_inference
from src.visualize import overlay_annotations


def parse_args():
    parser = argparse.ArgumentParser(description="LOOCV evaluation")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--ensemble-dir", type=str, default="checkpoints",
                        help="Directory containing fold_*/phase3_*.pth")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--output", type=str, default="results/loocv_metrics.csv")
    return parser.parse_args()


def load_fold_models(ensemble_dir: Path, fold_id: str, cfg: dict,
                     device: torch.device):
    """Load all models for a fold (5 seeds × 3 snapshots = 15 models)."""
    models = []
    n_seeds = cfg["training"]["n_seeds"]
    snapshot_epochs = cfg["training"]["n_snapshot_epochs"]

    for seed_idx in range(n_seeds):
        seed = seed_idx + 42  # seeds start at 42
        fold_dir = ensemble_dir / f"fold_{fold_id}_seed{seed}"

        for epoch in snapshot_epochs:
            ckpt_path = fold_dir / f"phase3_{epoch}.pth"
            if not ckpt_path.exists():
                # Try best checkpoint instead
                ckpt_path = fold_dir / "phase3_best.pth"
            if not ckpt_path.exists():
                continue

            model = ImmunogoldCenterNet(
                bifpn_channels=cfg["model"]["bifpn_channels"],
                bifpn_rounds=cfg["model"]["bifpn_rounds"],
                num_classes=cfg["model"]["num_classes"],
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            models.append(model)

    return models


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    records = discover_synapse_data(cfg["data"]["root"], cfg["data"]["synapse_ids"])
    synapse_ids = cfg["data"]["synapse_ids"]
    incomplete_6nm = set(cfg["data"].get("incomplete_6nm", []))
    ensemble_dir = Path(args.ensemble_dir)

    all_results = []
    match_radii = {k: float(v) for k, v in cfg["evaluation"]["match_radii_px"].items()}
    val_offset = cfg["evaluation"]["loocv_val_offset"]

    for test_idx, test_sid in enumerate(synapse_ids):
        print(f"\n{'='*60}")
        print(f"Fold {test_idx}: test={test_sid}")

        # Val image for threshold tuning
        val_idx = (test_idx + val_offset) % len(synapse_ids)
        val_sid = synapse_ids[val_idx]

        # Load test and val data
        test_record = [r for r in records if r.synapse_id == test_sid][0]
        val_record = [r for r in records if r.synapse_id == val_sid][0]

        test_data = load_synapse(test_record)
        val_data = load_synapse(val_record)

        has_6nm = test_sid not in incomplete_6nm

        # Load ensemble models
        models = load_fold_models(ensemble_dir, test_sid, cfg, device)
        if not models:
            print(f"  No models found for fold {test_sid}, skipping")
            all_results.append({
                "fold": test_sid,
                "n_models": 0,
                "6nm_f1": float("nan"),
                "12nm_f1": float("nan"),
                "mean_f1": float("nan"),
            })
            continue

        print(f"  Loaded {len(models)} ensemble members")

        # Tune threshold on validation image
        val_hm, val_off = ensemble_predict(
            models, val_data["image"], device, use_tta=args.use_tta,
        )
        val_hm_t = torch.from_numpy(val_hm)
        val_off_t = torch.from_numpy(val_off)

        # Get all detections at low threshold for sweep
        val_dets = extract_peaks(
            val_hm_t, val_off_t, stride=cfg["data"]["stride"],
            conf_threshold=0.05,
            nms_kernel_sizes=cfg["postprocessing"]["nms_kernel_size"],
        )
        best_thresholds = sweep_confidence_threshold(
            val_dets, val_data["annotations"], match_radii,
        )
        print(f"  Best thresholds: {best_thresholds}")

        # Test inference
        test_hm, test_off = ensemble_predict(
            models, test_data["image"], device, use_tta=args.use_tta,
        )
        test_hm_t = torch.from_numpy(test_hm)
        test_off_t = torch.from_numpy(test_off)

        # Use per-class thresholds
        all_detections = []
        for cls in ["6nm", "12nm"]:
            thr = best_thresholds.get(cls, 0.3)
            cls_dets = extract_peaks(
                test_hm_t, test_off_t,
                stride=cfg["data"]["stride"],
                conf_threshold=thr,
                nms_kernel_sizes=cfg["postprocessing"]["nms_kernel_size"],
            )
            all_detections.extend([d for d in cls_dets if d["class"] == cls])

        # Post-processing
        if test_data["mask"] is not None:
            all_detections = apply_structural_mask_filter(
                all_detections, test_data["mask"],
                margin_px=cfg["postprocessing"]["mask_filter_margin_px"],
            )
        all_detections = cross_class_nms(
            all_detections, cfg["postprocessing"]["cross_class_nms_distance_px"],
        )

        # Evaluate
        results = match_detections_to_gt(
            all_detections,
            test_data["annotations"].get("6nm", np.empty((0, 2))),
            test_data["annotations"].get("12nm", np.empty((0, 2))),
            match_radii,
        )

        fold_result = {
            "fold": test_sid,
            "n_models": len(models),
            "6nm_f1": results["6nm"]["f1"] if has_6nm else float("nan"),
            "6nm_precision": results["6nm"]["precision"] if has_6nm else float("nan"),
            "6nm_recall": results["6nm"]["recall"] if has_6nm else float("nan"),
            "12nm_f1": results["12nm"]["f1"],
            "12nm_precision": results["12nm"]["precision"],
            "12nm_recall": results["12nm"]["recall"],
            "mean_f1": results["mean_f1"],
        }
        all_results.append(fold_result)

        for cls in ["6nm", "12nm"]:
            r = results[cls]
            note = " (N/A)" if cls == "6nm" and not has_6nm else ""
            print(f"  {cls}: F1={r['f1']:.3f}, P={r['precision']:.3f}, "
                  f"R={r['recall']:.3f}{note}")
        print(f"  Mean F1: {results['mean_f1']:.3f}")

        # Save per-fold visualization
        overlay_annotations(
            test_data["image"], test_data["annotations"],
            title=f"Fold {test_sid} — F1={results['mean_f1']:.3f}",
            save_path=Path("results/per_fold_predictions") / f"{test_sid}.png",
            predictions=all_detections,
        )

    # Summary
    df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("LOOCV Results:")
    f1_6nm = df["6nm_f1"].dropna()
    f1_12nm = df["12nm_f1"].dropna()
    mean_f1 = df["mean_f1"].dropna()

    print(f"  6nm  F1: {f1_6nm.mean():.3f} ± {f1_6nm.std():.3f} (n={len(f1_6nm)})")
    print(f"  12nm F1: {f1_12nm.mean():.3f} ± {f1_12nm.std():.3f} (n={len(f1_12nm)})")
    print(f"  Mean F1: {mean_f1.mean():.3f} ± {mean_f1.std():.3f}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
