"""
Inference script: detect immunogold particles in new images.

Usage:
    python predict.py --image path/to/image.tif --checkpoint checkpoints/fold_S1_seed42/phase3_best.pth
    python predict.py --fold S1 --checkpoint checkpoints/fold_S1_seed42/phase3_best.pth --config config/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.heatmap import extract_peaks
from src.model import ImmunogoldCenterNet
from src.postprocess import apply_structural_mask_filter, cross_class_nms
from src.preprocessing import load_image, load_mask
from src.ensemble import sliding_window_inference, d4_tta_predict
from src.visualize import overlay_annotations


def parse_args():
    parser = argparse.ArgumentParser(description="Predict immunogold particles")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--mask", type=str, help="Path to mask (optional)")
    parser.add_argument("--fold", type=str, help="Fold synapse ID for evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tta", action="store_true", help="Enable D4 TTA")
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="results/predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    # Load model
    model = ImmunogoldCenterNet(
        bifpn_channels=cfg["model"]["bifpn_channels"],
        bifpn_rounds=cfg["model"]["bifpn_rounds"],
        num_classes=cfg["model"]["num_classes"],
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val_f1={ckpt.get('val_f1_mean', '?')}")

    # Load image
    if args.fold:
        from src.preprocessing import discover_synapse_data, load_synapse
        records = discover_synapse_data(cfg["data"]["root"], cfg["data"]["synapse_ids"])
        record = [r for r in records if r.synapse_id == args.fold][0]
        data = load_synapse(record)
        image = data["image"]
        preprocessed = data["image"]
        mask = data["mask"]
        annotations = data["annotations"]
        name = args.fold
    else:
        image = load_image(Path(args.image))
        preprocessed = image
        mask = load_mask(Path(args.mask)) if args.mask else None
        annotations = {"6nm": np.empty((0, 2)), "12nm": np.empty((0, 2))}
        name = Path(args.image).stem

    # Inference
    if args.tta:
        print("Running D4 TTA inference...")
        heatmap_np, offset_np = d4_tta_predict(model, preprocessed, device)
    else:
        print("Running sliding window inference...")
        heatmap_np, offset_np = sliding_window_inference(
            model, preprocessed,
            patch_size=cfg["data"]["patch_size"],
            device=device,
        )

    # Extract detections
    heatmap_t = torch.from_numpy(heatmap_np)
    offset_t = torch.from_numpy(offset_np)

    detections = extract_peaks(
        heatmap_t, offset_t,
        stride=cfg["data"]["stride"],
        conf_threshold=args.conf_threshold,
        nms_kernel_sizes=cfg["postprocessing"]["nms_kernel_size"],
    )

    # Post-processing
    if mask is not None:
        detections = apply_structural_mask_filter(
            detections, mask,
            margin_px=cfg["postprocessing"]["mask_filter_margin_px"],
        )
    detections = cross_class_nms(
        detections, cfg["postprocessing"]["cross_class_nms_distance_px"],
    )

    # Print results
    n_6nm = sum(1 for d in detections if d["class"] == "6nm")
    n_12nm = sum(1 for d in detections if d["class"] == "12nm")
    print(f"\nDetections: {n_6nm} 6nm, {n_12nm} 12nm ({len(detections)} total)")

    # Evaluate if GT available
    if annotations and (len(annotations["6nm"]) > 0 or len(annotations["12nm"]) > 0):
        from src.evaluate import match_detections_to_gt
        results = match_detections_to_gt(
            detections, annotations["6nm"], annotations["12nm"],
            {k: float(v) for k, v in cfg["evaluation"]["match_radii_px"].items()},
        )
        for cls in ["6nm", "12nm", "overall"]:
            r = results[cls]
            print(f"  {cls}: F1={r['f1']:.3f}, P={r['precision']:.3f}, "
                  f"R={r['recall']:.3f} (TP={r['tp']}, FP={r['fp']}, FN={r['fn']})")

    # Save visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_annotations(
        image, annotations,
        title=f"{name} — {n_6nm} 6nm, {n_12nm} 12nm detected",
        save_path=output_dir / f"{name}_predictions.png",
        predictions=detections,
    )
    print(f"Saved overlay to {output_dir / f'{name}_predictions.png'}")


if __name__ == "__main__":
    main()
