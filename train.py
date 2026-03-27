"""
Main training script for immunogold CenterNet.

Usage:
    python train.py --fold S1 --seed 42 --config config/config.yaml
    python train.py --fold S1 --seed 42 --config config/config.yaml --dry-run
    python train.py --fold S1 --seed 42 --config config/config.yaml --device cuda:0
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import ImmunogoldDataset
from src.evaluate import match_detections_to_gt
from src.heatmap import extract_peaks
from src.loss import total_loss
from src.model import ImmunogoldCenterNet
from src.preprocessing import discover_synapse_data, load_synapse
from src.ensemble import sliding_window_inference
from src.postprocess import cross_class_nms


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train immunogold CenterNet")
    parser.add_argument("--fold", type=str, required=True,
                        help="Synapse ID to hold out (e.g., S1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data, build model, run 1 batch, exit")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def validate_epoch(
    model, val_data, device, cfg, conf_threshold=0.3,
):
    """
    Run validation: sliding window inference on held-out image.

    Returns dict with val_loss, val_f1_6nm, val_f1_12nm, val_f1_mean.
    """
    model.eval()
    has_6nm = val_data["synapse_id"] not in cfg["data"].get("incomplete_6nm", [])

    with torch.no_grad():
        heatmap_np, offset_np = sliding_window_inference(
            model, val_data["image"],
            patch_size=cfg["data"]["patch_size"],
            device=device,
        )

    # Extract detections
    heatmap_t = torch.from_numpy(heatmap_np)
    offset_t = torch.from_numpy(offset_np)

    detections = extract_peaks(
        heatmap_t, offset_t,
        stride=cfg["data"]["stride"],
        conf_threshold=conf_threshold,
        nms_kernel_sizes=cfg["postprocessing"]["nms_kernel_size"],
    )
    detections = cross_class_nms(
        detections,
        cfg["postprocessing"]["cross_class_nms_distance_px"],
    )

    # Evaluate
    gt = val_data["annotations"]
    results = match_detections_to_gt(
        detections,
        gt.get("6nm", np.empty((0, 2))),
        gt.get("12nm", np.empty((0, 2))),
        match_radii={k: float(v) for k, v in cfg["evaluation"]["match_radii_px"].items()},
    )

    return {
        "val_f1_6nm": results["6nm"]["f1"] if has_6nm else float("nan"),
        "val_f1_12nm": results["12nm"]["f1"],
        "val_f1_mean": results["mean_f1"],
        "detections": detections,
        "results": results,
    }


def train_phase(
    model, train_loader, optimizer, scheduler, device, cfg,
    phase_num, n_epochs, writer, global_epoch, val_data,
    best_f1, checkpoint_dir, snapshot_epochs,
):
    """Train one phase, return updated global_epoch and best_f1."""
    model.train()
    focal_alpha = cfg["training"]["loss"]["focal_alpha"]
    focal_beta = cfg["training"]["loss"]["focal_beta"]
    lambda_offset = cfg["training"]["loss"]["lambda_offset"]
    patience = cfg["training"]["early_stopping"]["patience"]
    no_improve = 0

    for epoch in range(n_epochs):
        global_epoch += 1
        epoch_loss = 0.0
        epoch_hm_loss = 0.0
        epoch_off_loss = 0.0
        n_batches = 0

        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            hm_gt = batch["heatmap"].to(device)
            off_gt = batch["offsets"].to(device)
            off_mask = batch["offset_mask"].to(device)
            conf_map = batch["conf_map"].to(device)

            optimizer.zero_grad()
            hm_pred, off_pred = model(images)

            loss, hm_loss, off_loss = total_loss(
                hm_pred, hm_gt, off_pred, off_gt, off_mask,
                lambda_offset=lambda_offset,
                focal_alpha=focal_alpha,
                focal_beta=focal_beta,
                conf_weights=conf_map,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_hm_loss += hm_loss
            epoch_off_loss += off_loss
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_hm = epoch_hm_loss / max(n_batches, 1)
        avg_off = epoch_off_loss / max(n_batches, 1)

        # Log
        writer.add_scalar(f"Phase{phase_num}/train_loss", avg_loss, global_epoch)
        writer.add_scalar(f"Phase{phase_num}/hm_loss", avg_hm, global_epoch)
        writer.add_scalar(f"Phase{phase_num}/off_loss", avg_off, global_epoch)

        # Validate every 5 epochs
        val_metrics = None
        if global_epoch % 5 == 0 or epoch == n_epochs - 1:
            val_metrics = validate_epoch(model, val_data, device, cfg)
            writer.add_scalar(f"Phase{phase_num}/val_f1_mean", val_metrics["val_f1_mean"], global_epoch)

            if not np.isnan(val_metrics["val_f1_6nm"]):
                writer.add_scalar(f"Phase{phase_num}/val_f1_6nm", val_metrics["val_f1_6nm"], global_epoch)
            writer.add_scalar(f"Phase{phase_num}/val_f1_12nm", val_metrics["val_f1_12nm"], global_epoch)

            # Early stopping check
            if val_metrics["val_f1_mean"] > best_f1:
                best_f1 = val_metrics["val_f1_mean"]
                no_improve = 0
                # Save best checkpoint
                torch.save({
                    "epoch": global_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1_mean": best_f1,
                    "phase": phase_num,
                }, checkpoint_dir / f"phase{phase_num}_best.pth")
            else:
                no_improve += 5  # validated every 5 epochs

        # Snapshot checkpoints
        if global_epoch in snapshot_epochs:
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "val_f1_mean": best_f1,
                "phase": phase_num,
            }, checkpoint_dir / f"phase{phase_num}_{global_epoch}.pth")

        # Status
        f1_str = f", val_f1={val_metrics['val_f1_mean']:.4f}" if val_metrics else ""
        print(
            f"  Phase {phase_num} | Epoch {global_epoch:3d} | "
            f"Loss {avg_loss:.4f} (hm={avg_hm:.4f}, off={avg_off:.4f})"
            f"{f1_str}"
        )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {global_epoch} (patience={patience})")
            break

    return global_epoch, best_f1


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}, Fold: {args.fold}, Seed: {args.seed}")

    # Discover data
    records = discover_synapse_data(
        cfg["data"]["root"], cfg["data"]["synapse_ids"]
    )

    # Load validation image
    val_record = [r for r in records if r.synapse_id == args.fold]
    if not val_record:
        raise ValueError(f"Fold {args.fold} not found in synapse IDs")
    val_data = load_synapse(val_record[0])

    # Create dataset
    train_dataset = ImmunogoldDataset(
        records=records,
        fold_id=args.fold,
        mode="train",
        patch_size=cfg["data"]["patch_size"],
        stride=cfg["data"]["stride"],
        hard_mining_fraction=cfg["training"]["hard_mining_fraction"],
        copy_paste_per_class=cfg["training"]["copy_paste_per_class"],
        sigmas=cfg["heatmap"]["sigmas"],
        samples_per_epoch=200,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=ImmunogoldDataset.worker_init_fn,
    )

    # Build model
    pretrained = cfg["model"]["pretrained_weights"]
    if pretrained and not Path(pretrained).exists():
        print(f"Warning: CEM500K weights not found at {pretrained}, using ImageNet")
        pretrained = None

    model = ImmunogoldCenterNet(
        pretrained_path=pretrained,
        bifpn_channels=cfg["model"]["bifpn_channels"],
        bifpn_rounds=cfg["model"]["bifpn_rounds"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Checkpoint directory
    checkpoint_dir = Path("checkpoints") / f"fold_{args.fold}_seed{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=f"logs/fold_{args.fold}_seed{args.seed}")

    # Snapshot epochs for ensemble
    snapshot_epochs = set(cfg["training"]["n_snapshot_epochs"])

    # --- Dry run ---
    if args.dry_run:
        print("=== DRY RUN ===")
        batch = next(iter(train_loader))
        images = batch["image"].to(device)
        print(f"Input shape: {images.shape}")
        hm, off = model(images)
        print(f"Heatmap shape: {hm.shape}, Offset shape: {off.shape}")

        loss_val, hm_loss, off_loss = total_loss(
            hm, batch["heatmap"].to(device),
            off, batch["offsets"].to(device),
            batch["offset_mask"].to(device),
        )
        print(f"Loss: {loss_val.item():.4f} (hm={hm_loss:.4f}, off={off_loss:.4f})")
        print("=== DRY RUN PASSED ===")
        writer.close()
        return

    # --- Phase 1: Frozen encoder ---
    print("\n=== Phase 1: Frozen encoder ===")
    phase1_cfg = cfg["training"]["phases"]["phase1"]
    model.freeze_encoder()

    param_groups = model.get_param_groups(1, phase1_cfg)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=phase1_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    global_epoch = 0
    best_f1 = 0.0

    global_epoch, best_f1 = train_phase(
        model, train_loader, optimizer, scheduler, device, cfg,
        phase_num=1, n_epochs=phase1_cfg["epochs"],
        writer=writer, global_epoch=global_epoch,
        val_data=val_data, best_f1=best_f1,
        checkpoint_dir=checkpoint_dir,
        snapshot_epochs=snapshot_epochs,
    )

    # --- Phase 2: Unfreeze deep layers ---
    print("\n=== Phase 2: Unfreeze layer3+layer4 ===")
    phase2_cfg = cfg["training"]["phases"]["phase2"]
    model.unfreeze_deep_layers()

    param_groups = model.get_param_groups(2, phase2_cfg)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=phase2_cfg["weight_decay"]
    )
    scheduler = None  # No scheduler for phase 2

    global_epoch, best_f1 = train_phase(
        model, train_loader, optimizer, scheduler, device, cfg,
        phase_num=2, n_epochs=phase2_cfg["epochs"],
        writer=writer, global_epoch=global_epoch,
        val_data=val_data, best_f1=best_f1,
        checkpoint_dir=checkpoint_dir,
        snapshot_epochs=snapshot_epochs,
    )

    # --- Phase 3: Full fine-tuning ---
    print("\n=== Phase 3: Full fine-tuning ===")
    phase3_cfg = cfg["training"]["phases"]["phase3"]
    model.unfreeze_all()

    param_groups = model.get_param_groups(3, phase3_cfg)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=phase3_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase3_cfg["epochs"],
        eta_min=phase3_cfg["eta_min"],
    )

    global_epoch, best_f1 = train_phase(
        model, train_loader, optimizer, scheduler, device, cfg,
        phase_num=3, n_epochs=phase3_cfg["epochs"],
        writer=writer, global_epoch=global_epoch,
        val_data=val_data, best_f1=best_f1,
        checkpoint_dir=checkpoint_dir,
        snapshot_epochs=snapshot_epochs,
    )

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    writer.close()


if __name__ == "__main__":
    main()
