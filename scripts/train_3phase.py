"""
Three-Phase Training Script for Immunogold Particle Detection

This script implements the progressive layer-unfreezing strategy used to train
the CenterNet detector on immunogold particles in transmission electron microscopy.

Phases:
    Phase 1: Encoder frozen, train detection heads (40 epochs)
    Phase 2: Unfreeze deep encoder layers, fine-tune (40 epochs)
    Phase 3: Full network training with layered learning rates (60 epochs)

Usage:
    python scripts/train_3phase.py --config config/config.yaml --device cuda:0
    python scripts/train_3phase.py --config config/config.yaml --device mps
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import ImmunogoldDataset
from src.model import ImmunogoldCenterNet
from src.loss import total_loss
from src.preprocessing import discover_synapse_data


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device, phase_name=""):
    """Train for one epoch and return average loss."""
    model.train()
    loss_sum = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        heatmaps_gt = batch["heatmap"].to(device)
        offsets_gt = batch["offsets"].to(device)
        offset_masks = batch["offset_mask"].to(device)
        conf_maps = batch["conf_map"].to(device)

        # Forward pass
        optimizer.zero_grad()
        heatmap_pred, offset_pred = model(images)

        # Compute loss
        loss, _, _ = total_loss(
            heatmap_pred, heatmaps_gt,
            offset_pred, offsets_gt,
            offset_masks,
            conf_weights=conf_maps,
        )

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_sum += loss.item()
        n_batches += 1

    avg_loss = loss_sum / n_batches
    return avg_loss


def train_phase_1(model, loader, device, num_epochs=40):
    """
    Phase 1: Frozen Encoder
    - ResNet-50 backbone remains frozen with CEM500K weights
    - Only BiFPN, Decoder, and detection heads are trained
    - High learning rate for new components (1e-3)
    """
    print("\n" + "="*70)
    print("PHASE 1: FROZEN ENCODER (40 epochs)")
    print("="*70)
    print("Training: BiFPN, Decoder, Heatmap Head, Offset Head")
    print("Frozen: ResNet-50 encoder layers")
    print("Learning rate: 1e-3 (high, since starting from scratch on these heads)")
    print()

    model.freeze_encoder()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2
    )

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, phase_name="Phase 1")
        scheduler.step()

        if epoch % 10 == 0 or epoch == num_epochs:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Time: {elapsed/60:6.1f}m")

    return start_time


def train_phase_2(model, loader, device, start_time, num_epochs=40):
    """
    Phase 2: Unfreeze Deep Encoder Layers
    - Layer3 and Layer4 of ResNet-50 are now trainable
    - Lower learning rates to prevent catastrophic forgetting
    - Higher rates for BiFPN and heads
    """
    print("\n" + "="*70)
    print("PHASE 2: FINE-TUNE DEEP ENCODER LAYERS (40 epochs)")
    print("="*70)
    print("Training: Layer3 (5e-5), Layer4 (1e-5), BiFPN+ (5e-4)")
    print("Frozen: ResNet-50 stem, Layer1, Layer2")
    print()

    model.unfreeze_deep_layers()

    optimizer = torch.optim.AdamW([
        {"params": model.layer3.parameters(), "lr": 5e-5},
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.bifpn.parameters(), "lr": 5e-4},
        {"params": model.upsample.parameters(), "lr": 5e-4},
        {"params": model.heatmap_head.parameters(), "lr": 5e-4},
        {"params": model.offset_head.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, phase_name="Phase 2")

        if epoch % 10 == 0 or epoch == num_epochs:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Time: {elapsed/60:6.1f}m")

    return start_time


def train_phase_3(model, loader, device, start_time, num_epochs=60):
    """
    Phase 3: Full Fine-tune
    - All layers now trainable with carefully tuned learning rates
    - Lowest rates for early layers (stem, layer1)
    - Progressively higher rates for deeper layers
    - Highest rates for new components (BiFPN, decoder, heads)
    """
    print("\n" + "="*70)
    print("PHASE 3: FULL NETWORK FINE-TUNE (60 epochs)")
    print("="*70)
    print("Training: All layers with layered learning rates")
    print("  Stem: 1e-6      (lowest - preserve ImageNet/CEM500K knowledge)")
    print("  Layer1: 5e-6    (early features)")
    print("  Layer2: 1e-5    (intermediate features)")
    print("  Layer3: 5e-5    (deep features)")
    print("  Layer4: 1e-4    (very deep features)")
    print("  BiFPN+: 2e-4    (highest - specialized for detection)")
    print()

    model.unfreeze_all()

    optimizer = torch.optim.AdamW([
        {"params": model.stem.parameters(), "lr": 1e-6},
        {"params": model.layer1.parameters(), "lr": 5e-6},
        {"params": model.layer2.parameters(), "lr": 1e-5},
        {"params": model.layer3.parameters(), "lr": 5e-5},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.bifpn.parameters(), "lr": 2e-4},
        {"params": model.upsample.parameters(), "lr": 2e-4},
        {"params": model.heatmap_head.parameters(), "lr": 2e-4},
        {"params": model.offset_head.parameters(), "lr": 2e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, phase_name="Phase 3")
        scheduler.step()

        if epoch % 10 == 0 or epoch == num_epochs:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Time: {elapsed/60:6.1f}m")


def main():
    parser = argparse.ArgumentParser(
        description="Three-phase training for immunogold particle detection"
    )
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config file")
    parser.add_argument("--device", default="auto",
                        help="Device: 'cuda:0', 'mps', or 'auto'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fold-id", default=None,
                        help="Hold-out image ID for LOOCV (None = use all images)")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    # Device selection
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"MidasMap: Three-Phase Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")
    print()

    # Load data
    records = discover_synapse_data(
        config["data"]["root"],
        config["data"]["synapse_ids"]
    )

    fold_id = args.fold_id if args.fold_id else "__NONE__"
    dataset = ImmunogoldDataset(
        records=records,
        fold_id=fold_id,
        mode="train",
        patch_size=config["data"]["patch_size"],
        stride=config["data"]["stride"],
        hard_mining_fraction=config["training"]["hard_mining_fraction"],
        copy_paste_per_class=config["training"]["copy_paste_per_class"],
        sigmas=config["heatmap"]["sigmas"],
        samples_per_epoch=500,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        worker_init_fn=ImmunogoldDataset.worker_init_fn,
    )

    n_particles = sum(
        len(a.get("6nm", [])) + len(a.get("12nm", []))
        for a in dataset.annotations.values()
    )

    if fold_id == "__NONE__":
        print(f"Training on ALL {len(dataset.images)} images")
    else:
        print(f"Training on {len(dataset.images)} images (holdout: {fold_id})")

    print(f"Total particles: {n_particles}")
    print(f"Dataset samples per epoch: {len(dataset)}")
    print()

    # Create model
    pretrained_path = config["model"].get("pretrained_weights")
    if pretrained_path and not Path(pretrained_path).exists():
        print(f"Warning: CEM500K weights not found at {pretrained_path}")
        print("Using ImageNet pretraining instead\n")
        pretrained_path = None

    model = ImmunogoldCenterNet(
        pretrained_path=pretrained_path,
        bifpn_channels=config["model"]["bifpn_channels"],
        bifpn_rounds=config["model"]["bifpn_rounds"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # Create checkpoint directory
    out_dir = Path("checkpoints/train_3phase")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train three phases
    start_time = time.time()

    train_phase_1(model, loader, device, num_epochs=40)
    torch.save(
        {"model_state_dict": model.state_dict(), "phase": 1, "epoch": 40},
        out_dir / "phase1_checkpoint.pth"
    )

    train_phase_2(model, loader, device, start_time, num_epochs=40)
    torch.save(
        {"model_state_dict": model.state_dict(), "phase": 2, "epoch": 80},
        out_dir / "phase2_checkpoint.pth"
    )

    train_phase_3(model, loader, device, start_time, num_epochs=60)
    torch.save(
        {"model_state_dict": model.state_dict(), "phase": 3, "epoch": 140},
        out_dir / "phase3_checkpoint.pth"
    )

    # Final model
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 140,
        "config": config,
        "seed": args.seed,
    }
    torch.save(final_checkpoint, out_dir / "final_model.pth")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training Complete: 140 epochs in {elapsed/3600:.2f}h")
    print(f"{'='*70}")
    print(f"Checkpoints saved to: {out_dir}")
    print(f"\nFinal model: {out_dir / 'final_model.pth'}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on test set: python evaluate_loocv.py")
    print(f"  2. Generate visualizations: python scripts/visualize_*.py")
    print(f"  3. Deploy model: python predict.py --checkpoint {out_dir / 'final_model.pth'}")


if __name__ == "__main__":
    main()
