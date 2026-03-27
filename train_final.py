"""
Train the final deployable model on ALL 10 images (no holdout).

LOOCV proved F1=0.94. This trains the production model using every
labeled particle for maximum generalization to new unseen images.

Usage:
    python train_final.py --config config/config.yaml --device cuda:0
    python train_final.py --config config/config.yaml --device mps
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import ImmunogoldDataset
from src.model import ImmunogoldCenterNet
from src.loss import total_loss
from src.preprocessing import discover_synapse_data, load_synapse


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = 0
    n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        optimizer.zero_grad()
        hm_pred, off_pred = model(imgs)
        loss, hm_l, off_l = total_loss(
            hm_pred, batch["heatmap"].to(device),
            off_pred, batch["offsets"].to(device),
            batch["offset_mask"].to(device),
            conf_weights=batch["conf_map"].to(device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_sum += loss.item()
        n += 1
    return loss_sum / n


def main():
    parser = argparse.ArgumentParser(description="Train final deployable model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load ALL data — no holdout
    records = discover_synapse_data(cfg["data"]["root"], cfg["data"]["synapse_ids"])

    # Dataset uses ALL images for training (fold_id=None means no exclusion)
    dataset = ImmunogoldDataset(
        records=records,
        fold_id="__NONE__",  # no image excluded
        mode="train",
        patch_size=cfg["data"]["patch_size"],
        stride=cfg["data"]["stride"],
        hard_mining_fraction=cfg["training"]["hard_mining_fraction"],
        copy_paste_per_class=cfg["training"]["copy_paste_per_class"],
        sigmas=cfg["heatmap"]["sigmas"],
        samples_per_epoch=500,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=4, drop_last=True,
        worker_init_fn=ImmunogoldDataset.worker_init_fn,
    )

    print(f"Training on ALL {len(dataset.images)} images, "
          f"{sum(len(a['6nm'])+len(a['12nm']) for a in dataset.annotations.values())} particles")

    # Model
    pretrained = cfg["model"]["pretrained_weights"]
    if not Path(pretrained).exists():
        pretrained = None
        print("Warning: CEM500K weights not found, using ImageNet")

    model = ImmunogoldCenterNet(
        pretrained_path=pretrained,
        bifpn_channels=cfg["model"]["bifpn_channels"],
        bifpn_rounds=cfg["model"]["bifpn_rounds"],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    out_dir = Path("checkpoints/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Phase 1: Frozen encoder (40 epochs — slightly shorter since more data)
    print("\n=== Phase 1: Frozen encoder (40 epochs) ===")
    model.freeze_encoder()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)

    for ep in range(1, 41):
        loss = train_epoch(model, loader, opt, device)
        sched.step()
        if ep % 10 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {ep:3d} | loss={loss:.4f} | {elapsed:.0f}s")

    torch.save({"model_state_dict": model.state_dict(), "epoch": 40},
               out_dir / "phase1.pth")

    # Phase 2: Unfreeze deep layers (40 epochs)
    print("\n=== Phase 2: Unfreeze layer3+4 (40 epochs) ===")
    model.unfreeze_deep_layers()
    opt = torch.optim.AdamW([
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 5e-5},
        {"params": model.bifpn.parameters(), "lr": 5e-4},
        {"params": model.upsample.parameters(), "lr": 5e-4},
        {"params": model.heatmap_head.parameters(), "lr": 5e-4},
        {"params": model.offset_head.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)

    for ep in range(41, 81):
        loss = train_epoch(model, loader, opt, device)
        if ep % 10 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {ep:3d} | loss={loss:.4f} | {elapsed:.0f}s")

    torch.save({"model_state_dict": model.state_dict(), "epoch": 80},
               out_dir / "phase2.pth")

    # Phase 3: Full fine-tune (60 epochs)
    print("\n=== Phase 3: Full fine-tune (60 epochs) ===")
    model.unfreeze_all()
    opt = torch.optim.AdamW([
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-7)

    for ep in range(81, 141):
        loss = train_epoch(model, loader, opt, device)
        sched.step()
        if ep % 10 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {ep:3d} | loss={loss:.4f} | {elapsed:.0f}s")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep,
            }, out_dir / f"phase3_{ep}.pth")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 140,
        "config": cfg,
    }, out_dir / "final_model.pth")

    elapsed = time.time() - start
    print(f"\n=== Done: 140 epochs in {elapsed:.0f}s ({elapsed/60:.1f}min) ===")
    print(f"Final model: {out_dir / 'final_model.pth'}")
    print(f"\nTo detect particles in a new image:")
    print(f"  python predict.py --image path/to/new_image.tif --checkpoint {out_dir / 'final_model.pth'}")


if __name__ == "__main__":
    main()
