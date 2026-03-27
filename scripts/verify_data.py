"""
Data verification script: loads all images and annotations,
validates counts, and saves visual overlays.

Usage:
    python scripts/verify_data.py --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import discover_synapse_data, load_synapse
from src.visualize import overlay_annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("Immunogold Data Verification")
    print("=" * 60)

    records = discover_synapse_data(cfg["data"]["root"], cfg["data"]["synapse_ids"])

    total_6nm = 0
    total_12nm = 0
    output_dir = Path("results/verification")

    for record in records:
        print(f"\n--- {record.synapse_id} ---")
        print(f"  Image: {record.image_path.name}")
        print(f"  Mask:  {record.mask_path.name if record.mask_path else 'NONE'}")
        print(f"  6nm CSVs:  {[p.name for p in record.csv_6nm_paths]}")
        print(f"  12nm CSVs: {[p.name for p in record.csv_12nm_paths]}")

        data = load_synapse(record)
        img = data["image"]
        annots = data["annotations"]

        n6 = len(annots["6nm"])
        n12 = len(annots["12nm"])
        total_6nm += n6
        total_12nm += n12

        print(f"  Image shape: {img.shape}")
        print(f"  6nm particles: {n6}")
        print(f"  12nm particles: {n12}")

        if data["mask"] is not None:
            # Check how many particles fall within mask
            mask = data["mask"]
            for cls, coords in annots.items():
                if len(coords) == 0:
                    continue
                inside = sum(
                    1 for x, y in coords
                    if 0 <= int(y) < mask.shape[0] and
                       0 <= int(x) < mask.shape[1] and
                       mask[int(y), int(x)]
                )
                print(f"  {cls} in mask: {inside}/{len(coords)}")

        # Save overlay
        overlay_annotations(
            img, annots,
            title=f"{record.synapse_id}: {n6} 6nm, {n12} 12nm",
            save_path=output_dir / f"{record.synapse_id}_annotations.png",
        )

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_6nm} 6nm + {total_12nm} 12nm = {total_6nm + total_12nm}")
    print(f"Expected: 403 6nm + 50 12nm = 453")

    if total_6nm + total_12nm >= 400:
        print("PASS: Particle counts look reasonable")
    else:
        print("WARNING: Total count is lower than expected")

    print(f"\nOverlays saved to: {output_dir}")


if __name__ == "__main__":
    main()
