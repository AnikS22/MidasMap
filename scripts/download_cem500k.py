"""
Download CEM500K MoCoV2 ResNet-50 pretrained weights from Zenodo.

Usage:
    python scripts/download_cem500k.py
    python scripts/download_cem500k.py --output weights/cem500k_mocov2_resnet50.pth.tar
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


CEM500K_URL = "https://zenodo.org/records/6453140/files/cem500k_mocov2_resnet50_200ep.pth.tar?download=1"
DEFAULT_OUTPUT = "weights/cem500k_mocov2_resnet50.pth.tar"


def download_with_progress(url: str, output_path: str):
    """Download file with progress indicator."""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")

    def _progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
            sys.stdout.flush()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urllib.request.urlretrieve(url, output_path, reporthook=_progress_hook)
    print()  # newline after progress


def verify_file(path: str):
    """Verify the downloaded file is a valid PyTorch checkpoint."""
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
        print(f"Checkpoint keys: {keys}")
        if "state_dict" in ckpt:
            n_params = len(ckpt["state_dict"])
            print(f"State dict entries: {n_params}")
        print("Verification PASSED")
        return True
    except Exception as e:
        print(f"Verification FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download CEM500K weights")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"File already exists: {args.output} ({size_mb:.1f} MB)")
        print("Use --force to re-download")
        verify_file(args.output)
        return

    download_with_progress(CEM500K_URL, args.output)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Downloaded: {size_mb:.1f} MB")

    verify_file(args.output)


if __name__ == "__main__":
    main()
