"""
Data loading, annotation parsing, and preprocessing for immunogold TEM images.

The model receives raw images — the CEM500K backbone was pretrained on raw EM.
Top-hat preprocessing is only used by LodeStar (Stage 1).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile


# ---------------------------------------------------------------------------
# Data registry: robust discovery of images, masks, and annotations
# ---------------------------------------------------------------------------

@dataclass
class SynapseRecord:
    """Metadata for one synapse sample."""
    synapse_id: str
    image_path: Path
    mask_path: Optional[Path]
    csv_6nm_paths: List[Path] = field(default_factory=list)
    csv_12nm_paths: List[Path] = field(default_factory=list)
    has_6nm: bool = False
    has_12nm: bool = False


def discover_synapse_data(root: str, synapse_ids: List[str]) -> List[SynapseRecord]:
    """
    Discover all TIF images, masks, and CSV annotations for each synapse.

    Handles naming inconsistencies:
    - S22: main image is S22_0003.tif, two Results folders
    - S25: 12nm CSV has no space ("Results12nm")
    - CSV patterns: "Results 6nm XY" vs "Results XY in microns 6nm"
    """
    root = Path(root)
    analyzed = root / "analyzed synapses"
    records = []

    for sid in synapse_ids:
        folder = analyzed / sid
        if not folder.exists():
            raise FileNotFoundError(f"Synapse folder not found: {folder}")

        # --- Find main image (TIF without 'mask' or 'color' in name) ---
        all_tifs = list(folder.glob("*.tif"))
        main_tifs = [
            t for t in all_tifs
            if "mask" not in t.stem.lower() and "color" not in t.stem.lower()
        ]
        if not main_tifs:
            raise FileNotFoundError(f"No main image found in {folder}")
        # Prefer the largest file (main EM image) if multiple found
        image_path = max(main_tifs, key=lambda t: t.stat().st_size)

        # --- Find mask ---
        mask_tifs = [t for t in all_tifs if "mask" in t.stem.lower()]
        mask_path = None
        if mask_tifs:
            # Prefer plain "mask.tif" over "mask 1.tif" / "mask 2.tif"
            plain = [t for t in mask_tifs if t.stem.lower().endswith("mask")]
            mask_path = plain[0] if plain else mask_tifs[0]

        # --- Find CSVs across all Results* subdirectories ---
        results_dirs = sorted(folder.glob("Results*"))
        # Also check direct subdirs like "Results 1", "Results 2"
        csv_6nm_paths = []
        csv_12nm_paths = []

        for rdir in results_dirs:
            if rdir.is_dir():
                for csv_file in rdir.glob("*.csv"):
                    name_lower = csv_file.name.lower()
                    if "6nm" in name_lower:
                        csv_6nm_paths.append(csv_file)
                    elif "12nm" in name_lower:
                        csv_12nm_paths.append(csv_file)

        record = SynapseRecord(
            synapse_id=sid,
            image_path=image_path,
            mask_path=mask_path,
            csv_6nm_paths=csv_6nm_paths,
            csv_12nm_paths=csv_12nm_paths,
            has_6nm=len(csv_6nm_paths) > 0,
            has_12nm=len(csv_12nm_paths) > 0,
        )
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """
    Load a TIF image as grayscale uint8.

    Handles:
    - RGB images (take first channel)
    - Palette-mode images
    - Already-grayscale images
    """
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        # RGB or multi-channel — take first channel (all channels identical in these images)
        img = img[:, :, 0] if img.shape[2] <= 4 else img[0]
    return img.astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    """
    Load mask TIF as binary array.

    Mask is RGB where tissue regions have values < 250 in at least one channel.
    Returns boolean array: True = tissue/structural region.
    """
    mask_rgb = tifffile.imread(str(path))
    if mask_rgb.ndim == 2:
        return mask_rgb < 250
    # RGB mask: tissue where any channel is not white
    return np.any(mask_rgb < 250, axis=-1)


# ---------------------------------------------------------------------------
# Annotation loading and coordinate conversion
# ---------------------------------------------------------------------------

def load_annotations_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load annotation CSV with columns [index, X, Y].

    CSV headers have leading space: " ,X,Y".
    Coordinates are normalized [0, 1] despite 'microns' in filename.
    """
    df = pd.read_csv(csv_path)
    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    # Rename unnamed index column
    if "" in df.columns:
        df = df.rename(columns={"": "idx"})
    return df[["X", "Y"]]


def load_all_annotations(
    record: SynapseRecord, image_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """
    Load and convert annotations for one synapse to pixel coordinates.

    Args:
        record: SynapseRecord with CSV paths.
        image_shape: (height, width) of the corresponding image.

    Returns:
        Dictionary with keys '6nm' and '12nm', each containing
        an Nx2 array of (x, y) pixel coordinates.
    """
    h, w = image_shape[:2]
    result = {"6nm": np.empty((0, 2), dtype=np.float64),
              "12nm": np.empty((0, 2), dtype=np.float64)}

    for cls, paths in [("6nm", record.csv_6nm_paths),
                       ("12nm", record.csv_12nm_paths)]:
        all_coords = []
        for csv_path in paths:
            df = load_annotations_csv(csv_path)
            # Validate normalized coordinates
            assert df["X"].between(0, 1).all(), \
                f"X coords not in [0,1] in {csv_path}"
            assert df["Y"].between(0, 1).all(), \
                f"Y coords not in [0,1] in {csv_path}"
            # Convert to pixel space
            px_x = df["X"].values * w
            px_y = df["Y"].values * h
            all_coords.append(np.stack([px_x, px_y], axis=1))

        if all_coords:
            coords = np.concatenate(all_coords, axis=0)
            # Deduplicate (for S22 merged results): remove within 3px
            if len(coords) > 1:
                coords = _deduplicate_coords(coords, min_dist=3.0)
            result[cls] = coords

    return result


def _deduplicate_coords(
    coords: np.ndarray, min_dist: float = 3.0
) -> np.ndarray:
    """Remove duplicate coordinates within min_dist pixels."""
    from scipy.spatial.distance import cdist

    if len(coords) <= 1:
        return coords
    dists = cdist(coords, coords)
    np.fill_diagonal(dists, np.inf)
    keep = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if not keep[i]:
            continue
        # Mark later duplicates
        for j in range(i + 1, len(coords)):
            if keep[j] and dists[i, j] < min_dist:
                keep[j] = False
    return coords[keep]


# ---------------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------------

def preprocess_image(img: np.ndarray, bead_class: str,
                     tophat_radii: Optional[Dict[str, int]] = None,
                     clahe_clip_limit: float = 0.03,
                     clahe_kernel_size: int = 64) -> np.ndarray:
    """
    Top-hat + CLAHE preprocessing. Used ONLY by LodeStar (Stage 1).

    Not used for model training — the CEM500K backbone expects raw EM images.
    """
    from skimage import exposure
    from skimage.morphology import disk, white_tophat

    if tophat_radii is None:
        tophat_radii = {"6nm": 8, "12nm": 12}

    img_inv = (255 - img).astype(np.float32)
    radius = tophat_radii[bead_class]
    tophat = white_tophat(img_inv, disk(radius))

    tophat_max = tophat.max()
    if tophat_max > 0:
        tophat_norm = tophat / tophat_max
    else:
        tophat_norm = tophat

    enhanced = exposure.equalize_adapthist(
        tophat_norm,
        clip_limit=clahe_clip_limit,
        kernel_size=clahe_kernel_size,
    )
    return (enhanced * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Convenience: load everything for one synapse
# ---------------------------------------------------------------------------

def load_synapse(record: SynapseRecord) -> dict:
    """
    Load image, mask, and annotations for one synapse.

    Returns dict with keys: 'image', 'mask', 'annotations',
                            'synapse_id', 'image_shape'
    """
    img = load_image(record.image_path)
    mask = load_mask(record.mask_path) if record.mask_path else None
    annotations = load_all_annotations(record, img.shape)

    return {
        "synapse_id": record.synapse_id,
        "image": img,
        "mask": mask,
        "annotations": annotations,
        "image_shape": img.shape,
    }
