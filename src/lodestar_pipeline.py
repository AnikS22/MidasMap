"""
Stage 1: LodeStar pseudo-label generation.

LodeStar (Midtvedt et al., Nature Communications 2022) is a self-supervised
particle detector that trains from a SINGLE unlabeled particle crop.
Uses DeepTrack2 library (requires TensorFlow).

Purpose:
    1. Validate existing manual annotations
    2. Discover potentially missed particles
    3. Generate pseudo-labels for images with sparse/missing annotations
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.preprocessing import preprocess_image


def select_best_crop(
    image: np.ndarray,
    coords: np.ndarray,
    bead_class: str,
    crop_sizes: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Select the best example crop for LodeStar training.

    Picks the particle with highest local contrast (most clearly visible).
    """
    if crop_sizes is None:
        crop_sizes = {"6nm": 32, "12nm": 48}

    crop_size = crop_sizes[bead_class]
    half = crop_size // 2
    h, w = image.shape[:2]

    best_crop = None
    best_contrast = -1

    for x, y in coords:
        xi, yi = int(round(x)), int(round(y))
        if yi - half < 0 or yi + half > h or xi - half < 0 or xi + half > w:
            continue

        crop = image[yi - half : yi + half, xi - half : xi + half]
        if crop.shape != (crop_size, crop_size):
            continue

        # Contrast: difference between center and border
        center_val = crop[half - 2 : half + 2, half - 2 : half + 2].mean()
        border_val = np.concatenate([
            crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]
        ]).mean()
        contrast = abs(center_val - border_val)

        if contrast > best_contrast:
            best_contrast = contrast
            best_crop = crop.copy()

    if best_crop is None:
        raise ValueError(
            f"Could not extract any valid crop for class {bead_class}"
        )

    return best_crop


def train_lodestar_model(
    example_crop: np.ndarray,
    bead_class: str,
    epochs: int = 100,
):
    """
    Train a LodeStar model from a single example crop.

    LodeStar exploits roto-translational equivariance to generalize
    from one particle to detect all similar particles in full images.

    Args:
        example_crop: preprocessed crop centered on a known bead
        bead_class: '6nm' or '12nm'
        epochs: training epochs

    Returns:
        Trained LodeStar model
    """
    try:
        import deeptrack as dt
    except ImportError:
        raise ImportError(
            "DeepTrack2 required for LodeStar. Install via: pip install deeptrack"
        )

    # LodeStar model from DeepTrack2
    model = dt.models.LodeSTAR(
        input_shape=example_crop.shape + (1,),
    )

    # Create training data pipeline from single crop
    # LodeStar uses self-supervised augmentation internally
    crop_norm = example_crop.astype(np.float32) / 255.0
    crop_4d = crop_norm[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)

    model.fit(
        crop_4d,
        epochs=epochs,
        verbose=1,
    )

    return model


def lodestar_inference(
    model,
    preprocessed_image: np.ndarray,
    confidence_percentile: float = 70,
) -> List[Tuple[float, float, float]]:
    """
    Run LodeStar inference on a full preprocessed image.

    Args:
        model: trained LodeStar model
        preprocessed_image: top-hat + CLAHE preprocessed uint8 image
        confidence_percentile: keep detections above this percentile

    Returns:
        List of (x, y, confidence) detections
    """
    img_norm = preprocessed_image.astype(np.float32) / 255.0
    img_4d = img_norm[np.newaxis, :, :, np.newaxis]

    # LodeStar returns positions and weights
    positions, weights = model.predict(img_4d)

    if len(positions) == 0:
        return []

    # Filter by confidence percentile
    threshold = np.percentile(weights, confidence_percentile)
    mask = weights >= threshold

    detections = []
    for pos, w in zip(positions[mask], weights[mask]):
        detections.append((float(pos[0]), float(pos[1]), float(w)))

    return detections


def merge_annotations(
    manual_coords: np.ndarray,
    lodestar_detections: List[Tuple[float, float, float]],
    confirmation_radius: float = 8.0,
    discovery_radius: float = 15.0,
    discovery_confidence: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge LodeStar pseudo-labels with manual annotations.

    Strategy:
    - LodeStar within confirmation_radius of manual → confirmed (conf=1.0)
    - LodeStar without nearby manual, high confidence → discovered (conf=0.5)
    - Manual without LodeStar match → manual only (conf=1.0)

    Returns:
        merged_coords: (N, 2) array of (x, y)
        merged_confidence: (N,) array of confidence weights
    """
    if len(lodestar_detections) == 0:
        if len(manual_coords) == 0:
            return np.empty((0, 2)), np.empty(0)
        return manual_coords.copy(), np.ones(len(manual_coords))

    lode_coords = np.array([(x, y) for x, y, _ in lodestar_detections])
    lode_confs = np.array([c for _, _, c in lodestar_detections])

    merged_coords = []
    merged_confs = []
    manual_matched = set()

    if len(manual_coords) > 0 and len(lode_coords) > 0:
        dists = cdist(lode_coords, manual_coords)

        for i in range(len(lode_coords)):
            min_dist = dists[i].min() if len(manual_coords) > 0 else np.inf
            min_j = dists[i].argmin() if len(manual_coords) > 0 else -1

            if min_dist < confirmation_radius:
                # Confirmed: use manual position with full confidence
                merged_coords.append(manual_coords[min_j])
                merged_confs.append(1.0)
                manual_matched.add(min_j)
            elif min_dist > discovery_radius:
                # Discovered: new detection
                merged_coords.append(lode_coords[i])
                merged_confs.append(discovery_confidence)

    # Add unmatched manual annotations
    for j in range(len(manual_coords)):
        if j not in manual_matched:
            merged_coords.append(manual_coords[j])
            merged_confs.append(1.0)

    if not merged_coords:
        return np.empty((0, 2)), np.empty(0)

    return np.array(merged_coords), np.array(merged_confs)


def save_merged_annotations(
    coords: np.ndarray,
    confidences: np.ndarray,
    output_path: Path,
):
    """Save merged annotations to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "X": coords[:, 0] if len(coords) > 0 else [],
        "Y": coords[:, 1] if len(coords) > 0 else [],
        "confidence": confidences if len(confidences) > 0 else [],
    })
    df.to_csv(output_path, index=True)


def run_lodestar_pipeline(
    synapse_data: List[dict],
    bead_class: str,
    output_dir: str,
    tophat_radii: Optional[Dict[str, int]] = None,
    crop_sizes: Optional[Dict[str, int]] = None,
    confidence_percentile: float = 70,
    confirmation_radius: float = 8.0,
    discovery_radius: float = 15.0,
):
    """
    Full LodeStar pipeline for one bead class.

    Args:
        synapse_data: list of dicts from load_synapse()
        bead_class: '6nm' or '12nm'
        output_dir: directory for merged annotation CSVs
    """
    output_dir = Path(output_dir)

    # Find best training crop from the image with most annotations
    best_image = None
    best_coords = None
    max_count = 0

    for data in synapse_data:
        coords = data["annotations"][bead_class]
        if len(coords) > max_count:
            max_count = len(coords)
            best_image = data["image"]
            best_coords = coords

    if max_count == 0:
        print(f"No annotations for class {bead_class}, skipping LodeStar")
        return

    # Preprocess the best image
    preprocessed = preprocess_image(best_image, bead_class, tophat_radii)

    # Extract best crop
    crop = select_best_crop(preprocessed, best_coords, bead_class, crop_sizes)

    # Train LodeStar
    print(f"Training LodeStar for {bead_class} from {max_count}-particle image...")
    model = train_lodestar_model(crop, bead_class)

    # Run inference and merge for each image
    for data in synapse_data:
        sid = data["synapse_id"]
        img = data["image"]
        manual = data["annotations"][bead_class]

        # Preprocess
        preprocessed = preprocess_image(img, bead_class, tophat_radii)

        # Detect
        detections = lodestar_inference(model, preprocessed, confidence_percentile)

        # Merge
        merged_coords, merged_confs = merge_annotations(
            manual, detections,
            confirmation_radius=confirmation_radius,
            discovery_radius=discovery_radius,
        )

        # Save
        out_path = output_dir / f"{sid}_{bead_class}_merged.csv"
        save_merged_annotations(merged_coords, merged_confs, out_path)

        print(
            f"  {sid}: manual={len(manual)}, "
            f"lodestar={len(detections)}, "
            f"merged={len(merged_coords)}"
        )
