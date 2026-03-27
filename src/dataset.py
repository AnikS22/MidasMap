"""
PyTorch Dataset for immunogold particle detection.

Implements patch-based training with:
- 70% hard mining (patches centered near particles)
- 30% random patches (background recognition)
- Copy-paste augmentation with Gaussian-blended bead bank
- Albumentations pipeline with keypoint co-transforms
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.heatmap import generate_heatmap_gt
from src.preprocessing import (
    SynapseRecord,
    load_all_annotations,
    load_image,
    load_mask,
)


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def get_train_augmentation() -> A.Compose:
    """
    Training augmentation pipeline.

    Conservative intensity limits: contrast delta is only 11-39 units on uint8.
    DO NOT use Cutout/Mixup/JPEG artifacts — they destroy or mimic particles.
    """
    return A.Compose(
        [
            # Geometric (co-transform keypoints)
            A.RandomRotate90(p=1.0),  # EM is rotation invariant
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Only ±10° to avoid interpolation artifacts that destroy contrast
            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            # Mild elastic deformation (simulates section flatness variation)
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            # Intensity (image only)
            A.RandomBrightnessContrast(
                brightness_limit=0.08,  # NOT default 0.2
                contrast_limit=0.08,
                p=0.7,
            ),
            # EM shot noise simulation
            A.GaussNoise(p=0.5),
            # Mild blur — simulate slight defocus
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=True,
            label_fields=["class_labels"],
        ),
    )


def get_val_augmentation() -> A.Compose:
    """No augmentation for validation — identity transform."""
    return A.Compose(
        [],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=True,
            label_fields=["class_labels"],
        ),
    )


# ---------------------------------------------------------------------------
# Bead bank for copy-paste augmentation
# ---------------------------------------------------------------------------

class BeadBank:
    """
    Pre-extracted particle crops for copy-paste augmentation.

    Stores small patches centered on annotated particles from training
    images. During training, random beads are pasted onto patches to
    increase particle density and address class imbalance.
    """

    def __init__(self):
        self.crops: Dict[str, List[Tuple[np.ndarray, int]]] = {
            "6nm": [],
            "12nm": [],
        }
        self.crop_sizes = {"6nm": 32, "12nm": 48}

    def extract_from_image(
        self,
        image: np.ndarray,
        annotations: Dict[str, np.ndarray],
    ):
        """Extract bead crops from a training image."""
        h, w = image.shape[:2]

        for cls, coords in annotations.items():
            crop_size = self.crop_sizes[cls]
            half = crop_size // 2

            for x, y in coords:
                xi, yi = int(round(x)), int(round(y))
                # Skip if too close to edge
                if yi - half < 0 or yi + half > h or xi - half < 0 or xi + half > w:
                    continue

                crop = image[yi - half : yi + half, xi - half : xi + half].copy()
                if crop.shape == (crop_size, crop_size):
                    self.crops[cls].append((crop, half))

    def paste_beads(
        self,
        image: np.ndarray,
        coords_6nm: List[Tuple[float, float]],
        coords_12nm: List[Tuple[float, float]],
        class_labels: List[str],
        mask: Optional[np.ndarray] = None,
        n_paste_per_class: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Tuple[float, float]], List[str]]:
        """
        Paste random beads onto image with Gaussian alpha blending.

        Returns augmented image and updated coordinate lists.
        """
        if rng is None:
            rng = np.random.default_rng()

        image = image.copy()
        h, w = image.shape[:2]
        new_coords_6nm = list(coords_6nm)
        new_coords_12nm = list(coords_12nm)
        new_labels = list(class_labels)

        for cls in ["6nm", "12nm"]:
            if not self.crops[cls]:
                continue

            crop_size = self.crop_sizes[cls]
            half = crop_size // 2
            n_paste = min(n_paste_per_class, len(self.crops[cls]))

            for _ in range(n_paste):
                # Random paste location (within image bounds)
                px = rng.integers(half + 5, w - half - 5)
                py = rng.integers(half + 5, h - half - 5)

                # Skip if outside tissue mask
                if mask is not None:
                    if py >= mask.shape[0] or px >= mask.shape[1] or not mask[py, px]:
                        continue

                # Check minimum distance from existing particles (avoid overlap)
                too_close = False
                all_existing = new_coords_6nm + new_coords_12nm
                for ex, ey in all_existing:
                    if (ex - px) ** 2 + (ey - py) ** 2 < (half * 1.5) ** 2:
                        too_close = True
                        break
                if too_close:
                    continue

                # Select random crop
                crop, _ = self.crops[cls][rng.integers(len(self.crops[cls]))]

                # Gaussian alpha mask for soft blending
                yy, xx = np.mgrid[:crop_size, :crop_size]
                center = crop_size / 2
                sigma = half * 0.7
                alpha = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))

                # Blend
                region = image[py - half : py + half, px - half : px + half]
                if region.shape != crop.shape:
                    continue
                blended = (alpha * crop + (1 - alpha) * region).astype(np.uint8)
                image[py - half : py + half, px - half : px + half] = blended

                # Add to annotations
                if cls == "6nm":
                    new_coords_6nm.append((float(px), float(py)))
                else:
                    new_coords_12nm.append((float(px), float(py)))
                new_labels.append(cls)

        return image, new_coords_6nm, new_coords_12nm, new_labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImmunogoldDataset(Dataset):
    """
    Patch-based dataset for immunogold particle detection.

    Sampling strategy:
    - 70% of patches centered within 100px of a known particle (hard mining)
    - 30% of patches at random locations (background recognition)

    This ensures the model sees particles in nearly every batch despite
    particles occupying <0.1% of image area.
    """

    def __init__(
        self,
        records: List[SynapseRecord],
        fold_id: str,
        mode: str = "train",
        patch_size: int = 512,
        stride: int = 2,
        hard_mining_fraction: float = 0.7,
        copy_paste_per_class: int = 5,
        sigmas: Optional[Dict[str, float]] = None,
        samples_per_epoch: int = 200,
        seed: int = 42,
    ):
        """
        Args:
            records: all SynapseRecord entries
            fold_id: synapse_id to hold out (test set)
            mode: 'train' or 'val'
            patch_size: training patch size
            stride: model output stride
            hard_mining_fraction: fraction of patches near particles
            copy_paste_per_class: beads to paste per class
            sigmas: heatmap Gaussian sigmas per class
            samples_per_epoch: virtual epoch size
            seed: random seed
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.hard_mining_fraction = hard_mining_fraction
        self.copy_paste_per_class = copy_paste_per_class if mode == "train" else 0
        self.sigmas = sigmas or {"6nm": 1.0, "12nm": 1.5}
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        self._base_seed = seed
        self.rng = np.random.default_rng(seed)

        # Split records
        if mode == "train":
            self.records = [r for r in records if r.synapse_id != fold_id]
        elif mode == "val":
            self.records = [r for r in records if r.synapse_id == fold_id]
        else:
            self.records = records

        # Pre-load all images and annotations into memory (~4MB each × 10 = 40MB)
        self.images = {}
        self.masks = {}
        self.annotations = {}

        for record in self.records:
            sid = record.synapse_id
            self.images[sid] = load_image(record.image_path)
            if record.mask_path:
                self.masks[sid] = load_mask(record.mask_path)
            self.annotations[sid] = load_all_annotations(record, self.images[sid].shape)

        # Build particle index for hard mining
        self._build_particle_index()

        # Build bead bank for copy-paste
        self.bead_bank = BeadBank()
        if mode == "train":
            for sid in self.images:
                self.bead_bank.extract_from_image(
                    self.images[sid], self.annotations[sid]
                )

        # Augmentation
        if mode == "train":
            self.transform = get_train_augmentation()
        else:
            self.transform = get_val_augmentation()

    def _build_particle_index(self):
        """Build flat index of all particles for hard mining."""
        self.particle_list = []  # (synapse_id, x, y, class)
        for sid, annots in self.annotations.items():
            for cls in ["6nm", "12nm"]:
                for x, y in annots[cls]:
                    self.particle_list.append((sid, x, y, cls))

    @staticmethod
    def worker_init_fn(worker_id: int):
        """Re-seed RNG per DataLoader worker to avoid identical sequences."""
        import torch
        seed = torch.initial_seed() % (2**32) + worker_id
        np.random.seed(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        # Reseed RNG using idx so each call produces a unique patch.
        # Without this, the same 200 patches repeat every epoch → instant overfitting.
        self.rng = np.random.default_rng(self._base_seed + idx + int(torch.initial_seed() % 100000))
        """
        Sample a patch with ground truth heatmap.

        Returns dict with:
            'image': (1, patch_size, patch_size) float32 tensor
            'heatmap': (2, patch_size//stride, patch_size//stride) float32
            'offsets': (2, patch_size//stride, patch_size//stride) float32
            'offset_mask': (patch_size//stride, patch_size//stride) bool
            'conf_map': (2, patch_size//stride, patch_size//stride) float32
        """
        # Decide: hard or random patch
        do_hard = (self.rng.random() < self.hard_mining_fraction
                   and len(self.particle_list) > 0
                   and self.mode == "train")

        if do_hard:
            # Pick random particle, center patch on it with jitter
            pidx = self.rng.integers(len(self.particle_list))
            sid, px, py, _ = self.particle_list[pidx]
            # Jitter center up to 128px
            jitter = 128
            cx = int(px + self.rng.integers(-jitter, jitter + 1))
            cy = int(py + self.rng.integers(-jitter, jitter + 1))
        else:
            # Random image and location
            sid = list(self.images.keys())[
                self.rng.integers(len(self.images))
            ]
            h, w = self.images[sid].shape[:2]
            cx = self.rng.integers(self.patch_size // 2, w - self.patch_size // 2)
            cy = self.rng.integers(self.patch_size // 2, h - self.patch_size // 2)

        # Extract patch
        image = self.images[sid]
        h, w = image.shape[:2]
        half = self.patch_size // 2

        # Clamp to image bounds
        cx = max(half, min(w - half, cx))
        cy = max(half, min(h - half, cy))

        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half

        patch = image[y0:y1, x0:x1].copy()

        # Pad if needed (edge cases)
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            padded = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
            ph, pw = patch.shape[:2]
            padded[:ph, :pw] = patch
            patch = padded

        # Get annotations within this patch (convert to patch-local coordinates)
        keypoints = []
        class_labels = []
        for cls in ["6nm", "12nm"]:
            for ax, ay in self.annotations[sid][cls]:
                # Convert to patch-local coords
                lx = ax - x0
                ly = ay - y0
                if 0 <= lx < self.patch_size and 0 <= ly < self.patch_size:
                    keypoints.append((lx, ly))
                    class_labels.append(cls)

        # Copy-paste augmentation (before geometric transforms)
        if self.copy_paste_per_class > 0 and self.mode == "train":
            local_6nm = [(x, y) for (x, y), c in zip(keypoints, class_labels) if c == "6nm"]
            local_12nm = [(x, y) for (x, y), c in zip(keypoints, class_labels) if c == "12nm"]
            mask_patch = None
            if sid in self.masks:
                mask_patch = self.masks[sid][y0:y1, x0:x1]

            patch, local_6nm, local_12nm, class_labels = self.bead_bank.paste_beads(
                patch, local_6nm, local_12nm, class_labels,
                mask=mask_patch,
                n_paste_per_class=self.copy_paste_per_class,
                rng=self.rng,
            )
            # Rebuild keypoints from updated coords
            keypoints = [(x, y) for x, y in local_6nm] + [(x, y) for x, y in local_12nm]
            class_labels = ["6nm"] * len(local_6nm) + ["12nm"] * len(local_12nm)

        # Apply augmentation (co-transforms keypoints)
        transformed = self.transform(
            image=patch,
            keypoints=keypoints,
            class_labels=class_labels,
        )
        patch_aug = transformed["image"]
        kp_aug = transformed["keypoints"]
        cl_aug = transformed["class_labels"]

        # Separate keypoints by class
        coords_6nm = np.array(
            [(x, y) for (x, y), c in zip(kp_aug, cl_aug) if c == "6nm"],
            dtype=np.float64,
        ).reshape(-1, 2)
        coords_12nm = np.array(
            [(x, y) for (x, y), c in zip(kp_aug, cl_aug) if c == "12nm"],
            dtype=np.float64,
        ).reshape(-1, 2)

        # Generate heatmap GT from TRANSFORMED coordinates (never warp heatmap)
        heatmap, offsets, offset_mask, conf_map = generate_heatmap_gt(
            coords_6nm, coords_12nm,
            self.patch_size, self.patch_size,
            sigmas=self.sigmas,
            stride=self.stride,
        )

        # Convert to tensors
        patch_tensor = torch.from_numpy(patch_aug).float().unsqueeze(0) / 255.0

        return {
            "image": patch_tensor,
            "heatmap": torch.from_numpy(heatmap),
            "offsets": torch.from_numpy(offsets),
            "offset_mask": torch.from_numpy(offset_mask),
            "conf_map": torch.from_numpy(conf_map),
        }
