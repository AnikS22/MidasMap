"""
Generate synthetic TEM-like images with simulated gold particles for pipeline visualization.

This creates demo images that look realistic enough to show the pipeline,
even when the actual training dataset isn't available.

Usage:
    python scripts/generate_demo_data.py --output demo_images/ --n-images 3
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import os

def create_tem_background(height=2048, width=2048, noise_level=15):
    """
    Create a synthetic TEM image background (grayscale with texture).

    TEM images have specific characteristics:
    - Relatively uniform background
    - Fine grain texture (noise)
    - Some structural variations
    """
    # Start with base intensity
    background = np.random.normal(180, noise_level, (height, width))

    # Add some large-scale structure (mimics tissue features)
    for _ in range(5):
        y_pos = np.random.randint(0, height)
        x_pos = np.random.randint(0, width)
        radius = np.random.randint(200, 400)

        yy, xx = np.ogrid[:height, :width]
        dist = np.sqrt((xx - x_pos)**2 + (yy - y_pos)**2)
        mask = dist < radius
        background[mask] *= np.random.uniform(0.8, 1.2)

    # Clip to valid range
    background = np.clip(background, 0, 255)
    return background.astype(np.uint8)

def add_gold_particle(image, x, y, size_nm, noise_level=15):
    """
    Add a simulated gold particle to the image.

    Parameters:
    - x, y: center coordinates (in pixels)
    - size_nm: particle size in nanometers (6 or 12)
    - noise_level: add realistic noise
    """
    height, width = image.shape

    # Convert nanometer size to pixel size (6nm ≈ 4-6px, 12nm ≈ 8-12px)
    if size_nm == 6:
        radius_px = np.random.uniform(2, 3)  # 4-6px diameter
        intensity = 230
    else:  # 12nm
        radius_px = np.random.uniform(4, 6)  # 8-12px diameter
        intensity = 240

    # Create particle mask with soft edges (Gaussian)
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - x)**2 + (yy - y)**2)

    # Gaussian blob
    particle = intensity * np.exp(-(dist**2) / (2 * radius_px**2))

    # Add noise to particle for realism
    particle_noise = np.random.normal(0, noise_level, (height, width))
    particle_with_noise = particle + particle_noise

    # Composite onto image (max blend)
    image = np.maximum(image.astype(float), particle_with_noise)

    return image.astype(np.uint8)

def create_synthetic_image(height=2048, width=2048, n_6nm=15, n_12nm=10):
    """
    Create a complete synthetic TEM image with gold particles.

    Returns:
    - image: (H, W) uint8 grayscale image
    - annotations: dict with '6nm' and '12nm' particle coordinates
    """
    # Create TEM background
    image = create_tem_background(height, width)

    # Add 6nm particles
    coords_6nm = []
    for _ in range(n_6nm):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)

        # Avoid overlap
        if not any(np.hypot(x - c[0], y - c[1]) < 30 for c in coords_6nm):
            image = add_gold_particle(image, x, y, size_nm=6)
            coords_6nm.append((x, y))

    # Add 12nm particles
    coords_12nm = []
    for _ in range(n_12nm):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)

        # Avoid overlap with other particles
        all_coords = coords_6nm + coords_12nm
        if not any(np.hypot(x - c[0], y - c[1]) < 40 for c in all_coords):
            image = add_gold_particle(image, x, y, size_nm=12)
            coords_12nm.append((x, y))

    annotations = {
        '6nm': np.array(coords_6nm),
        '12nm': np.array(coords_12nm)
    }

    return image, annotations

def apply_augmentations(image):
    """
    Apply various augmentations to the image for visualization.
    Returns list of (name, augmented_image) tuples.
    """
    augmentations = []

    # Original
    augmentations.append(("Original", image))

    # Brightness variations
    augmentations.append(("Brightness +10%", np.clip(image * 1.1, 0, 255).astype(np.uint8)))
    augmentations.append(("Brightness -10%", np.clip(image * 0.9, 0, 255).astype(np.uint8)))

    # Contrast variations
    mean = image.mean()
    augmentations.append(("High Contrast", np.clip((image - mean) * 1.3 + mean, 0, 255).astype(np.uint8)))

    # Rotations
    augmentations.append(("Rotate 90°", np.rot90(image)))
    augmentations.append(("Rotate 180°", np.rot90(image, 2)))

    # Flips
    augmentations.append(("Flip Horizontal", np.fliplr(image)))
    augmentations.append(("Flip Vertical", np.flipud(image)))

    # Gaussian blur (simulate slightly out-of-focus)
    from scipy.ndimage import gaussian_filter
    augmentations.append(("Blurred (σ=1)", gaussian_filter(image, sigma=1).astype(np.uint8)))

    # Noise
    noisy = np.clip(image.astype(float) + np.random.normal(0, 10, image.shape), 0, 255).astype(np.uint8)
    augmentations.append(("With Noise", noisy))

    return augmentations

def create_heatmap(height, width, annotations, sigma=1.5):
    """
    Create a Gaussian heatmap from particle annotations.
    This is what the model learns to predict.
    """
    heatmap_6nm = np.zeros((height, width), dtype=np.float32)
    heatmap_12nm = np.zeros((height, width), dtype=np.float32)

    yy, xx = np.mgrid[0:height, 0:width]

    # Add Gaussians for 6nm particles
    for x, y in annotations['6nm']:
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        heatmap_6nm = np.maximum(heatmap_6nm, gaussian)

    # Add Gaussians for 12nm particles
    for x, y in annotations['12nm']:
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        heatmap_12nm = np.maximum(heatmap_12nm, gaussian)

    return heatmap_6nm, heatmap_12nm

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TEM images with particles")
    parser.add_argument("--output", default="demo_images/", help="Output directory")
    parser.add_argument("--n-images", type=int, default=3, help="Number of images to generate")
    parser.add_argument("--height", type=int, default=1024, help="Image height (reduced for speed)")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating {args.n_images} synthetic TEM images...")

    for i in range(args.n_images):
        print(f"\n  Image {i+1}/{args.n_images}")

        # Generate image
        image, annotations = create_synthetic_image(
            height=args.height,
            width=args.width,
            n_6nm=np.random.randint(10, 20),
            n_12nm=np.random.randint(8, 15)
        )

        # Save original
        img_path = output_dir / f"synapse_{i:02d}_original.png"
        Image.fromarray(image).save(img_path)
        print(f"    ✓ Saved original: {img_path}")

        # Save heatmaps
        hm_6nm, hm_12nm = create_heatmap(args.height, args.width, annotations)

        hm_6nm_img = Image.fromarray((hm_6nm * 255).astype(np.uint8))
        hm_6nm_path = output_dir / f"synapse_{i:02d}_heatmap_6nm.png"
        hm_6nm_img.save(hm_6nm_path)

        hm_12nm_img = Image.fromarray((hm_12nm * 255).astype(np.uint8))
        hm_12nm_path = output_dir / f"synapse_{i:02d}_heatmap_12nm.png"
        hm_12nm_img.save(hm_12nm_path)
        print(f"    ✓ Saved heatmaps: 6nm, 12nm")

        # Save augmented versions
        augmentations = apply_augmentations(image)
        for aug_name, aug_image in augmentations[1:4]:  # Just save a few for demo
            safe_name = aug_name.lower().replace(' ', '_').replace('+', 'plus').replace('-', 'minus').replace('°', 'deg')
            aug_path = output_dir / f"synapse_{i:02d}_aug_{safe_name}.png"
            Image.fromarray(aug_image).save(aug_path)
        print(f"    ✓ Saved 3 augmented versions")

        # Save annotations
        annot_path = output_dir / f"synapse_{i:02d}_annotations.txt"
        with open(annot_path, 'w') as f:
            for x, y in annotations['6nm']:
                f.write(f"6nm {x:.1f} {y:.1f}\n")
            for x, y in annotations['12nm']:
                f.write(f"12nm {x:.1f} {y:.1f}\n")
        print(f"    ✓ Saved annotations: {len(annotations['6nm'])} 6nm, {len(annotations['12nm'])} 12nm particles")

    print(f"\n✓ Demo data generated in {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - synapse_*.png (original images)")
    print(f"  - synapse_*_heatmap_*.png (model training targets)")
    print(f"  - synapse_*_aug_*.png (augmented versions)")
    print(f"  - synapse_*_annotations.txt (ground truth)")

if __name__ == "__main__":
    main()
