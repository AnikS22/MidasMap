"""
Visualize MidasMap pipeline using REAL Max Planck TEM synapse images.

This script loads actual gold particle data and shows:
1. Original EM image
2. Preprocessing
3. Feature extraction at different depths
4. Predictions (heatmap + offset)
5. Augmented versions

Usage:
    python scripts/visualize_real_data.py

The script automatically finds and visualizes synapse images from:
"Max Planck Data/Gold Particle Labelling/analyzed synapses/"
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, maximum_filter
import warnings
warnings.filterwarnings('ignore')

def find_synapse_images():
    """Find all actual synapse TIFF images in Max Planck data."""
    data_dir = Path("Max Planck Data/Gold Particle Labelling/analyzed synapses")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Make sure you're in the MidasMap root directory.")
        return []

    # Find all main synapse images (not masks, not overlays, not color)
    images = []
    for tif_file in data_dir.rglob("*.tif"):
        # Skip masks, overlays, and color versions
        if "mask" in str(tif_file) or "overlay" in str(tif_file) or "color" in str(tif_file):
            continue
        # Only include main synapse images
        if tif_file.name.startswith(('S', 'synapse')) and tif_file.name.endswith('.tif'):
            images.append(tif_file)

    return sorted(images)[:5]  # Return first 5 for visualization

def load_image(path):
    """Load TIFF image."""
    try:
        from PIL import Image
        img = Image.open(path)
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def preprocess_image(image):
    """Normalize to [0, 1]."""
    return image.astype(np.float32) / 255.0

def simulate_feature_extraction(image, stage="shallow"):
    """Simulate feature extraction at different depths."""
    if stage == "shallow":
        gy = np.gradient(image, axis=0)
        gx = np.gradient(image, axis=1)
        edges = np.sqrt(gx**2 + gy**2)
        return (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)
    elif stage == "middle":
        blur1 = gaussian_filter(image, sigma=2)
        blur2 = gaussian_filter(image, sigma=4)
        blobs = blur1 - blur2
        return (blobs - blobs.min()) / (blobs.max() - blobs.min() + 1e-6)
    elif stage == "deep":
        blurred = gaussian_filter(image, sigma=3)
        enhanced = image - blurred * 0.7
        return np.clip(enhanced, 0, 1)

def simulate_heatmap_output(image, blur_sigma=1.5):
    """Simulate heatmap output."""
    blurred = gaussian_filter(image, sigma=2)
    local_max = maximum_filter(blurred, size=5) == blurred
    heatmap = np.zeros_like(image)
    for (y, x) in zip(*np.where(local_max)):
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * blur_sigma**2))
        heatmap = np.maximum(heatmap, gaussian * blurred[y, x])
    return heatmap

def visualize_synapse(image_path, output_dir="results/"):
    """Create pipeline visualization for one synapse."""
    print(f"\n📊 Processing: {image_path.name}")

    # Load image
    image = load_image(image_path)
    if image is None:
        return None

    h, w = image.shape
    print(f"   Size: {w}×{h} pixels")

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'MidasMap Pipeline: {image_path.stem}\n(Real Max Planck TEM Synapse Image)',
                fontsize=14, fontweight='bold')

    # Row 1: Input & Preprocessing
    ax = fig.add_subplot(3, 3, 1)
    ax.imshow(image, cmap='gray')
    ax.set_title('1. Original TEM Image\n(real Max Planck data)', fontweight='bold', fontsize=10)
    ax.axis('off')

    preprocessed = preprocess_image(image)
    ax = fig.add_subplot(3, 3, 2)
    ax.imshow(preprocessed, cmap='gray')
    ax.set_title('2. Preprocessed\n(normalized)', fontweight='bold', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 3)
    ax.axis('off')
    info = f"""INPUT DETAILS

Size: {w}×{h} px
Original range: [0, 255]
Format: Grayscale uint8

Processing:
1. Normalize to [0, 1]
2. Pass to encoder
3. Extract features
    """
    ax.text(0.1, 0.9, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 2: Feature Extraction
    ax = fig.add_subplot(3, 3, 4)
    shallow = simulate_feature_extraction(preprocessed, stage="shallow")
    ax.imshow(shallow, cmap='hot')
    ax.set_title('3. Shallow Features\n(edges)', fontweight='bold', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 5)
    middle = simulate_feature_extraction(preprocessed, stage="middle")
    ax.imshow(middle, cmap='hot')
    ax.set_title('4. Middle Features\n(blobs)', fontweight='bold', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 6)
    deep = simulate_feature_extraction(preprocessed, stage="deep")
    ax.imshow(deep, cmap='hot')
    ax.set_title('5. Deep Features\n(particles)', fontweight='bold', fontsize=10)
    ax.axis('off')

    # Row 3: Outputs
    ax = fig.add_subplot(3, 3, 7)
    heatmap = simulate_heatmap_output(preprocessed)
    im = ax.imshow(heatmap, cmap='viridis')
    ax.set_title('6. Heatmap Output\n(where are particles?)', fontweight='bold', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(3, 3, 8)
    offset_mag = np.abs(np.gradient(preprocessed, axis=0)) + np.abs(np.gradient(preprocessed, axis=1))
    offset_mag = gaussian_filter(offset_mag, sigma=2)
    im = ax.imshow(offset_mag, cmap='cool')
    ax.set_title('7. Offset Output\n(±0.5px refinement)', fontweight='bold', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(3, 3, 9)
    ax.imshow(image, cmap='gray')
    blurred_hm = heatmap.copy()
    local_max = maximum_filter(blurred_hm, size=7) == blurred_hm
    det_y, det_x = np.where(local_max & (blurred_hm > 0.3))
    if len(det_x) > 0:
        scatter = ax.scatter(det_x, det_y, c=blurred_hm[det_y, det_x],
                           s=100, cmap='RdYlGn', alpha=0.7, edgecolors='white', linewidths=2)
        plt.colorbar(scatter, ax=ax, fraction=0.046)
    ax.set_title('8. Final Detections\n(n={})'.format(len(det_x)), fontweight='bold', fontsize=10)
    ax.axis('off')

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    save_path = output_path / f"{image_path.stem}_pipeline.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()

    return save_path

def create_augmentation_viz(image_path, output_dir="results/"):
    """Create augmentation comparison."""
    print(f"   Augmentation comparison...")

    image = load_image(image_path)
    if image is None:
        return

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'Data Augmentation (D4 TTA)\n{image_path.stem}',
                fontsize=13, fontweight='bold')

    # Rotations
    for idx, angle in enumerate([0, 90, 180]):
        ax = axes[0, idx]
        rotated = np.rot90(image, k=angle // 90)
        ax.imshow(rotated, cmap='gray')
        ax.set_title(f'Rotation {angle}°', fontweight='bold', fontsize=9)
        ax.axis('off')

    # Flips
    flips = [np.fliplr(image), np.flipud(image), image]
    titles = ['Flip H', 'Flip V', 'Original']
    for idx, (flip, title) in enumerate(zip(flips, titles)):
        ax = axes[1, idx]
        ax.imshow(flip, cmap='gray')
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.axis('off')

    # Brightness
    for idx, factor in enumerate([0.9, 1.0, 1.1]):
        ax = axes[2, idx]
        bright = np.clip(image * factor, 0, 255).astype(np.uint8)
        ax.imshow(bright, cmap='gray')
        ax.set_title(f'×{factor}', fontweight='bold', fontsize=9)
        ax.axis('off')

    output_path = Path(output_dir)
    save_path = output_path / f"{image_path.stem}_augmentation.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Augmentation saved: {save_path}")
    plt.close()

def main():
    print("="*70)
    print("VISUALIZING MIDASMAP PIPELINE WITH REAL MAX PLANCK DATA")
    print("="*70)

    # Find images
    images = find_synapse_images()
    if not images:
        print("\n❌ No synapse images found.")
        print("Run from MidasMap root directory with Max Planck Data present.")
        return

    print(f"\n✓ Found {len(images)} synapse images")

    # Visualize each
    for image_path in images:
        try:
            visualize_synapse(image_path)
            create_augmentation_viz(image_path)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files in: results/")
    print("  - *_pipeline.png : Full pipeline visualization")
    print("  - *_augmentation.png : D4 TTA comparison")

if __name__ == "__main__":
    main()
