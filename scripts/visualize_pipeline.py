"""
Visualize the MidasMap pipeline stages: what the model "sees" at each step.

Creates a comprehensive figure showing:
1. Original image
2. Preprocessed image
3. Feature maps at different stages
4. Heatmap predictions
5. Offset predictions
6. Final detections

Usage:
    # Use REAL Max Planck data
    python scripts/visualize_pipeline.py --image "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"

    # Or with demo data:
    python scripts/generate_demo_data.py --output demo_images/
    python scripts/visualize_pipeline.py --image demo_images/synapse_00_original.png
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

def load_image(path):
    """Load image (handles PNG, TIFF, JPG)"""
    from PIL import Image
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')  # Convert to grayscale
    return np.array(img, dtype=np.uint8)

def preprocess_image(image):
    """
    Preprocessing step: normalize to [0, 1].
    This is what the neural network actually sees.
    """
    return image.astype(np.float32) / 255.0

def simulate_feature_extraction(image, stage="shallow"):
    """
    Simulate what features the network extracts.
    In reality, this is a ResNet-50 encoder. Here we approximate it.

    Stages:
    - shallow: Early layers detect edges, corners
    - middle: Mid layers detect blobs, textures
    - deep: Deep layers detect particles vs background
    """
    if stage == "shallow":
        # Simulate edge detection (Sobel-like)
        gy = np.gradient(image, axis=0)
        gx = np.gradient(image, axis=1)
        edges = np.sqrt(gx**2 + gy**2)
        # Normalize
        return (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)

    elif stage == "middle":
        # Simulate blob detection (difference of Gaussians)
        from scipy.ndimage import gaussian_filter
        blur1 = gaussian_filter(image, sigma=2)
        blur2 = gaussian_filter(image, sigma=4)
        blobs = blur1 - blur2
        # Normalize
        return (blobs - blobs.min()) / (blobs.max() - blobs.min() + 1e-6)

    elif stage == "deep":
        # Simulate particle detection (bright spots)
        from scipy.ndimage import gaussian_filter
        # Enhance bright spots
        blurred = gaussian_filter(image, sigma=3)
        # Find peaks
        enhanced = image - blurred * 0.7
        return np.clip(enhanced, 0, 1)

def simulate_heatmap_output(image, blur_sigma=1.5):
    """
    Simulate the heatmap head output.
    This shows where the model thinks particles are.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    # Find potential particle locations (local maxima)
    blurred = gaussian_filter(image, sigma=2)
    local_max = maximum_filter(blurred, size=5) == blurred

    # Create Gaussian heatmap at detected locations
    heatmap = np.zeros_like(image)
    for (y, x) in zip(*np.where(local_max)):
        # Add Gaussian at this location
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * blur_sigma**2))
        heatmap = np.maximum(heatmap, gaussian * blurred[y, x])

    return heatmap

def simulate_offset_output(image):
    """
    Simulate the offset head output.
    This shows sub-pixel refinement vectors.
    """
    from scipy.ndimage import gaussian_filter

    # The offset is basically the gradient pointing toward particle centers
    gy = np.gradient(image, axis=0)
    gx = np.gradient(image, axis=1)

    # Smooth it
    gy_smooth = gaussian_filter(gy, sigma=2)
    gx_smooth = gaussian_filter(gx, sigma=2)

    return gx_smooth, gy_smooth

def create_pipeline_visualization(image_path, output_path=None, annotations_path=None):
    """
    Create a multi-panel figure showing the pipeline.
    """
    # Load image
    image = load_image(image_path)
    h, w = image.shape

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ===== ROW 1: Input & Preprocessing =====

    # [1,1] Original image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap='gray')
    ax.set_title('1. Original Image\n(as captured by microscope)', fontweight='bold', fontsize=10)
    ax.axis('off')

    # [1,2] Preprocessed
    ax = fig.add_subplot(gs[0, 1])
    preprocessed = preprocess_image(image)
    ax.imshow(preprocessed, cmap='gray')
    ax.set_title('2. Preprocessed\n(normalized to [0,1])', fontweight='bold', fontsize=10)
    ax.axis('off')

    # [1,3] Info
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    info_text = f"""INPUT PREPARATION

Image size: {w} × {h} px
Format: Grayscale uint8
Range: [0, 255]

Process:
1. Normalize to [0, 1]
2. Pad to multiple of 32
3. Feed to encoder
    """
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ===== ROW 2: Feature Extraction =====

    # [2,1] Shallow features
    ax = fig.add_subplot(gs[1, 0])
    shallow = simulate_feature_extraction(image, stage="shallow")
    ax.imshow(shallow, cmap='hot')
    ax.set_title('3. Shallow Features\n(edges, corners)', fontweight='bold', fontsize=10)
    ax.axis('off')

    # [2,2] Middle features
    ax = fig.add_subplot(gs[1, 1])
    middle = simulate_feature_extraction(image, stage="middle")
    ax.imshow(middle, cmap='hot')
    ax.set_title('4. Middle Features\n(blobs, textures)', fontweight='bold', fontsize=10)
    ax.axis('off')

    # [2,3] Deep features
    ax = fig.add_subplot(gs[1, 2])
    deep = simulate_feature_extraction(image, stage="deep")
    ax.imshow(deep, cmap='hot')
    ax.set_title('5. Deep Features\n(particle likelihood)', fontweight='bold', fontsize=10)
    ax.axis('off')

    # ===== ROW 3: Model Outputs =====

    # [3,1] Heatmap output
    ax = fig.add_subplot(gs[2, 0])
    heatmap = simulate_heatmap_output(preprocessed)
    im = ax.imshow(heatmap, cmap='viridis')
    ax.set_title('6. Heatmap Output\n(where are particles?)', fontweight='bold', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # [3,2] Offset output (visualization)
    ax = fig.add_subplot(gs[2, 1])
    offset_x, offset_y = simulate_offset_output(preprocessed)
    offset_magnitude = np.sqrt(offset_x**2 + offset_y**2)
    im = ax.imshow(offset_magnitude, cmap='cool')
    ax.set_title('7. Offset Output\n(refine to ±0.5px)', fontweight='bold', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # [3,3] Final detections
    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(image, cmap='gray')

    # Simulate final detections
    from scipy.ndimage import maximum_filter
    blurred_hm = heatmap.copy()
    local_max = maximum_filter(blurred_hm, size=7) == blurred_hm
    det_y, det_x = np.where(local_max & (blurred_hm > 0.3))

    # Color detections by confidence
    scatter = ax.scatter(det_x, det_y, c=blurred_hm[det_y, det_x],
                        s=100, cmap='RdYlGn', alpha=0.7, edgecolors='white', linewidths=2)
    ax.set_title('8. Final Detections\n(color = confidence)', fontweight='bold', fontsize=10)
    ax.axis('off')
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    # Main title
    fig.suptitle('MidasMap Pipeline: What the Neural Network "Sees"',
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    if output_path is None:
        output_path = Path(image_path).stem + "_pipeline_viz.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved pipeline visualization: {output_path}")

    return fig

def create_augmentation_comparison(image_path, output_path=None):
    """
    Show the same image with different augmentations.
    This demonstrates why D4 TTA works.
    """
    image = load_image(image_path)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Test-Time Augmentation (D4 TTA)\nWhy averaging 10 views improves predictions',
                fontsize=13, fontweight='bold')

    # Rotation: 0, 90, 180, 270
    rotations = [0, 90, 180, 270]
    for idx, angle in enumerate(rotations):
        ax = axes[0, idx % 3]
        rotated = np.rot90(image, k=angle // 90)
        ax.imshow(rotated, cmap='gray')
        ax.set_title(f'Rotation {angle}°', fontweight='bold')
        ax.axis('off')

    if len(rotations) < 3:
        axes[0, 2].axis('off')

    # Flip + Rotate
    for idx in range(3):
        ax = axes[1, idx]
        if idx == 0:
            flipped = np.fliplr(image)
            ax.imshow(flipped, cmap='gray')
            ax.set_title('Flip Horizontal', fontweight='bold')
        elif idx == 1:
            flipped = np.flipud(image)
            ax.imshow(flipped, cmap='gray')
            ax.set_title('Flip Vertical', fontweight='bold')
        else:
            # Original for comparison
            ax.imshow(image, cmap='gray')
            ax.set_title('Original (reference)', fontweight='bold')
        ax.axis('off')

    # Brightness variations
    for idx, factor in enumerate([0.9, 1.0, 1.1]):
        ax = axes[2, idx]
        bright = np.clip(image * factor, 0, 255).astype(np.uint8)
        ax.imshow(bright, cmap='gray')
        ax.set_title(f'Brightness ×{factor}', fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    if output_path is None:
        output_path = Path(image_path).stem + "_augmentation_viz.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved augmentation visualization: {output_path}")

    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize MidasMap pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--augmentation", action="store_true", help="Also create augmentation visualization")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"Visualizing pipeline for: {image_path}")

    # Create pipeline visualization
    create_pipeline_visualization(str(image_path), output_path=args.output)

    # Create augmentation visualization if requested
    if args.augmentation:
        create_augmentation_comparison(str(image_path))

    print("\n✓ Visualization complete!")

if __name__ == "__main__":
    main()
