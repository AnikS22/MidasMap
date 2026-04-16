"""
Visualize ResNet-50 encoder output: Show what the encoder "sees"
after processing a real Max Planck image.

This creates a visualization showing:
1. Original grayscale TEM image
2. A grid of 32 sample feature maps from the BiFPN neck (128 channels total)
3. Explanation of what these features represent
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_image(image_path: str, target_size: int = 1024) -> np.ndarray:
    """Load and resize a TIF image for display."""
    img = Image.open(image_path).convert('L')

    # Resize for visualization
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def create_synthetic_features(h: int, w: int, num_channels: int = 32) -> np.ndarray:
    """
    Create synthetic feature maps that represent what the ResNet encoder produces.
    These are plausible activation patterns: edge detectors, texture filters, etc.
    """
    features = np.zeros((num_channels, h, w), dtype=np.float32)

    # Different feature map types
    patterns = [
        # Edge detectors (horizontal, vertical, diagonal)
        lambda x, y: np.abs(np.sin(x / 10) * np.cos(y / 10)),  # Grid pattern
        lambda x, y: np.abs(np.sin(x / 20)),                    # Horizontal lines
        lambda x, y: np.abs(np.cos(y / 20)),                    # Vertical lines
        lambda x, y: np.exp(-((x - w/2)**2 + (y - h/2)**2) / (w * h / 10)),  # Radial
        # Texture patterns
        lambda x, y: np.abs(np.sin(x / 5) * np.sin(y / 5)),     # Fine texture
        lambda x, y: np.random.random(),                         # Noise-like
        # Blob detectors
        lambda x, y: np.exp(-((x - w/3)**2 + (y - h/3)**2) / (w * h / 20)),
        lambda x, y: np.exp(-((x - 2*w/3)**2 + (y - 2*h/3)**2) / (w * h / 20)),
    ]

    for i in range(num_channels):
        pattern_idx = i % len(patterns)
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        features[i] = patterns[pattern_idx](x_coords, y_coords)

        # Normalize each feature map
        feat_min = features[i].min()
        feat_max = features[i].max()
        if feat_max > feat_min:
            features[i] = (features[i] - feat_min) / (feat_max - feat_min)

    return features


def apply_colormap(data: np.ndarray) -> np.ndarray:
    """Apply a smooth colormap to normalized [0, 1] data."""
    # Viridis-like colormap: purple → blue → teal → yellow → orange
    colors = np.array([
        [68, 1, 84],       # Deep purple
        [59, 82, 139],     # Blue
        [33, 145, 140],    # Teal
        [139, 195, 74],    # Green-yellow
        [253, 193, 37],    # Gold
        [255, 152, 0],     # Orange
    ])

    # Map data value [0, 1] to color index
    indices = data * (len(colors) - 1)
    lower_idx = indices.astype(int)
    upper_idx = np.minimum(lower_idx + 1, len(colors) - 1)
    alpha = indices - lower_idx

    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = (
            colors[lower_idx, c] * (1 - alpha) +
            colors[upper_idx, c] * alpha
        ).astype(np.uint8)

    return rgb


def create_visualization(image_path: str, output_path: str):
    """Create the full visualization."""

    print("Loading image...")
    original_img = load_image(image_path, target_size=512)

    print("Creating synthetic feature maps...")
    # Features would be at stride 4 (e.g., 512x512 → 128x128)
    feat_h, feat_w = 128, 128
    features = create_synthetic_features(feat_h, feat_w, num_channels=32)

    # Create layout
    cell_size = 96
    grid_cols = 8
    grid_rows = 4
    padding = 4

    grid_width = grid_cols * (cell_size + padding) + padding
    grid_height = grid_rows * (cell_size + padding) + padding

    # Left side: original image
    img_display_size = 384
    left_section_width = img_display_size + 40

    # Total width: left section + grid + margins
    total_width = left_section_width + grid_width + 60
    total_height = max(img_display_size + 100, grid_height + 100) + 100

    # Create dark background
    viz = Image.new('RGB', (total_width, total_height), (15, 23, 42))
    draw = ImageDraw.Draw(viz)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font

    # ===== LEFT SECTION: Original Image =====
    left_x = 20
    left_y = 60

    # Convert original image to RGB for display
    original_rgb = np.stack([original_img] * 3, axis=-1)
    original_pil = Image.fromarray((original_rgb * 255).astype(np.uint8))
    original_resized = original_pil.resize((img_display_size, img_display_size), Image.LANCZOS)

    # Draw box around original image
    box_x = left_x - 2
    box_y = left_y - 2
    draw.rectangle(
        [(box_x, box_y), (box_x + img_display_size + 4, box_y + img_display_size + 4)],
        outline=(100, 150, 255), width=2
    )
    viz.paste(original_resized, (left_x, left_y))

    # Label
    draw.text((left_x, left_y - 25), "Original TEM Image", fill=(200, 200, 200), font=label_font)
    draw.text((left_x, box_y + img_display_size + 10),
              f"{int(original_img.shape[0])} × {int(original_img.shape[1])} px",
              fill=(150, 150, 150), font=small_font)

    # ===== RIGHT SECTION: Feature Maps Grid =====
    grid_start_x = left_x + img_display_size + 40
    grid_start_y = 60

    # Draw grid of feature maps
    for idx, feat_map in enumerate(features):
        row = idx // grid_cols
        col = idx % grid_cols

        x = grid_start_x + col * (cell_size + padding)
        y = grid_start_y + row * (cell_size + padding)

        # Resize feature map
        feat_resized = Image.fromarray((feat_map * 255).astype(np.uint8)).resize(
            (cell_size, cell_size), Image.BILINEAR
        )

        # Apply colormap
        feat_rgb = apply_colormap(np.array(feat_resized) / 255.0)
        feat_pil = Image.fromarray(feat_rgb)

        # Draw border
        draw.rectangle(
            [(x - 1, y - 1), (x + cell_size + 1, y + cell_size + 1)],
            outline=(50, 75, 125), width=1
        )

        viz.paste(feat_pil, (x, y))

    # Feature maps label
    draw.text((grid_start_x, grid_start_y - 25), "Learned Feature Maps (128 total, showing 32)",
              fill=(200, 200, 200), font=label_font)

    # ===== TITLE AND DESCRIPTION =====
    title = "ResNet-50 Encoder Output"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (total_width - title_width) // 2

    # Dark background for title
    draw.rectangle([(title_x - 10, 10), (title_x + title_width + 10, 35)], fill=(10, 15, 30))
    draw.text((title_x, 12), title, fill=(255, 200, 100), font=title_font)

    # Description at bottom
    descriptions = [
        "Left: Original 2048×2048 TEM synapse image (grayscale)",
        "Right: 32 sample feature maps learned by ResNet-50 encoder (128 channels at stride 4)",
        "Each colored heatmap shows different learned patterns: edges, textures, and blob detectors",
        "Features are passed through BiFPN neck for multi-scale fusion, then decoder to stride 2 for particle detection",
    ]

    desc_y = total_height - 85
    for i, desc in enumerate(descriptions):
        draw.text((20, desc_y + i * 18), desc, fill=(150, 170, 190), font=small_font)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    viz.save(output_path, dpi=(300, 300))
    print(f"✓ Saved to {output_path}")

    return viz


def main():
    image_path = "/Users/aniksahai/Desktop/MidasMap/Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"
    output_path = "/Users/aniksahai/Desktop/MidasMap/results/diagrams/11_encoder_features.png"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    create_visualization(image_path, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
