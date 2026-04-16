"""
Extract REAL encoder features from the trained MidasMap model.
Run with: conda activate immunogold && python scripts/extract_real_encoder_features.py
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.model import ImmunogoldCenterNet


def load_image(image_path: str) -> tuple:
    """Load TIF image and return as tensor."""
    img = Image.open(image_path).convert('L')

    width, height = img.size
    print(f"Original image size: {width} × {height}")

    # Resize if too large
    if width > 2048 or height > 2048:
        img.thumbnail((2048, 2048), Image.LANCZOS)

    # Convert to array
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Pad to multiple of 32
    h, w = img_array.shape
    pad_h = ((h + 31) // 32) * 32 - h
    pad_w = ((w + 31) // 32) * 32 - w

    if pad_h > 0 or pad_w > 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='constant')

    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor, img_array


def extract_features(checkpoint_path: str, image_tensor: torch.Tensor):
    """Load model and extract intermediate features."""

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print("Building model...")
    model = ImmunogoldCenterNet(
        bifpn_channels=128,
        bifpn_rounds=2,
        imagenet_encoder_fallback=False,
    )

    print("Loading weights...")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Capture features from layer1 (p2)
    features_dict = {}

    def hook_layer1(module, input, output):
        features_dict['layer1'] = output.detach().cpu()

    def hook_bifpn(module, input, output):
        features_dict['bifpn'] = [f.detach().cpu() for f in output]

    # Register hooks
    model.layer1.register_forward_hook(hook_layer1)
    model.bifpn.register_forward_hook(hook_bifpn)

    print("Running inference...")
    with torch.no_grad():
        model(image_tensor)

    return features_dict


def apply_colormap(data: np.ndarray) -> np.ndarray:
    """Apply viridis-like colormap."""
    colors = np.array([
        [68, 1, 84],      # Purple
        [59, 82, 139],    # Blue
        [33, 145, 140],   # Teal
        [139, 195, 74],   # Green
        [253, 193, 37],   # Yellow
        [255, 152, 0],    # Orange
    ])

    indices = np.clip(data * (len(colors) - 1), 0, len(colors) - 1.0001)
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


def create_visualization(original_img: np.ndarray, features_dict: dict, output_path: str):
    """Create visualization of real features."""

    # Use layer1 output - 256 channels at stride 4
    layer1_features = features_dict['layer1'][0].numpy()  # (256, H, W)
    print(f"Layer1 feature shape: {layer1_features.shape}")
    print(f"Layer1 feature range: [{layer1_features.min():.3f}, {layer1_features.max():.3f}]")

    # Also use BiFPN P2 for comparison
    bifpn_p2 = features_dict['bifpn'][0][0].numpy()  # (128, H, W)
    print(f"BiFPN P2 feature shape: {bifpn_p2.shape}")
    print(f"BiFPN P2 feature range: [{bifpn_p2.min():.3f}, {bifpn_p2.max():.3f}]")

    # Use layer1 (256 channels, more diverse)
    features_to_viz = layer1_features[:32]

    # Normalize each channel independently
    normalized = np.zeros_like(features_to_viz)
    for i in range(len(features_to_viz)):
        feat = features_to_viz[i]
        feat_min = feat.min()
        feat_max = feat.max()
        if feat_max > feat_min:
            normalized[i] = (feat - feat_min) / (feat_max - feat_min)
        else:
            normalized[i] = 0.5  # Gray for empty features

    # Create layout
    cell_size = 96
    grid_cols = 8
    grid_rows = 4
    padding = 4

    grid_width = grid_cols * (cell_size + padding) + padding
    grid_height = grid_rows * (cell_size + padding) + padding

    img_display_size = 384
    left_section_width = img_display_size + 40

    total_width = left_section_width + grid_width + 60
    total_height = max(img_display_size + 100, grid_height + 100) + 120

    print("Creating visualization...")
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

    # ===== LEFT: Original Image =====
    left_x = 20
    left_y = 60

    orig_pil = Image.fromarray((original_img * 255).astype(np.uint8))
    orig_resized = orig_pil.resize((img_display_size, img_display_size), Image.LANCZOS)

    box_x = left_x - 2
    box_y = left_y - 2
    draw.rectangle(
        [(box_x, box_y), (box_x + img_display_size + 4, box_y + img_display_size + 4)],
        outline=(100, 150, 255), width=2
    )
    viz.paste(orig_resized, (left_x, left_y))

    draw.text((left_x, left_y - 25), "Original TEM Image", fill=(200, 200, 200), font=label_font)
    draw.text((left_x, box_y + img_display_size + 10),
              f"{int(original_img.shape[0])} × {int(original_img.shape[1])} px",
              fill=(150, 150, 150), font=small_font)

    # ===== RIGHT: Feature Maps Grid =====
    grid_start_x = left_x + img_display_size + 40
    grid_start_y = 60

    for idx, feat_map in enumerate(normalized):
        row = idx // grid_cols
        col = idx % grid_cols

        x = grid_start_x + col * (cell_size + padding)
        y = grid_start_y + row * (cell_size + padding)

        # Resize feature map to cell size
        feat_pil = Image.fromarray((feat_map * 255).astype(np.uint8))
        feat_resized = feat_pil.resize((cell_size, cell_size), Image.BILINEAR)

        # Apply colormap
        feat_rgb = apply_colormap(np.array(feat_resized) / 255.0)
        feat_color = Image.fromarray(feat_rgb)

        # Border
        draw.rectangle(
            [(x - 1, y - 1), (x + cell_size + 1, y + cell_size + 1)],
            outline=(50, 75, 125), width=1
        )

        viz.paste(feat_color, (x, y))

    draw.text((grid_start_x, grid_start_y - 25),
              "REAL Layer1 Features (256 total, showing 32)",
              fill=(200, 200, 200), font=label_font)

    # ===== TITLE =====
    title = "ResNet-50 Encoder Output (Actual Learned Features)"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = max(10, (total_width - title_width) // 2)

    draw.rectangle([(title_x - 10, 10), (title_x + title_width + 10, 35)], fill=(10, 15, 30))
    draw.text((title_x, 12), title, fill=(255, 200, 100), font=title_font)

    # ===== DESCRIPTION =====
    descriptions = [
        "Left: Original 2048×2048 TEM synapse image",
        "Right: 32 of 256 REAL learned feature maps from ResNet-50 Layer1 (stride 4, ~512×512)",
        "Each feature has learned a unique pattern from training on Max Planck immunogold particles",
        "These patterns detect edges, corners, textures, blobs - all critical for finding 4-6nm particles",
    ]

    desc_y = total_height - 90
    for i, desc in enumerate(descriptions):
        draw.text((20, desc_y + i * 18), desc, fill=(150, 170, 190), font=small_font)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    viz.save(output_path, dpi=(300, 300))
    print(f"✓ Saved to {output_path}")


def main():
    checkpoint_path = "checkpoints/final/final_model.pth"
    image_path = "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"
    output_path = "results/diagrams/11_encoder_features.png"

    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path}")
        return

    if not os.path.exists(image_path):
        print(f"Error: {image_path}")
        return

    print("Loading image...")
    image_tensor, original_img = load_image(image_path)

    print("Extracting features...")
    features_dict = extract_features(checkpoint_path, image_tensor)

    print("Visualizing...")
    create_visualization(original_img, features_dict, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
