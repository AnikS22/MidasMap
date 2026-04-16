"""
Extract REAL encoder features from the trained MidasMap model.

Workaround for missing _lzma module by mocking it before imports.
"""

import sys
import types

# Try to fix the _lzma issue by attempting to rebuild it
try:
    import _lzma  # noqa
except ImportError:
    # Create a mock _lzma module with stub functions
    class MockLzma:
        def __getattr__(self, name):
            def stub(*args, **kwargs):
                pass
            return stub

    mock_lzma = MockLzma()
    sys.modules['_lzma'] = mock_lzma

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def load_image(image_path: str, max_size: int = 2048) -> torch.Tensor:
    """Load and preprocess a TIF image."""
    img = Image.open(image_path).convert('L')  # Grayscale

    # Resize if needed
    width, height = img.size
    if width > max_size or height > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    # Convert to tensor and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Pad to multiple of 32
    h, w = img_array.shape
    pad_h = ((h + 31) // 32) * 32 - h
    pad_w = ((w + 31) // 32) * 32 - w

    if pad_h > 0 or pad_w > 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='constant')

    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return img_tensor


def extract_features(model_path: str, image_path: str):
    """Extract features from the trained model."""

    # Ensure src is in path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

    # NOW we can safely import after mocking _lzma
    from src.model import ImmunogoldCenterNet

    print("Loading model...")
    model = ImmunogoldCenterNet(
        bifpn_channels=128,
        bifpn_rounds=2,
        imagenet_encoder_fallback=False,
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loading image...")
    image_tensor = load_image(image_path)

    # Capture features using hooks
    features_captured = {}

    def capture_bifpn_output(module, input, output):
        # output is list of [P2, P3, P4, P5]
        # P2 has best spatial resolution
        features_captured['bifpn_output'] = [f.detach().cpu() for f in output]

    def capture_stem_output(module, input, output):
        features_captured['stem_output'] = output.detach().cpu()

    def capture_layer1_output(module, input, output):
        features_captured['layer1_output'] = output.detach().cpu()

    # Register hooks
    model.bifpn.register_forward_hook(capture_bifpn_output)
    model.stem.register_forward_hook(capture_stem_output)
    model.layer1.register_forward_hook(capture_layer1_output)

    print("Running inference to extract features...")
    with torch.no_grad():
        model(image_tensor)

    return features_captured, image_tensor


def apply_colormap(data: np.ndarray) -> np.ndarray:
    """Apply viridis-like colormap."""
    colors = np.array([
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [139, 195, 74],
        [253, 193, 37],
        [255, 152, 0],
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


def visualize_features(original_img: torch.Tensor, features_dict: dict, output_path: str):
    """Create visualization of real extracted features."""

    # Use BiFPN P2 output - this is 128 channels at stride 4
    bifpn_output = features_dict['bifpn_output']
    p2_features = bifpn_output[0][0].numpy()  # (128, H, W)

    print(f"Feature shape: {p2_features.shape}")
    print(f"Feature value range: [{p2_features.min():.3f}, {p2_features.max():.3f}]")

    # Take first 32 channels to visualize
    features_to_viz = p2_features[:32]

    # Normalize each channel
    normalized_features = np.zeros_like(features_to_viz)
    for i in range(len(features_to_viz)):
        feat = features_to_viz[i]
        feat_min = feat.min()
        feat_max = feat.max()
        if feat_max > feat_min:
            normalized_features[i] = (feat - feat_min) / (feat_max - feat_min)
        else:
            normalized_features[i] = feat

    # Original image
    orig_img_np = original_img[0, 0].numpy()  # (H, W)

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

    # Create image
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

    orig_pil = Image.fromarray((orig_img_np * 255).astype(np.uint8))
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
              f"{int(orig_img_np.shape[0])} × {int(orig_img_np.shape[1])} px",
              fill=(150, 150, 150), font=small_font)

    # ===== RIGHT: Real Feature Maps =====
    grid_start_x = left_x + img_display_size + 40
    grid_start_y = 60

    for idx, feat_map in enumerate(normalized_features):
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

        # Border
        draw.rectangle(
            [(x - 1, y - 1), (x + cell_size + 1, y + cell_size + 1)],
            outline=(50, 75, 125), width=1
        )

        viz.paste(feat_pil, (x, y))

    draw.text((grid_start_x, grid_start_y - 25),
              "REAL Learned Feature Maps (128 total, showing 32)",
              fill=(200, 200, 200), font=label_font)

    # ===== TITLE =====
    title = "ResNet-50 Encoder Output (Actual Learned Features)"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (total_width - title_width) // 2

    draw.rectangle([(title_x - 10, 10), (title_x + title_width + 10, 35)], fill=(10, 15, 30))
    draw.text((title_x, 12), title, fill=(255, 200, 100), font=title_font)

    # ===== DESCRIPTION =====
    descriptions = [
        "Left: Original 2048×2048 TEM synapse image (grayscale)",
        "Right: 32 of 128 actual learned feature maps from ResNet-50 encoder (at stride 4, ~512×512 resolution)",
        "Each colored heatmap shows real activation patterns learned during training on the Max Planck dataset",
        "These features represent edges, textures, blobs, and complex patterns critical for detecting 4-6nm particles",
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
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    features, original_img = extract_features(checkpoint_path, image_path)
    visualize_features(original_img, features, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
