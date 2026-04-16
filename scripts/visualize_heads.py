"""
Visualize Detection Heads: Heatmap and Offset
Shows how decoder features are transformed into particle predictions and sub-pixel refinement.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.model import ImmunogoldCenterNet


def load_image(image_path: str) -> tuple:
    """Load TIF image."""
    img = Image.open(image_path).convert('L')
    width, height = img.size

    if width > 2048 or height > 2048:
        img.thumbnail((2048, 2048), Image.LANCZOS)

    img_array = np.array(img, dtype=np.float32) / 255.0

    h, w = img_array.shape
    pad_h = ((h + 31) // 32) * 32 - h
    pad_w = ((w + 31) // 32) * 32 - w

    if pad_h > 0 or pad_w > 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='constant')

    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor, img_array


def extract_head_features(checkpoint_path: str, image_tensor: torch.Tensor):
    """Extract features from all stages including heads."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2, imagenet_encoder_fallback=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features_dict = {}

    def hook_decoder(module, input, output):
        features_dict['decoder_output'] = output.detach().cpu()

    def hook_heatmap(module, input, output):
        features_dict['heatmap_output'] = output.detach().cpu()

    def hook_offset(module, input, output):
        features_dict['offset_output'] = output.detach().cpu()

    model.upsample.register_forward_hook(hook_decoder)
    model.heatmap_head.register_forward_hook(hook_heatmap)
    model.offset_head.register_forward_hook(hook_offset)

    with torch.no_grad():
        model(image_tensor)

    return features_dict


def apply_colormap_hot(data: np.ndarray) -> np.ndarray:
    """Apply hot colormap for heatmap (black → red → yellow → white)."""
    colors = np.array([
        [0, 0, 0],
        [50, 0, 50],
        [128, 0, 0],
        [255, 0, 0],
        [255, 100, 0],
        [255, 200, 0],
        [255, 255, 150],
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


def apply_colormap_diverging(data: np.ndarray) -> np.ndarray:
    """Apply diverging colormap for offset (blue ← 0 → red)."""
    # Maps [-1, 1] range to blue → white → red
    colors_neg = np.array([
        [0, 0, 255],      # Blue
        [100, 150, 255],  # Light blue
        [200, 200, 255],  # Very light blue
    ])
    colors_pos = np.array([
        [255, 200, 200],  # Very light red
        [255, 100, 100],  # Light red
        [255, 0, 0],      # Red
    ])

    # Normalize data from any range to [-1, 1]
    if data.max() > -data.min():
        norm_data = 2 * (data / data.max()) - 1 if data.max() > 0 else data
    else:
        norm_data = -2 * (data / -data.min()) + 1 if data.min() < 0 else data
    norm_data = np.clip(norm_data, -1, 1)

    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)

    # Negative values (blue side)
    neg_mask = norm_data < 0
    if neg_mask.any():
        neg_indices = np.clip(-norm_data[neg_mask] * (len(colors_neg) - 1), 0, len(colors_neg) - 1.0001)
        lower_idx = neg_indices.astype(int)
        upper_idx = np.minimum(lower_idx + 1, len(colors_neg) - 1)
        alpha = neg_indices - lower_idx

        for c in range(3):
            rgb[neg_mask, c] = (
                colors_neg[lower_idx, c] * (1 - alpha) +
                colors_neg[upper_idx, c] * alpha
            ).astype(np.uint8)

    # Positive values (red side)
    pos_mask = norm_data > 0
    if pos_mask.any():
        pos_indices = np.clip(norm_data[pos_mask] * (len(colors_pos) - 1), 0, len(colors_pos) - 1.0001)
        lower_idx = pos_indices.astype(int)
        upper_idx = np.minimum(lower_idx + 1, len(colors_pos) - 1)
        alpha = pos_indices - lower_idx

        for c in range(3):
            rgb[pos_mask, c] = (
                colors_pos[lower_idx, c] * (1 - alpha) +
                colors_pos[upper_idx, c] * alpha
            ).astype(np.uint8)

    return rgb


def normalize(x):
    """Normalize to [0, 1]."""
    if x.max() > x.min():
        return (x - x.min()) / (x.max() - x.min())
    return np.zeros_like(x)


def visualize_heads(features_dict: dict, output_path: str):
    """Create heads visualization."""

    print("Processing head features...")

    # Decoder output
    decoder_out = features_dict['decoder_output'][0].numpy()  # (64, H, W)
    decoder_max = normalize(np.max(decoder_out, axis=0))

    # Heatmap output (2 channels: 6nm, 12nm)
    heatmap = features_dict['heatmap_output'][0].numpy()  # (2, H, W)
    heatmap_6nm = heatmap[0]  # Confidence for 6nm particles
    heatmap_12nm = heatmap[1]  # Confidence for 12nm particles
    heatmap_combined = np.maximum(heatmap_6nm, heatmap_12nm)

    # Offset output (2 channels: dx, dy)
    offset = features_dict['offset_output'][0].numpy()  # (2, H, W)
    offset_x = offset[0]  # X offset
    offset_y = offset[1]  # Y offset

    print(f"Decoder output: {decoder_out.shape}, range [{decoder_out.min():.2f}, {decoder_out.max():.2f}]")
    print(f"Heatmap 6nm: range [{heatmap_6nm.min():.3f}, {heatmap_6nm.max():.3f}]")
    print(f"Heatmap 12nm: range [{heatmap_12nm.min():.3f}, {heatmap_12nm.max():.3f}]")
    print(f"Offset X: range [{offset_x.min():.3f}, {offset_x.max():.3f}]")
    print(f"Offset Y: range [{offset_y.min():.3f}, {offset_y.max():.3f}]")

    # Create display images
    display_size = 280

    decoder_pil = Image.fromarray((decoder_max * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    heatmap_6nm_pil = Image.fromarray((heatmap_6nm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    heatmap_12nm_pil = Image.fromarray((heatmap_12nm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    offset_x_pil = Image.fromarray((normalize(offset_x) * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    offset_y_pil = Image.fromarray((normalize(offset_y) * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)

    # Apply colormaps
    decoder_rgb = np.stack([np.array(decoder_pil)] * 3, axis=-1)
    decoder_color = Image.fromarray(decoder_rgb)

    heatmap_6nm_rgb = apply_colormap_hot(np.array(heatmap_6nm_pil) / 255.0)
    heatmap_6nm_color = Image.fromarray(heatmap_6nm_rgb)

    heatmap_12nm_rgb = apply_colormap_hot(np.array(heatmap_12nm_pil) / 255.0)
    heatmap_12nm_color = Image.fromarray(heatmap_12nm_rgb)

    offset_x_rgb = apply_colormap_diverging(np.array(offset_x_pil) / 255.0)
    offset_x_color = Image.fromarray(offset_x_rgb)

    offset_y_rgb = apply_colormap_diverging(np.array(offset_y_pil) / 255.0)
    offset_y_color = Image.fromarray(offset_y_rgb)

    # Create layout
    margin = 35
    spacing = 30

    # Two rows: Heatmap head and Offset head
    # Row 1: Decoder → Heatmap (6nm) → Heatmap (12nm)
    # Row 2: (spacer) → Offset X → Offset Y

    total_width = display_size + spacing + display_size + spacing + display_size + margin * 2
    total_height = 2 * (display_size + 60) + margin * 2 + 180

    viz = Image.new('RGB', (total_width, total_height), (15, 23, 42))
    draw = ImageDraw.Draw(viz)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        tiny_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font
        tiny_font = title_font

    # ===== ROW 1: HEATMAP HEAD =====
    y_row1 = margin

    # Decoder input
    x1 = margin
    draw.rectangle([(x1 - 2, y_row1 - 2), (x1 + display_size + 2, y_row1 + display_size + 2)],
                  outline=(100, 150, 255), width=2)
    viz.paste(decoder_color, (x1, y_row1))
    draw.text((x1, y_row1 + display_size + 8), "Decoder Output", fill=(200, 200, 200), font=label_font)
    draw.text((x1, y_row1 + display_size + 25), "64 channels, stride-2", fill=(150, 150, 150), font=small_font)

    # Arrow
    draw.line([(x1 + display_size + 10, y_row1 + display_size/2), (x1 + display_size + spacing - 10, y_row1 + display_size/2)],
             fill=(255, 200, 100), width=3)
    draw.polygon([(x1 + display_size + spacing - 10, y_row1 + display_size/2),
                 (x1 + display_size + spacing, y_row1 + display_size/2 - 5),
                 (x1 + display_size + spacing, y_row1 + display_size/2 + 5)],
                fill=(255, 200, 100))

    # Heatmap 6nm
    x2 = x1 + display_size + spacing
    draw.rectangle([(x2 - 2, y_row1 - 2), (x2 + display_size + 2, y_row1 + display_size + 2)],
                  outline=(255, 100, 100), width=2)
    viz.paste(heatmap_6nm_color, (x2, y_row1))
    draw.text((x2, y_row1 + display_size + 8), "6nm Heatmap", fill=(200, 200, 200), font=label_font)
    draw.text((x2, y_row1 + display_size + 25), "AMPA receptors", fill=(150, 150, 150), font=small_font)
    draw.text((x2, y_row1 + display_size + 40), f"range [0, {heatmap_6nm.max():.3f}]", fill=(120, 120, 120), font=tiny_font)

    # Arrow
    draw.line([(x2 + display_size + 10, y_row1 + display_size/2), (x2 + display_size + spacing - 10, y_row1 + display_size/2)],
             fill=(255, 200, 100), width=3)
    draw.polygon([(x2 + display_size + spacing - 10, y_row1 + display_size/2),
                 (x2 + display_size + spacing, y_row1 + display_size/2 - 5),
                 (x2 + display_size + spacing, y_row1 + display_size/2 + 5)],
                fill=(255, 200, 100))

    # Heatmap 12nm
    x3 = x2 + display_size + spacing
    draw.rectangle([(x3 - 2, y_row1 - 2), (x3 + display_size + 2, y_row1 + display_size + 2)],
                  outline=(255, 100, 100), width=2)
    viz.paste(heatmap_12nm_color, (x3, y_row1))
    draw.text((x3, y_row1 + display_size + 8), "12nm Heatmap", fill=(200, 200, 200), font=label_font)
    draw.text((x3, y_row1 + display_size + 25), "NMDA receptors", fill=(150, 150, 150), font=small_font)
    draw.text((x3, y_row1 + display_size + 40), f"range [0, {heatmap_12nm.max():.3f}]", fill=(120, 120, 120), font=tiny_font)

    # ===== ROW 2: OFFSET HEAD =====
    y_row2 = y_row1 + display_size + 80

    # Offset X
    draw.rectangle([(x2 - 2, y_row2 - 2), (x2 + display_size + 2, y_row2 + display_size + 2)],
                  outline=(100, 200, 255), width=2)
    viz.paste(offset_x_color, (x2, y_row2))
    draw.text((x2, y_row2 + display_size + 8), "Offset X (dx)", fill=(200, 200, 200), font=label_font)
    draw.text((x2, y_row2 + display_size + 25), "Horizontal refinement", fill=(150, 150, 150), font=small_font)
    draw.text((x2, y_row2 + display_size + 40), f"range [{offset_x.min():.3f}, {offset_x.max():.3f}]", fill=(120, 120, 120), font=tiny_font)

    # Offset Y
    x4 = x3
    draw.rectangle([(x4 - 2, y_row2 - 2), (x4 + display_size + 2, y_row2 + display_size + 2)],
                  outline=(100, 200, 255), width=2)
    viz.paste(offset_y_color, (x4, y_row2))
    draw.text((x4, y_row2 + display_size + 8), "Offset Y (dy)", fill=(200, 200, 200), font=label_font)
    draw.text((x4, y_row2 + display_size + 25), "Vertical refinement", fill=(150, 150, 150), font=small_font)
    draw.text((x4, y_row2 + display_size + 40), f"range [{offset_y.min():.3f}, {offset_y.max():.3f}]", fill=(120, 120, 120), font=tiny_font)

    # Arrow for offset
    draw.line([(x1 + display_size + 10, y_row2 + display_size/2), (x2 - 10, y_row2 + display_size/2)],
             fill=(100, 200, 255), width=3)
    draw.polygon([(x2 - 10, y_row2 + display_size/2),
                 (x2 - 20, y_row2 + display_size/2 - 5),
                 (x2 - 20, y_row2 + display_size/2 + 5)],
                fill=(100, 200, 255))
    draw.text((x1 + display_size + spacing/2 - 30, y_row2 + display_size/2 - 15),
             "Offset Head", fill=(100, 200, 255), font=small_font)

    # Title
    title = "Detection Heads: Heatmap (Confidence) + Offset (Sub-pixel Refinement)"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = max(20, (total_width - title_width) // 2)

    draw.rectangle([(title_x - 10, 8), (title_x + title_width + 10, 30)], fill=(10, 15, 30))
    draw.text((title_x, 9), title, fill=(255, 200, 100), font=title_font)

    # Description
    desc_y = y_row2 + display_size + 50
    descriptions = [
        "HEATMAP HEAD (Red outputs): Classifies each stride-2 grid location as particle or background",
        "  • Two separate channels: 6nm AMPA receptors (left) and 12nm NMDA receptors (center)",
        "  • Sigmoid activation produces confidence scores 0-1 (bright = high confidence particle)",
        "  • Only grid cells with high confidence are processed as particle detections",
        "",
        "OFFSET HEAD (Blue outputs): Refines particle location within each grid cell",
        "  • Two channels: dx (left) and dy (right) sub-pixel offsets",
        "  • Unbounded regression (-∞ to +∞) predicts offset from grid center",
        "  • Combined with grid location gives sub-pixel accuracy (±0.5px) for final coordinates",
        "",
        "Together: High-confidence heatmap grid cells + offset refinement = precise particle positions",
    ]

    for i, desc in enumerate(descriptions):
        draw.text((margin, desc_y + i * 13), desc, fill=(150, 170, 190), font=small_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    viz.save(output_path, dpi=(300, 300))
    print(f"✓ Saved to {output_path}")


def main():
    checkpoint_path = "checkpoints/final/final_model.pth"
    image_path = "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"
    output_path = "results/diagrams/14_detection_heads.png"

    if not os.path.exists(checkpoint_path) or not os.path.exists(image_path):
        print("Error: Missing files")
        return

    print("Loading image...")
    image_tensor, _ = load_image(image_path)

    print("Extracting head features...")
    features_dict = extract_head_features(checkpoint_path, image_tensor)

    print("Creating visualization...")
    visualize_heads(features_dict, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
