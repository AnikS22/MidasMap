"""
Visualize the Decoder output (stride-2, 64 channels).
Shows the actual learned features before the detection heads.
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


def extract_decoder_features(checkpoint_path: str, image_tensor: torch.Tensor):
    """Extract decoder output before heads."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2, imagenet_encoder_fallback=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features_dict = {}

    def hook_bifpn(module, input, output):
        features_dict['bifpn_output'] = [f.detach().cpu() for f in output]

    def hook_decoder(module, input, output):
        features_dict['decoder_output'] = output.detach().cpu()

    model.bifpn.register_forward_hook(hook_bifpn)
    model.upsample.register_forward_hook(hook_decoder)

    with torch.no_grad():
        model(image_tensor)

    return features_dict


def apply_colormap_plasma(data: np.ndarray) -> np.ndarray:
    """Apply plasma colormap."""
    colors = np.array([
        [13, 8, 135],
        [75, 0, 130],
        [148, 0, 211],
        [255, 0, 0],
        [255, 165, 0],
        [255, 255, 0],
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


def apply_colormap_viridis(data: np.ndarray) -> np.ndarray:
    """Apply viridis colormap."""
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


def normalize(x):
    """Normalize to [0, 1]."""
    if x.max() > x.min():
        return (x - x.min()) / (x.max() - x.min())
    return np.zeros_like(x)


def visualize_decoder(features_dict: dict, output_path: str):
    """Create decoder visualization with individual channels."""

    print("Processing decoder features...")

    # BiFPN P2 output
    bifpn_p2 = features_dict['bifpn_output'][0][0].numpy()  # (128, H, W)
    bifpn_p2_max = normalize(np.max(bifpn_p2, axis=0))

    # Decoder output (64 channels)
    decoder_out = features_dict['decoder_output'][0].numpy()  # (64, H, W)
    decoder_out_max = normalize(np.max(decoder_out, axis=0))

    # Extract individual decoder channels for grid visualization
    decoder_channels = decoder_out[:32]  # First 32 channels
    decoder_normalized = np.zeros_like(decoder_channels)
    for i in range(len(decoder_channels)):
        decoder_normalized[i] = normalize(decoder_channels[i])

    print(f"BiFPN P2: {bifpn_p2.shape} (128 channels, stride-4)")
    print(f"Decoder output: {decoder_out.shape} (64 channels, stride-2)")
    print(f"Feature value ranges:")
    print(f"  BiFPN: [{bifpn_p2.min():.2f}, {bifpn_p2.max():.2f}]")
    print(f"  Decoder: [{decoder_out.min():.2f}, {decoder_out.max():.2f}]")

    # Create visualization: left side (BiFPN), right side (decoder features grid)
    margin = 30
    spacing = 40

    # Left section: BiFPN output
    bifpn_display_size = 300
    bifpn_pil = Image.fromarray((bifpn_p2_max * 255).astype(np.uint8))
    bifpn_display = bifpn_pil.resize((bifpn_display_size, bifpn_display_size), Image.LANCZOS)
    bifpn_rgb = apply_colormap_plasma(np.array(bifpn_display) / 255.0)
    bifpn_color = Image.fromarray(bifpn_rgb)

    # Right section: Decoder feature channels grid
    cell_size = 64
    grid_cols = 4
    grid_rows = 8
    grid_padding = 2

    grid_width = grid_cols * (cell_size + grid_padding) + grid_padding
    grid_height = grid_rows * (cell_size + grid_padding) + grid_padding

    total_width = bifpn_display_size + spacing + grid_width + margin * 2
    total_height = max(bifpn_display_size, grid_height) + margin * 2 + 200

    viz = Image.new('RGB', (total_width, total_height), (15, 23, 42))
    draw = ImageDraw.Draw(viz)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
        tiny_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font
        tiny_font = title_font

    # Draw BiFPN P2 on the left
    x_bifpn = margin
    y_bifpn = margin

    draw.rectangle([(x_bifpn - 2, y_bifpn - 2), (x_bifpn + bifpn_display_size + 2, y_bifpn + bifpn_display_size + 2)],
                  outline=(100, 150, 255), width=2)
    viz.paste(bifpn_color, (x_bifpn, y_bifpn))

    draw.text((x_bifpn, y_bifpn + bifpn_display_size + 8), "BiFPN P2 Output", fill=(200, 200, 200), font=label_font)
    draw.text((x_bifpn, y_bifpn + bifpn_display_size + 25), "128 channels, Stride-4", fill=(150, 150, 150), font=small_font)
    draw.text((x_bifpn, y_bifpn + bifpn_display_size + 40), f"({bifpn_p2.shape[1]} × {bifpn_p2.shape[2]} pixels)", fill=(120, 120, 120), font=tiny_font)

    # Draw arrow
    arrow_x = x_bifpn + bifpn_display_size + spacing / 2
    arrow_y = y_bifpn + bifpn_display_size / 2
    draw.line([(x_bifpn + bifpn_display_size + 10, arrow_y), (arrow_x + 15, arrow_y)],
             fill=(255, 200, 100), width=3)
    draw.polygon([(arrow_x + 15, arrow_y), (arrow_x + 25, arrow_y - 5), (arrow_x + 25, arrow_y + 5)],
                fill=(255, 200, 100))

    # Draw upsampling info
    upsample_text = "ConvTranspose2d\n(kernel=4, stride=2)"
    draw.text((arrow_x - 35, arrow_y - 20), upsample_text, fill=(255, 200, 100), font=tiny_font)

    # Draw decoder feature grid on the right
    x_grid = x_bifpn + bifpn_display_size + spacing
    y_grid = y_bifpn

    # Draw label above grid
    draw.text((x_grid, y_grid - 25), "Decoder Features (64 channels, showing first 32)",
             fill=(200, 200, 200), font=label_font)

    # Draw feature grid
    for idx, feat_map in enumerate(decoder_normalized):
        row = idx // grid_cols
        col = idx % grid_cols

        x = x_grid + col * (cell_size + grid_padding)
        y = y_grid + row * (cell_size + grid_padding)

        # Resize feature map
        feat_pil = Image.fromarray((feat_map * 255).astype(np.uint8))
        feat_resized = feat_pil.resize((cell_size, cell_size), Image.LANCZOS)

        # Apply colormap
        feat_rgb = apply_colormap_viridis(np.array(feat_resized) / 255.0)
        feat_color = Image.fromarray(feat_rgb)

        # Subtle border
        draw.rectangle([(x - 1, y - 1), (x + cell_size + 1, y + cell_size + 1)],
                      outline=(50, 75, 125), width=1)

        viz.paste(feat_color, (x, y))

    # Draw stride comparison box
    compare_y = max(y_bifpn + bifpn_display_size, y_grid + grid_height) + 20

    draw.rectangle([(margin, compare_y), (margin + 600, compare_y + 90)],
                  outline=(255, 150, 100), width=2, fill=(25, 35, 60))

    draw.text((margin + 10, compare_y + 5), "WHY STRIDE-2 MATTERS:", fill=(255, 150, 100), font=label_font)
    draw.text((margin + 10, compare_y + 25),
             "4-6nm particle at STRIDE-4: Collapses to 1 pixel → SIGNAL LOST",
             fill=(255, 100, 100), font=small_font)
    draw.text((margin + 10, compare_y + 42),
             "4-6nm particle at STRIDE-2: Occupies 2-3 pixels → DETECTABLE",
             fill=(100, 255, 100), font=small_font)
    draw.text((margin + 10, compare_y + 59),
             "Gaussian peak extraction finds center with ±0.5px sub-pixel accuracy",
             fill=(150, 200, 255), font=small_font)

    # Title
    title = "Decoder: 2× Upsampling (Stride-4 → Stride-2) + Feature Learning"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = max(20, (total_width - title_width) // 2)

    draw.rectangle([(title_x - 10, 8), (title_x + title_width + 10, 30)], fill=(10, 15, 30))
    draw.text((title_x, 9), title, fill=(255, 200, 100), font=title_font)

    # Description at bottom
    desc_y = compare_y + 95
    descriptions = [
        "LEFT: BiFPN P2 output (128 channels) at stride-4 - coarse spatial grid",
        "ARROW: 2× upsampling via transposed convolution (ConvTranspose2d)",
        "RIGHT: Decoder output (64 channels) at stride-2 - fine spatial grid ready for detection",
        "INDIVIDUAL CHANNELS: Each of the 64 decoder channels has learned different features for detection",
        "NO HEADS APPLIED: This is raw decoder output before heatmap/offset classification layers",
    ]

    for i, desc in enumerate(descriptions):
        draw.text((margin, desc_y + i * 14), desc, fill=(150, 170, 190), font=small_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    viz.save(output_path, dpi=(300, 300))
    print(f"✓ Saved to {output_path}")


def main():
    checkpoint_path = "checkpoints/final/final_model.pth"
    image_path = "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"
    output_path = "results/diagrams/13_decoder_stride2.png"

    if not os.path.exists(checkpoint_path) or not os.path.exists(image_path):
        print("Error: Missing files")
        return

    print("Loading image...")
    image_tensor, _ = load_image(image_path)

    print("Extracting decoder features...")
    features_dict = extract_decoder_features(checkpoint_path, image_tensor)

    print("Creating visualization...")
    visualize_decoder(features_dict, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
