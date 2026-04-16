"""
Visualize the Decoder: BiFPN P2 (stride-4) → Stride-2 upsampling
Shows why stride-2 is critical for detecting 4-6nm particles.
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
    """Extract features before and after decoder."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2, imagenet_encoder_fallback=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features_dict = {}

    def hook_bifpn(module, input, output):
        features_dict['bifpn_output'] = [f.detach().cpu() for f in output]

    def hook_decoder(module, input, output):
        features_dict['decoder_output'] = output.detach().cpu()

    def hook_heatmap(module, input, output):
        features_dict['heatmap_output'] = output.detach().cpu()

    model.bifpn.register_forward_hook(hook_bifpn)
    model.upsample.register_forward_hook(hook_decoder)
    model.heatmap_head.register_forward_hook(hook_heatmap)

    with torch.no_grad():
        model(image_tensor)

    return features_dict


def apply_colormap_plasma(data: np.ndarray) -> np.ndarray:
    """Apply plasma colormap (purple → red → yellow)."""
    colors = np.array([
        [13, 8, 135],     # Dark purple
        [75, 0, 130],     # Indigo
        [148, 0, 211],    # Blue-violet
        [255, 0, 0],      # Red
        [255, 165, 0],    # Orange
        [255, 255, 0],    # Yellow
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


def apply_colormap_hot(data: np.ndarray) -> np.ndarray:
    """Apply hot colormap for heatmap output."""
    colors = np.array([
        [0, 0, 0],
        [50, 0, 50],
        [128, 0, 0],
        [255, 0, 0],
        [255, 165, 0],
        [255, 255, 0],
        [255, 255, 255],
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
    """Create decoder visualization."""

    print("Processing decoder features...")

    # BiFPN P2 output (stride 4)
    bifpn_p2 = features_dict['bifpn_output'][0][0].numpy()  # (128, H/4, W/4)
    bifpn_p2_max = normalize(np.max(bifpn_p2, axis=0))

    # Decoder output (before heads, stride 2)
    decoder_out = features_dict['decoder_output'][0].numpy()  # (64, H/2, W/2)
    decoder_out_max = normalize(np.max(decoder_out, axis=0))

    # Heatmap output (stride 2, 2 channels for 6nm and 12nm)
    heatmap = features_dict['heatmap_output'][0].numpy()  # (2, H/2, W/2)
    heatmap_max = normalize(np.maximum(heatmap[0], heatmap[1]))

    print(f"BiFPN P2: {bifpn_p2.shape} → stride 4")
    print(f"Decoder output: {decoder_out.shape} → stride 2 (2x upsampling)")
    print(f"Heatmap: {heatmap.shape} → stride 2 (detection)")

    # Create display images
    bifpn_pil = Image.fromarray((bifpn_p2_max * 255).astype(np.uint8))
    decoder_pil = Image.fromarray((decoder_out_max * 255).astype(np.uint8))
    heatmap_pil = Image.fromarray((heatmap_max * 255).astype(np.uint8))

    # Display sizes
    bifpn_display_size = 240
    decoder_display_size = 320  # 2x larger (stride 2 vs stride 4)
    heatmap_display_size = 320

    bifpn_display = bifpn_pil.resize((bifpn_display_size, bifpn_display_size), Image.LANCZOS)
    decoder_display = decoder_pil.resize((decoder_display_size, decoder_display_size), Image.LANCZOS)
    heatmap_display = heatmap_pil.resize((heatmap_display_size, heatmap_display_size), Image.LANCZOS)

    # Apply colormaps
    bifpn_rgb = apply_colormap_plasma(np.array(bifpn_display) / 255.0)
    bifpn_color = Image.fromarray(bifpn_rgb)

    decoder_rgb = apply_colormap_plasma(np.array(decoder_display) / 255.0)
    decoder_color = Image.fromarray(decoder_rgb)

    heatmap_rgb = apply_colormap_hot(np.array(heatmap_display) / 255.0)
    heatmap_color = Image.fromarray(heatmap_rgb)

    # Create visualization
    margin = 40
    spacing = 60

    total_width = bifpn_display_size + spacing + decoder_display_size + spacing + heatmap_display_size + margin * 2
    total_height = max(bifpn_display_size, decoder_display_size, heatmap_display_size) + margin * 2 + 180

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

    # Position 1: BiFPN P2 (stride-4)
    x1 = margin
    y_base = margin

    draw.rectangle([(x1 - 2, y_base - 2), (x1 + bifpn_display_size + 2, y_base + bifpn_display_size + 2)],
                  outline=(100, 150, 255), width=2)
    viz.paste(bifpn_color, (x1, y_base))

    draw.text((x1, y_base + bifpn_display_size + 8), "BiFPN Output (P2)", fill=(200, 200, 200), font=label_font)
    draw.text((x1, y_base + bifpn_display_size + 25), "128 channels, Stride 4", fill=(150, 150, 150), font=small_font)
    draw.text((x1, y_base + bifpn_display_size + 40), f"{bifpn_p2.shape[1]} × {bifpn_p2.shape[2]} pixels", fill=(120, 120, 120), font=tiny_font)

    # Draw arrow
    arrow_x = x1 + bifpn_display_size + spacing / 2
    arrow_y = y_base + bifpn_display_size / 2
    draw.line([(x1 + bifpn_display_size + 10, arrow_y), (arrow_x + 15, arrow_y)],
             fill=(255, 200, 100), width=3)
    draw.polygon([(arrow_x + 15, arrow_y), (arrow_x + 25, arrow_y - 5), (arrow_x + 25, arrow_y + 5)],
                fill=(255, 200, 100))

    # Draw upsampling info
    decode_text = "ConvTranspose2d\nkernel=4, stride=2\npadding=1"
    draw.text((arrow_x - 25, arrow_y - 25), decode_text, fill=(255, 200, 100), font=tiny_font)

    # Position 2: Decoder output (stride-2)
    x2 = x1 + bifpn_display_size + spacing
    # Center vertically based on size difference
    y2 = y_base + (bifpn_display_size - decoder_display_size) // 2

    draw.rectangle([(x2 - 2, y2 - 2), (x2 + decoder_display_size + 2, y2 + decoder_display_size + 2)],
                  outline=(100, 200, 150), width=2)
    viz.paste(decoder_color, (x2, y2))

    draw.text((x2, y2 + decoder_display_size + 8), "Decoder Output", fill=(200, 200, 200), font=label_font)
    draw.text((x2, y2 + decoder_display_size + 25), "64 channels, Stride 2", fill=(150, 150, 150), font=small_font)
    draw.text((x2, y2 + decoder_display_size + 40), f"{decoder_out.shape[1]} × {decoder_out.shape[2]} pixels", fill=(120, 120, 120), font=tiny_font)

    # Draw arrow
    arrow_x2 = x2 + decoder_display_size + spacing / 2
    arrow_y2 = y2 + decoder_display_size / 2
    draw.line([(x2 + decoder_display_size + 10, arrow_y2), (arrow_x2 + 15, arrow_y2)],
             fill=(255, 200, 100), width=3)
    draw.polygon([(arrow_x2 + 15, arrow_y2), (arrow_x2 + 25, arrow_y2 - 5), (arrow_x2 + 25, arrow_y2 + 5)],
                fill=(255, 200, 100))

    # Draw heads info
    heads_text = "Heatmap Head\nOffset Head"
    draw.text((arrow_x2 - 20, arrow_y2 - 25), heads_text, fill=(255, 200, 100), font=tiny_font)

    # Position 3: Heatmap output
    x3 = x2 + decoder_display_size + spacing
    y3 = y_base + (bifpn_display_size - heatmap_display_size) // 2

    draw.rectangle([(x3 - 2, y3 - 2), (x3 + heatmap_display_size + 2, y3 + heatmap_display_size + 2)],
                  outline=(255, 100, 100), width=2)
    viz.paste(heatmap_color, (x3, y3))

    draw.text((x3, y3 + heatmap_display_size + 8), "Heatmap Output", fill=(200, 200, 200), font=label_font)
    draw.text((x3, y3 + heatmap_display_size + 25), "2 channels, Stride 2", fill=(150, 150, 150), font=small_font)
    draw.text((x3, y3 + heatmap_display_size + 40), f"{heatmap.shape[1]} × {heatmap.shape[2]} pixels", fill=(120, 120, 120), font=tiny_font)

    # Title
    title = "Decoder: BiFPN P2 (Stride-4) → Detection Output (Stride-2)"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = max(20, (total_width - title_width) // 2)

    draw.rectangle([(title_x - 10, 8), (title_x + title_width + 10, 30)], fill=(10, 15, 30))
    draw.text((title_x, 9), title, fill=(255, 200, 100), font=title_font)

    # Description
    desc_y = total_height - 165
    descriptions = [
        "STRIDE-2 IS CRITICAL FOR PARTICLE DETECTION:",
        "",
        "At stride-4: A 4-6 pixel particle → collapsed to 1 pixel in feature space → LOST",
        "At stride-2: A 4-6 pixel particle → occupies 2-3 pixels in feature space → DETECTABLE",
        "",
        "PROCESS:",
        "  1. BiFPN P2 output: 128 channels at stride-4 (fine but coarse grid)",
        "  2. ConvTranspose2d decoder: 2× upsampling → 64 channels at stride-2",
        "  3. Detection heads: Output 2-channel heatmap (6nm AMPA, 12nm NMDA receptors)",
        "     + 2-channel offset map for sub-pixel refinement",
        "",
        "RESULT: Each particle occupies 2-3 pixels in output → Gaussian peak extraction finds center with ±0.5px accuracy",
    ]

    for i, desc in enumerate(descriptions):
        draw.text((margin, desc_y + i * 13), desc, fill=(150, 170, 190), font=small_font)

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
