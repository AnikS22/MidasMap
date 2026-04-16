"""
Visualize BiFPN (Bidirectional Feature Pyramid Network) fusion process.
Shows how multi-scale features are combined for better detection.
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


def extract_bifpn_features(checkpoint_path: str, image_tensor: torch.Tensor):
    """Extract BiFPN input and output."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2, imagenet_encoder_fallback=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features_dict = {}

    def hook_bifpn_input(module, input, output):
        # Capture the input to BiFPN - this is the list [P2, P3, P4, P5]
        features_dict['bifpn_input'] = [f.detach().cpu() for f in input[0]]

    def hook_bifpn_output(module, input, output):
        # Capture BiFPN output
        features_dict['bifpn_output'] = [f.detach().cpu() for f in output]

    model.bifpn.register_forward_hook(hook_bifpn_output)

    with torch.no_grad():
        # Get encoder outputs
        x0 = model.stem(image_tensor)
        p2 = model.layer1(x0)
        p3 = model.layer2(p2)
        p4 = model.layer3(p3)
        p5 = model.layer4(p4)

    features_dict['bifpn_input'] = [p2.detach().cpu(), p3.detach().cpu(),
                                     p4.detach().cpu(), p5.detach().cpu()]

    # Run BiFPN
    with torch.no_grad():
        bifpn_out = model.bifpn(features_dict['bifpn_input'])
        features_dict['bifpn_output'] = [f.detach().cpu() for f in bifpn_out]

    return features_dict


def apply_colormap_jet(data: np.ndarray) -> np.ndarray:
    """Apply jet colormap (blue → cyan → green → yellow → red)."""
    colors = np.array([
        [0, 0, 255],      # Blue
        [0, 255, 255],    # Cyan
        [0, 255, 0],      # Green
        [255, 255, 0],    # Yellow
        [255, 0, 0],      # Red
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


def visualize_bifpn(features_dict: dict, output_path: str):
    """Create BiFPN visualization."""

    print("Processing BiFPN features...")

    bifpn_input = features_dict['bifpn_input']
    bifpn_output = features_dict['bifpn_output']

    # Names and channel info
    levels = ['P2 (stride 4)', 'P3 (stride 8)', 'P4 (stride 16)', 'P5 (stride 32)']
    input_channels = [256, 512, 1024, 2048]

    # Extract max activation across channels
    input_features = []
    output_features = []

    for i, (inp, out) in enumerate(zip(bifpn_input, bifpn_output)):
        inp_np = inp[0].numpy()  # Remove batch dim
        out_np = out[0].numpy()

        inp_max = np.max(inp_np, axis=0)
        out_max = np.max(out_np, axis=0)

        input_features.append(normalize(inp_max))
        output_features.append(normalize(out_max))

        print(f"{levels[i]}: Input {inp_np.shape}, Output {out_np.shape}")

    # Display size varies by stride
    display_sizes = [256, 192, 128, 96]

    # Create layout
    margin = 40
    col_spacing = 20
    row_spacing = 80

    # Two rows: Input and Output
    total_width = sum(display_sizes) + (len(display_sizes) - 1) * col_spacing + margin * 2
    total_height = 2 * max(display_sizes) + row_spacing + margin * 2 + 150

    viz = Image.new('RGB', (total_width, total_height), (15, 23, 42))
    draw = ImageDraw.Draw(viz)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font

    # Draw input row
    x = margin
    for i, (feat, size, level, channels) in enumerate(zip(input_features, display_sizes, levels, input_channels)):
        img_pil = Image.fromarray((feat * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
        img_rgb = apply_colormap_jet(np.array(img_pil) / 255.0)
        img_color = Image.fromarray(img_rgb)

        # Border
        draw.rectangle([(x - 1, margin - 1), (x + size + 1, margin + size + 1)],
                      outline=(100, 150, 255), width=2)

        viz.paste(img_color, (x, margin))

        # Label
        draw.text((x, margin + size + 5), f"{level}", fill=(200, 200, 200), font=label_font)
        draw.text((x, margin + size + 20), f"{channels}ch", fill=(150, 150, 150), font=small_font)

        x += size + col_spacing

    # Draw arrows and fusion annotations
    arrow_y = margin + max(display_sizes) + 15
    draw.text((margin, arrow_y), "Encoder Output (4 scales)", fill=(200, 200, 200), font=label_font)

    # Draw output row
    x = margin
    for i, (feat, size, level) in enumerate(zip(output_features, display_sizes, levels)):
        y = margin + max(display_sizes) + row_spacing

        img_pil = Image.fromarray((feat * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
        img_rgb = apply_colormap_jet(np.array(img_pil) / 255.0)
        img_color = Image.fromarray(img_rgb)

        # Border (green for output)
        draw.rectangle([(x - 1, y - 1), (x + size + 1, y + size + 1)],
                      outline=(76, 220, 100), width=2)

        viz.paste(img_color, (x, y))

        # Label
        draw.text((x, y + size + 5), f"{level} Fused", fill=(200, 200, 200), font=label_font)
        draw.text((x, y + size + 20), f"128ch", fill=(150, 150, 150), font=small_font)

        x += size + col_spacing

    # Draw fusion annotations
    fusion_y = margin + max(display_sizes) + row_spacing - 30
    draw.text((margin, fusion_y), "BiFPN Fusion (top-down ↑ bottom-up ↓)", fill=(255, 200, 100), font=label_font)

    # Title
    title = "BiFPN: Multi-Scale Feature Fusion for Particle Detection"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = max(20, (total_width - title_width) // 2)

    draw.rectangle([(title_x - 10, 8), (title_x + title_width + 10, 30)], fill=(10, 15, 30))
    draw.text((title_x, 9), title, fill=(255, 200, 100), font=title_font)

    # Description
    desc_y = total_height - 120
    descriptions = [
        "TOP ROW: Encoder outputs at 4 different scales (resolutions)",
        "  • P2: Fine details at stride 4 (best for detecting 4-6nm particles)",
        "  • P3-P5: Coarser scales for context and large-scale patterns",
        "BIFPN FUSION (middle): Two passes combine information bidirectionally",
        "  • Top-down: High-level features inform finer scales",
        "  • Bottom-up: Fine details inform coarser scales",
        "BOTTOM ROW: Output after BiFPN - all unified to 128 channels with fused multi-scale context",
        "Result: P2 now contains both fine-grained details AND global context for robust detection",
    ]

    for i, desc in enumerate(descriptions):
        draw.text((margin, desc_y + i * 14), desc, fill=(150, 170, 190), font=small_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    viz.save(output_path, dpi=(300, 300))
    print(f"✓ Saved to {output_path}")


def main():
    checkpoint_path = "checkpoints/final/final_model.pth"
    image_path = "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"
    output_path = "results/diagrams/12_bifpn_fusion.png"

    if not os.path.exists(checkpoint_path) or not os.path.exists(image_path):
        print("Error: Missing files")
        return

    print("Loading image...")
    image_tensor, _ = load_image(image_path)

    print("Extracting BiFPN features...")
    features_dict = extract_bifpn_features(checkpoint_path, image_tensor)

    print("Creating visualization...")
    visualize_bifpn(features_dict, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
