"""
Visualize the complete pipeline showing the actual particle detections.
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


def extract_features(checkpoint_path: str, image_tensor: torch.Tensor):
    """Extract features."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2, imagenet_encoder_fallback=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features_dict = {}

    def hook_layer(name):
        def hook(module, input, output):
            features_dict[name] = output.detach().cpu()
        return hook

    model.layer1.register_forward_hook(hook_layer('layer1'))
    model.layer2.register_forward_hook(hook_layer('layer2'))
    model.layer3.register_forward_hook(hook_layer('layer3'))
    model.layer4.register_forward_hook(hook_layer('layer4'))

    with torch.no_grad():
        heatmap, offsets = model(image_tensor)

    features_dict['heatmap'] = heatmap.detach().cpu()
    features_dict['offsets'] = offsets.detach().cpu()

    return features_dict


def apply_colormap_hot(data: np.ndarray) -> np.ndarray:
    """Apply hot colormap."""
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


def visualize(original_img: np.ndarray, features_dict: dict, output_path: str):
    """Create the visualization."""

    print("Processing features...")

    # Layer features (max across channels)
    layer1_feat = features_dict['layer1'][0].numpy()
    layer2_feat = features_dict['layer2'][0].numpy()
    layer3_feat = features_dict['layer3'][0].numpy()
    layer4_feat = features_dict['layer4'][0].numpy()
    heatmap = features_dict['heatmap'][0].numpy()

    # Get max activation
    layer1_max = np.max(layer1_feat, axis=0)
    layer2_max = np.max(layer2_feat, axis=0)
    layer3_max = np.max(layer3_feat, axis=0)
    layer4_max = np.max(layer4_feat, axis=0)

    # Normalize to [0, 1]
    def normalize(x):
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        return np.zeros_like(x)

    layer1_norm = normalize(layer1_max)
    layer2_norm = normalize(layer2_max)
    layer3_norm = normalize(layer3_max)
    layer4_norm = normalize(layer4_max)

    # For heatmap, separate the two classes
    heatmap_6nm = heatmap[0]  # 6nm particles
    heatmap_12nm = heatmap[1]  # 12nm particles
    heatmap_max = np.maximum(heatmap_6nm, heatmap_12nm)
    heatmap_norm = normalize(heatmap_max)

    print(f"Heatmap 6nm range: [{heatmap_6nm.min():.3f}, {heatmap_6nm.max():.3f}]")
    print(f"Heatmap 12nm range: [{heatmap_12nm.min():.3f}, {heatmap_12nm.max():.3f}]")
    print(f"Heatmap max range: [{heatmap_max.min():.3f}, {heatmap_max.max():.3f}]")

    # Display size
    display_size = 280

    layer1_display = Image.fromarray((layer1_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    layer2_display = Image.fromarray((layer2_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    layer3_display = Image.fromarray((layer3_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    layer4_display = Image.fromarray((layer4_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    heatmap_display = Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    orig_display = Image.fromarray((original_img * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)

    # Layout
    cols = 3
    rows = 2
    spacing = 16
    margin = 35

    total_width = cols * (display_size + spacing) + margin * 2 + spacing
    total_height = rows * (display_size + spacing) + margin * 2 + 140

    viz = Image.new('RGB', (total_width, total_height), (15, 23, 42))
    draw = ImageDraw.Draw(viz)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font

    # Positions
    positions = [
        (margin, margin, orig_display, "Original\nInput", False),
        (margin + display_size + spacing, margin, layer1_display, "Layer 1\n(256 channels)", True),
        (margin + 2 * (display_size + spacing), margin, layer2_display, "Layer 2\n(512 channels)", True),
        (margin, margin + display_size + spacing, layer3_display, "Layer 3\n(1024 channels)", True),
        (margin + display_size + spacing, margin + display_size + spacing, layer4_display, "Layer 4\n(2048 channels)", True),
        (margin + 2 * (display_size + spacing), margin + display_size + spacing, heatmap_display, "Detection\nHeatmap", False),
    ]

    for x, y, img_pil, label, apply_viridis in positions:
        img_array = np.array(img_pil) / 255.0

        if label.startswith("Detection"):
            img_rgb = apply_colormap_hot(img_array)
        elif label.startswith("Original"):
            img_rgb = np.stack([np.array(img_pil)] * 3, axis=-1)
        else:
            img_rgb = apply_colormap_viridis(img_array)

        img_color = Image.fromarray(img_rgb)

        # Border
        border_color = (255, 100, 100) if label.startswith("Detection") else (100, 150, 255)
        draw.rectangle([(x - 2, y - 2), (x + display_size + 2, y + display_size + 2)],
                      outline=border_color, width=2)

        viz.paste(img_color, (x, y))

        # Label
        lines = label.split('\n')
        for i, line in enumerate(lines):
            label_y = y + display_size + 5 + i * 14
            draw.text((x, label_y), line, fill=(200, 200, 200), font=label_font)

    # Title
    title = "Neural Network Encoding Pipeline: Learning to Detect Immunogold Particles"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = max(20, (total_width - title_width) // 2)

    draw.rectangle([(title_x - 10, 8), (title_x + title_width + 10, 30)], fill=(10, 15, 30))
    draw.text((title_x, 9), title, fill=(255, 200, 100), font=title_font)

    # Description
    desc_y = total_height - 115
    descriptions = [
        "INPUT (left): Original 2048×2048 grayscale TEM synapse image",
        "ENCODING (middle 4 boxes): ResNet-50 learns increasingly complex features",
        "  • Layer 1: Detects edges, boundaries, simple textures (256 channels, stride 4)",
        "  • Layer 2-4: Learns hierarchical patterns specific to immunogold particles",
        "DETECTION OUTPUT (bottom right, red=high confidence): Heatmap showing predicted particle locations",
        "  • Bright red regions = model is confident a particle exists",
        "  • Separate channels for 6nm AMPA and 12nm NMDA receptors",
    ]

    for i, desc in enumerate(descriptions):
        draw.text((margin, desc_y + i * 15), desc, fill=(150, 170, 190), font=small_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    viz.save(output_path, dpi=(300, 300))
    print(f"✓ Saved to {output_path}")


def main():
    checkpoint_path = "checkpoints/final/final_model.pth"
    image_path = "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"
    output_path = "results/diagrams/11_encoder_features.png"

    if not os.path.exists(checkpoint_path) or not os.path.exists(image_path):
        print("Error: Missing checkpoint or image")
        return

    print("Loading image...")
    image_tensor, original_img = load_image(image_path)

    print("Extracting features...")
    features_dict = extract_features(checkpoint_path, image_tensor)

    print("Creating visualization...")
    visualize(original_img, features_dict, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
