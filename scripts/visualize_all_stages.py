"""
Visualize the complete encoding pipeline: input → layer features → heatmap output
Shows progression of learning through the network.
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
    print(f"Image size: {width} × {height}")

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


def extract_all_features(checkpoint_path: str, image_tensor: torch.Tensor):
    """Extract features from all layers and final output."""

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print("Loading model...")
    model = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2, imagenet_encoder_fallback=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features_dict = {}

    def hook_layer(name):
        def hook(module, input, output):
            features_dict[name] = output.detach().cpu()
        return hook

    # Register hooks on all encoder layers
    model.layer1.register_forward_hook(hook_layer('layer1'))
    model.layer2.register_forward_hook(hook_layer('layer2'))
    model.layer3.register_forward_hook(hook_layer('layer3'))
    model.layer4.register_forward_hook(hook_layer('layer4'))

    print("Running inference...")
    with torch.no_grad():
        heatmap, offsets = model(image_tensor)

    # heatmap shape: (1, 2, H/2, W/2) - two channels for 6nm and 12nm
    features_dict['heatmap'] = heatmap.detach().cpu()
    features_dict['offsets'] = offsets.detach().cpu()

    return features_dict


def apply_colormap_hot(data: np.ndarray) -> np.ndarray:
    """Apply hot colormap (black → red → yellow → white)."""
    colors = np.array([
        [0, 0, 0],        # Black
        [128, 0, 0],      # Dark red
        [255, 0, 0],      # Red
        [255, 165, 0],    # Orange
        [255, 255, 0],    # Yellow
        [255, 255, 255],  # White
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


def visualize_progression(original_img: np.ndarray, features_dict: dict, output_path: str):
    """Visualize progression through network layers and final output."""

    print("Preparing features...")

    # Extract and prepare features from each layer (taking max across channels for visibility)
    layer1_feat = features_dict['layer1'][0].numpy()  # (256, H, W)
    layer2_feat = features_dict['layer2'][0].numpy()  # (512, H, W)
    layer3_feat = features_dict['layer3'][0].numpy()  # (1024, H, W)
    layer4_feat = features_dict['layer4'][0].numpy()  # (2048, H, W)
    heatmap = features_dict['heatmap'][0].numpy()    # (2, H/2, W/2)

    print(f"Layer1: shape {layer1_feat.shape}, range [{layer1_feat.min():.2f}, {layer1_feat.max():.2f}]")
    print(f"Layer2: shape {layer2_feat.shape}, range [{layer2_feat.min():.2f}, {layer2_feat.max():.2f}]")
    print(f"Layer3: shape {layer3_feat.shape}, range [{layer3_feat.min():.2f}, {layer3_feat.max():.2f}]")
    print(f"Layer4: shape {layer4_feat.shape}, range [{layer4_feat.min():.2f}, {layer4_feat.max():.2f}]")
    print(f"Heatmap: shape {heatmap.shape}, range [{heatmap.min():.2f}, {heatmap.max():.2f}]")

    # Get max activation across channels for visualization
    layer1_max = np.max(layer1_feat, axis=0)
    layer2_max = np.max(layer2_feat, axis=0)
    layer3_max = np.max(layer3_feat, axis=0)
    layer4_max = np.max(layer4_feat, axis=0)

    # Normalize to [0, 1]
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        return (x - x_min) / (x_max - x_min + 1e-8) if x_max > x_min else np.zeros_like(x)

    layer1_norm = normalize(layer1_max)
    layer2_norm = normalize(layer2_max)
    layer3_norm = normalize(layer3_max)
    layer4_norm = normalize(layer4_max)

    # Heatmap - use max of 6nm and 12nm channels
    heatmap_max = np.maximum(heatmap[0], heatmap[1])
    heatmap_norm = normalize(heatmap_max)

    # Resize to consistent size for display
    display_size = 256

    layer1_display = Image.fromarray((layer1_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    layer2_display = Image.fromarray((layer2_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    layer3_display = Image.fromarray((layer3_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    layer4_display = Image.fromarray((layer4_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    heatmap_display = Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)
    orig_display = Image.fromarray((original_img * 255).astype(np.uint8)).resize((display_size, display_size), Image.LANCZOS)

    # Create visualization
    cols = 3
    rows = 2
    spacing = 20
    margin = 30

    total_width = cols * (display_size + spacing) + margin * 2
    total_height = rows * (display_size + spacing) + margin * 2 + 100

    viz = Image.new('RGB', (total_width, total_height), (15, 23, 42))
    draw = ImageDraw.Draw(viz)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font

    # Position each feature map
    positions = [
        (margin, margin, orig_display, "Original Input"),
        (margin + display_size + spacing, margin, layer1_display, "Layer1 Features"),
        (margin + 2 * (display_size + spacing), margin, layer2_display, "Layer2 Features"),
        (margin, margin + display_size + spacing, layer3_display, "Layer3 Features"),
        (margin + display_size + spacing, margin + display_size + spacing, layer4_display, "Layer4 Features"),
        (margin + 2 * (display_size + spacing), margin + display_size + spacing, heatmap_display, "Detection Output"),
    ]

    for x, y, img_pil, label in positions:
        # Apply colormap
        img_array = np.array(img_pil) / 255.0
        if label == "Detection Output":
            img_rgb = apply_colormap_hot(img_array)
        elif label == "Original Input":
            # Keep grayscale for original
            img_rgb = np.stack([np.array(img_pil)] * 3, axis=-1)
        else:
            img_rgb = apply_colormap_viridis(img_array)

        img_color = Image.fromarray(img_rgb)

        # Draw border
        draw.rectangle([(x - 1, y - 1), (x + display_size + 1, y + display_size + 1)],
                      outline=(100, 150, 255), width=2)

        viz.paste(img_color, (x, y))

        # Label
        label_y = y + display_size + 5
        draw.text((x, label_y), label, fill=(200, 200, 200), font=label_font)

    # Title
    title = "ResNet-50 Encoding Pipeline: Input → Features → Detection"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (total_width - title_width) // 2

    draw.rectangle([(title_x - 10, 10), (title_x + title_width + 10, 28)], fill=(10, 15, 30))
    draw.text((title_x, 11), title, fill=(255, 200, 100), font=title_font)

    # Description
    desc = "Each layer learns progressively complex features. Layer1 detects simple edges/textures. By Layer4, network sees high-level"
    desc2 = "patterns. Finally, detection heads output heatmaps showing predicted particle locations (bright = high confidence)."
    draw.text((margin, total_height - 40), desc, fill=(150, 170, 190), font=small_font)
    draw.text((margin, total_height - 25), desc2, fill=(150, 170, 190), font=small_font)

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
    features_dict = extract_all_features(checkpoint_path, image_tensor)

    print("Creating visualization...")
    visualize_progression(original_img, features_dict, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
