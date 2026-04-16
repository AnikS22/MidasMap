"""
Export head outputs with better visualization for sparse heatmaps.
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.model import ImmunogoldCenterNet


def load_image(image_path: str) -> torch.Tensor:
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
    return img_tensor


def extract_outputs(checkpoint_path: str, image_tensor: torch.Tensor):
    """Extract raw outputs."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2, imagenet_encoder_fallback=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features_dict = {}

    def hook_decoder(module, input, output):
        features_dict['decoder'] = output.detach().cpu()

    def hook_heatmap(module, input, output):
        features_dict['heatmap'] = output.detach().cpu()

    def hook_offset(module, input, output):
        features_dict['offset'] = output.detach().cpu()

    model.upsample.register_forward_hook(hook_decoder)
    model.heatmap_head.register_forward_hook(hook_heatmap)
    model.offset_head.register_forward_hook(hook_offset)

    with torch.no_grad():
        model(image_tensor)

    return features_dict


def apply_colormap_hot_enhanced(data: np.ndarray) -> np.ndarray:
    """Hot colormap with enhanced contrast for sparse data."""
    # Use power law to enhance low values
    data_enhanced = np.power(data, 0.5)  # Square root to amplify low values

    colors = np.array([
        [0, 0, 0],
        [50, 0, 50],
        [128, 0, 0],
        [255, 0, 0],
        [255, 100, 0],
        [255, 200, 0],
        [255, 255, 150],
    ])

    indices = np.clip(data_enhanced * (len(colors) - 1), 0, len(colors) - 1.0001)
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


def apply_colormap_offset(data: np.ndarray) -> np.ndarray:
    """Diverging colormap for offset."""
    norm_data = np.clip(data, -1, 1)

    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)

    # Negative (blue)
    neg_mask = norm_data < 0
    rgb[neg_mask, 0] = (50 * (1 + norm_data[neg_mask])).astype(np.uint8)
    rgb[neg_mask, 1] = (100 * (1 + norm_data[neg_mask])).astype(np.uint8)
    rgb[neg_mask, 2] = (255 * (1 + norm_data[neg_mask])).astype(np.uint8)

    # Positive (red)
    pos_mask = norm_data > 0
    rgb[pos_mask, 0] = np.uint8(255)
    rgb[pos_mask, 1] = (100 * (1 - norm_data[pos_mask])).astype(np.uint8)
    rgb[pos_mask, 2] = (50 * (1 - norm_data[pos_mask])).astype(np.uint8)

    return rgb


def main():
    checkpoint_path = "checkpoints/final/final_model.pth"
    image_path = "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"
    output_dir = "results/diagrams"

    if not os.path.exists(checkpoint_path) or not os.path.exists(image_path):
        print("Error: Missing files")
        return

    print("Loading image...")
    image_tensor = load_image(image_path)

    print("Extracting outputs...")
    outputs = extract_outputs(checkpoint_path, image_tensor)

    os.makedirs(output_dir, exist_ok=True)

    # Decoder output
    decoder = outputs['decoder'][0].numpy()
    decoder_max = np.max(decoder, axis=0)
    decoder_norm = (decoder_max - decoder_max.min()) / (decoder_max.max() - decoder_max.min() + 1e-8)
    decoder_img = Image.fromarray((decoder_norm * 255).astype(np.uint8))
    decoder_path = os.path.join(output_dir, "head_01_decoder_output.png")
    decoder_img.save(decoder_path)
    print(f"✓ {decoder_path}")

    # Heatmap 6nm with enhanced visualization
    heatmap = outputs['heatmap'][0].numpy()
    heatmap_6nm = heatmap[0]
    print(f"6nm heatmap stats: min={heatmap_6nm.min():.4f}, max={heatmap_6nm.max():.4f}, mean={heatmap_6nm.mean():.4f}")

    heatmap_6nm_rgb = apply_colormap_hot_enhanced(heatmap_6nm)
    heatmap_6nm_img = Image.fromarray(heatmap_6nm_rgb)
    heatmap_6nm_path = os.path.join(output_dir, "head_02_heatmap_6nm.png")
    heatmap_6nm_img.save(heatmap_6nm_path)
    print(f"✓ {heatmap_6nm_path} (enhanced contrast)")

    # Heatmap 12nm with enhanced visualization
    heatmap_12nm = heatmap[1]
    print(f"12nm heatmap stats: min={heatmap_12nm.min():.4f}, max={heatmap_12nm.max():.4f}, mean={heatmap_12nm.mean():.4f}")

    heatmap_12nm_rgb = apply_colormap_hot_enhanced(heatmap_12nm)
    heatmap_12nm_img = Image.fromarray(heatmap_12nm_rgb)
    heatmap_12nm_path = os.path.join(output_dir, "head_03_heatmap_12nm.png")
    heatmap_12nm_img.save(heatmap_12nm_path)
    print(f"✓ {heatmap_12nm_path} (enhanced contrast)")

    # Offset X
    offset = outputs['offset'][0].numpy()
    offset_x = offset[0]
    offset_x_rgb = apply_colormap_offset(offset_x)
    offset_x_img = Image.fromarray(offset_x_rgb)
    offset_x_path = os.path.join(output_dir, "head_04_offset_x.png")
    offset_x_img.save(offset_x_path)
    print(f"✓ {offset_x_path}")

    # Offset Y
    offset_y = offset[1]
    offset_y_rgb = apply_colormap_offset(offset_y)
    offset_y_img = Image.fromarray(offset_y_rgb)
    offset_y_path = os.path.join(output_dir, "head_05_offset_y.png")
    offset_y_img.save(offset_y_path)
    print(f"✓ {offset_y_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
