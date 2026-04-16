"""
Visualize data augmentations with before/after examples on white background.
Shows the effect of each augmentation strategy used during training.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os


def rotate_image(img, angle):
    """Rotate image by angle (degrees)."""
    return img.rotate(angle, expand=False, resample=Image.LANCZOS)


def apply_elastic_deform(img_array, alpha=30, sigma=5):
    """Simple elastic deformation."""
    h, w = img_array.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Random displacement fields
    np.random.seed(42)
    dx = np.random.randn(h, w) * sigma
    dy = np.random.randn(h, w) * sigma

    # Smooth displacement
    from scipy import ndimage
    dx = ndimage.gaussian_filter(dx, sigma=sigma) * (alpha / 10.0)
    dy = ndimage.gaussian_filter(dy, sigma=sigma) * (alpha / 10.0)

    x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
    y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)

    # Interpolate
    deformed = np.zeros_like(img_array)
    for i in range(h):
        for j in range(w):
            xi, yi = int(x_new[i, j]), int(y_new[i, j])
            if 0 <= xi < w - 1 and 0 <= yi < h - 1:
                wx = x_new[i, j] - xi
                wy = y_new[i, j] - yi
                deformed[i, j] = (
                    (1 - wx) * (1 - wy) * img_array[yi, xi] +
                    wx * (1 - wy) * img_array[yi, xi + 1] +
                    (1 - wx) * wy * img_array[yi + 1, xi] +
                    wx * wy * img_array[yi + 1, xi + 1]
                )
            else:
                deformed[i, j] = img_array[i, j]

    return deformed.astype(np.uint8)


def create_augmentation_examples():
    """Create before/after examples for each augmentation."""

    # Load a real TEM image from the dataset
    image_path = "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load and prepare image
    img_pil = Image.open(image_path).convert('L')
    img_array = np.array(img_pil, dtype=np.uint8)

    # Resize to smaller size for visualization
    h, w = img_array.shape
    aspect = w / h
    target_h = 250
    target_w = int(target_h * aspect)
    if target_w > 500:
        target_w = 500
        target_h = int(target_w / aspect)

    img_small = img_pil.resize((target_w, target_h), Image.LANCZOS)
    img_small_array = np.array(img_small, dtype=np.uint8)

    # Create canvas with white background
    canvas_width = 1600
    canvas_height = 2200
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Load fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        aug_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        title_font = ImageFont.load_default()
        aug_font = title_font
        label_font = title_font
        small_font = title_font

    # Title
    title = "Data Augmentation Examples (Training Pipeline)"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (canvas_width - title_width) // 2
    draw.text((title_x, 20), title, fill=(0, 0, 0), font=title_font)

    # Define augmentations
    augmentations_list = [
        ("Random 90° Rotate (100%)",
         lambda x: x.rotate(90, expand=False),
         "Rotates by 0°, 90°, 180°, or 270°\nGold beads rotationally invariant"),

        ("Horizontal Flip (50%)",
         lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
         "Mirrors image left-right\nDoubles effective training samples"),

        ("Vertical Flip (50%)",
         lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
         "Mirrors image top-bottom\nSymmetry common in synapses"),

        ("Small Rotation ±10° (50%)",
         lambda x: x.rotate(8, expand=False, resample=Image.LANCZOS),
         "Fine rotation (±10°)\nAvoids interpolation artifacts"),

        ("Brightness-Contrast ±8% (70%)",
         lambda x: Image.fromarray(np.clip(np.array(x) * 1.08, 0, 255).astype(np.uint8)),
         "Mild intensity variation\nConservative limits preserve particles"),

        ("Gaussian Noise (50%)",
         lambda x: Image.fromarray(np.clip(np.array(x) + np.random.randn(*np.array(x).shape) * 15, 0, 255).astype(np.uint8)),
         "Random shot noise\nSimulates EM acquisition variation"),

        ("Gaussian Blur (20%)",
         lambda x: x.filter(ImageFilter.GaussianBlur(radius=1)),
         "Slight defocus effect\nRare but good regularization"),

        ("Copy-Paste Augmentation",
         lambda x: x,
         "Pre-extracted crops with Gaussian blending\n5 particles per class per patch"),
    ]

    y_pos = 80
    margin = 40
    example_h = 200
    spacing = 20

    for aug_name, aug_func, description in augmentations_list:
        # Section background
        section_y = y_pos
        section_h = example_h + 100

        draw.rectangle(
            [(margin, section_y), (canvas_width - margin, section_y + section_h)],
            fill=(248, 248, 248),
            outline=(200, 200, 200),
            width=1
        )

        # Augmentation name and probability
        draw.text((margin + 15, section_y + 8), aug_name, fill=(0, 0, 0), font=aug_font)

        # Original image
        canvas.paste(img_small.convert('RGB'), (margin + 15, section_y + 38))

        # Arrow
        arrow_x = margin + 15 + target_w + 20
        arrow_y = section_y + 38 + example_h // 2
        draw.text((arrow_x + 15, arrow_y - 12), "→", fill=(100, 100, 100), font=aug_font)

        # Augmented image
        try:
            aug_img = aug_func(img_small)
            if aug_img.mode == 'L':
                aug_img_rgb = aug_img.convert('RGB')
            else:
                aug_img_rgb = aug_img
        except:
            aug_img_rgb = img_small.convert('RGB')

        aug_x = arrow_x + 50
        canvas.paste(aug_img_rgb, (aug_x, section_y + 38))

        # Labels
        draw.text((margin + 15, section_y + 38 + example_h + 8), "ORIGINAL", fill=(100, 100, 100), font=label_font)
        draw.text((aug_x, section_y + 38 + example_h + 8), "AUGMENTED", fill=(100, 100, 100), font=label_font)

        # Description (right side)
        desc_x = aug_x + target_w + 30
        desc_lines = description.split('\n')
        for i, line in enumerate(desc_lines):
            draw.text((desc_x, section_y + 45 + i * 25), line, fill=(60, 60, 60), font=small_font)

        y_pos += section_h + spacing

    # Add summary at bottom
    summary_y = y_pos + 20

    draw.rectangle(
        [(margin, summary_y), (canvas_width - margin, summary_y + 160)],
        fill=(240, 248, 255),
        outline=(100, 150, 255),
        width=2
    )

    summary_title = "AUGMENTATION STRATEGY SUMMARY"
    draw.text((margin + 15, summary_y + 8), summary_title, fill=(0, 0, 150), font=aug_font)

    summary_points = [
        "• Training patches: 70% hard mining (centered near particles) + 30% random background",
        "• Geometric: Rotations (100%), flips (50%), small rotations (50%) → D4 symmetry for spherical beads",
        "• Intensity: ±8% brightness/contrast (70%), shot noise (50%), blur (20%) → EM-aware variation",
        "• Copy-paste: Pre-extracted crops with Gaussian blending, 5 per class → addresses density imbalance",
        "• Result: Robust detector generalizing to unseen synapse images with varied EM conditions",
    ]

    for i, point in enumerate(summary_points):
        draw.text((margin + 20, summary_y + 45 + i * 22), point, fill=(0, 0, 0), font=small_font)

    # Save
    os.makedirs("results/diagrams", exist_ok=True)
    canvas.save("results/diagrams/17_augmentation_pipeline.png")
    print("✓ Saved 17_augmentation_pipeline.png")
    print(f"  Size: {canvas_width}×{canvas_height} pixels")
    print(f"  Background: White")
    print(f"  Content: Before/after augmentation examples with descriptions")


if __name__ == "__main__":
    print("Creating augmentation examples...")
    create_augmentation_examples()
    print("Done!")
