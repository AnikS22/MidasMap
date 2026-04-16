"""
Generate individual augmentation examples.
Each augmentation saved as a separate PNG showing the actual effect applied.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os


def add_label(img, label_text, font_size=20):
    """Add label text to image."""
    img_with_label = img.copy()
    draw = ImageDraw.Draw(img_with_label)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    # Add semi-transparent background for text
    bbox = draw.textbbox((10, 10), label_text, font=font)
    x0, y0, x1, y1 = bbox

    overlay = Image.new('RGBA', img_with_label.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(x0-5, y0-5), (x1+5, y1+5)], fill=(0, 0, 0, 180))

    img_with_label = Image.alpha_composite(img_with_label.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img_with_label)
    draw.text((10, 10), label_text, fill=(255, 255, 255), font=font)

    return img_with_label


def generate_augmentation_examples():
    """Generate individual augmentation examples."""

    image_path = "Max Planck Data/Gold Particle Labelling/analyzed synapses/S4/S4 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S4.tif"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load image
    img_pil = Image.open(image_path).convert('L')
    img_array = np.array(img_pil, dtype=np.uint8)

    # Resize for display
    h, w = img_array.shape
    aspect = w / h
    target_h = 600
    target_w = int(target_h * aspect)
    if target_w > 1200:
        target_w = 1200
        target_h = int(target_w / aspect)

    img_small = img_pil.resize((target_w, target_h), Image.BICUBIC)

    os.makedirs("results/diagrams/augmentations", exist_ok=True)

    # 1. Random 90° Rotation
    aug_rotate90 = img_small.rotate(90, expand=False)
    aug_rotate90 = add_label(aug_rotate90, "Random 90° Rotation (100%)")
    aug_rotate90.save("results/diagrams/augmentations/18_rotate_90deg.png")
    print("✓ 18_rotate_90deg.png")

    # 2. Horizontal Flip
    aug_hflip = img_small.transpose(Image.FLIP_LEFT_RIGHT)
    aug_hflip = add_label(aug_hflip, "Horizontal Flip (50%)")
    aug_hflip.save("results/diagrams/augmentations/19_horizontal_flip.png")
    print("✓ 19_horizontal_flip.png")

    # 3. Vertical Flip
    aug_vflip = img_small.transpose(Image.FLIP_TOP_BOTTOM)
    aug_vflip = add_label(aug_vflip, "Vertical Flip (50%)")
    aug_vflip.save("results/diagrams/augmentations/20_vertical_flip.png")
    print("✓ 20_vertical_flip.png")

    # 4. Small Rotation ±10°
    aug_small_rotate = img_small.rotate(8, expand=False, resample=Image.BICUBIC)
    aug_small_rotate = add_label(aug_small_rotate, "Small Rotation ±10° (50%)")
    aug_small_rotate.save("results/diagrams/augmentations/21_small_rotation.png")
    print("✓ 21_small_rotation.png")

    # 5. Brightness Increase
    img_arr = np.array(img_small, dtype=np.float32)
    brightness_aug = np.clip(img_arr * 1.15, 0, 255).astype(np.uint8)
    aug_bright = Image.fromarray(brightness_aug, mode='L').convert('RGB')
    aug_bright = add_label(aug_bright, "Brightness-Contrast +8% (70%)")
    aug_bright.save("results/diagrams/augmentations/22_brightness_contrast.png")
    print("✓ 22_brightness_contrast.png")

    # 6. Gaussian Noise
    np.random.seed(42)
    img_arr = np.array(img_small, dtype=np.float32)
    noise = np.random.normal(0, 20, img_arr.shape)
    noise_aug = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    aug_noise = Image.fromarray(noise_aug, mode='L').convert('RGB')
    aug_noise = add_label(aug_noise, "Gaussian Noise / Shot Noise (50%)")
    aug_noise.save("results/diagrams/augmentations/23_gaussian_noise.png")
    print("✓ 23_gaussian_noise.png")

    # 7. Gaussian Blur
    aug_blur = img_small.filter(ImageFilter.GaussianBlur(radius=2))
    aug_blur = add_label(aug_blur, "Gaussian Blur (20%)")
    aug_blur.save("results/diagrams/augmentations/24_gaussian_blur.png")
    print("✓ 24_gaussian_blur.png")

    # 8. Elastic Deformation (simple sine wave distortion)
    img_arr = np.array(img_small, dtype=np.uint8)
    h, w = img_arr.shape

    # Create distorted image with sine wave effect
    elastic_aug = np.zeros_like(img_arr)
    amplitude = 10
    frequency = 0.02

    for y in range(h):
        offset = int(amplitude * np.sin(y * frequency * 2 * np.pi))
        for x in range(w):
            src_x = (x + offset) % w
            elastic_aug[y, x] = img_arr[y, src_x]

    aug_elastic = Image.fromarray(elastic_aug, mode='L').convert('RGB')
    aug_elastic = add_label(aug_elastic, "Elastic Deformation (30%)")
    aug_elastic.save("results/diagrams/augmentations/25_elastic_deformation.png")
    print("✓ 25_elastic_deformation.png")

    # 9. Combined augmentation example
    # Apply multiple augmentations together
    combined = img_small.rotate(5, expand=False, resample=Image.BICUBIC)
    combined = combined.transpose(Image.FLIP_LEFT_RIGHT)
    img_arr = np.array(combined, dtype=np.float32)
    img_arr = np.clip(img_arr * 1.1, 0, 255).astype(np.uint8)
    combined = Image.fromarray(img_arr, mode='L').convert('RGB')
    combined = add_label(combined, "Combined: Rotation + Flip + Brightness (Training Example)")
    combined.save("results/diagrams/augmentations/26_combined_example.png")
    print("✓ 26_combined_example.png")

    print("\nAll augmentation examples generated!")
    print(f"Saved to: results/diagrams/augmentations/")


if __name__ == "__main__":
    print("Generating individual augmentation examples...")
    generate_augmentation_examples()
