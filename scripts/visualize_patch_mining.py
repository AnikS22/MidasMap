"""
Visualize patch-based hard mining strategy for training.
Shows how 256×256 patches are extracted with 70% hard mining + 30% random background.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def create_patch_mining_diagram():
    """Create patch-based mining strategy visualization."""

    width = 1600
    height = 1100
    margin = 50

    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font

    # Title
    title = "Patch-Based Hard Mining for Training"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 20), title, fill=(0, 0, 0), font=title_font)

    # ============================================================
    # LEFT: Full Image with Particles
    # ============================================================

    left_x = margin
    left_y = 100
    img_w = 500
    img_h = 500

    # Draw full image background
    draw.rectangle(
        [(left_x, left_y), (left_x + img_w, left_y + img_h)],
        fill=(200, 200, 200),
        outline=(0, 0, 0),
        width=2
    )
    draw.text((left_x, left_y - 35), "Full TEM Image (e.g., 2048×2048)",
              fill=(0, 0, 0), font=label_font)

    # Draw particles as small dots
    np.random.seed(42)
    num_particles = 12
    particles = []
    for _ in range(num_particles):
        px = np.random.randint(left_x + 30, left_x + img_w - 30)
        py = np.random.randint(left_y + 30, left_y + img_h - 30)
        particles.append((px, py))
        # Draw particle
        draw.ellipse([(px-4, py-4), (px+4, py+4)], fill=(255, 0, 0), outline=(200, 0, 0), width=2)

    draw.text((left_x + 10, left_y + img_h + 15), f"Contains ~50-60 particles (sparse)",
              fill=(0, 0, 0), font=small_font)

    # ============================================================
    # MIDDLE-LEFT: Hard Mining Example
    # ============================================================

    hard_x = left_x + img_w + 80
    hard_y = left_y
    patch_w = 200
    patch_h = 200

    draw.text((hard_x, hard_y - 35), "70% Hard Mining",
              fill=(0, 100, 0), font=label_font)

    # Draw example hard mining patch
    draw.rectangle(
        [(hard_x, hard_y), (hard_x + patch_w, hard_y + patch_h)],
        fill=(220, 255, 220),
        outline=(0, 150, 0),
        width=3
    )

    # Draw a particle in the center of the hard mining patch
    center_x = hard_x + patch_w // 2
    center_y = hard_y + patch_h // 2
    draw.ellipse([(center_x-5, center_y-5), (center_x+5, center_y+5)],
                 fill=(255, 0, 0), outline=(200, 0, 0), width=2)

    # Draw arrows and labels
    draw.line([(hard_x - 20, hard_y + patch_h // 2), (hard_x - 5, hard_y + patch_h // 2)],
              fill=(0, 150, 0), width=2)
    draw.text((hard_x - 70, hard_y + patch_h // 2 - 8), "Select",
              fill=(0, 100, 0), font=small_font)

    # Add explanation text
    explanations_hard = [
        "1. Randomly pick a particle",
        "2. Extract 256×256 patch",
        "   centered on particle",
        "3. Ensures every patch has",
        "   at least one particle",
        "",
        "Benefit: Model sees particles",
        "despite <0.1% density in full image"
    ]

    exp_y = hard_y + patch_h + 30
    for i, text in enumerate(explanations_hard):
        draw.text((hard_x, exp_y + i * 18), text, fill=(0, 100, 0), font=small_font)

    # ============================================================
    # MIDDLE-RIGHT: Random Background Example
    # ============================================================

    rand_x = hard_x + patch_w + 80
    rand_y = hard_y

    draw.text((rand_x, rand_y - 35), "30% Random Background",
              fill=(100, 100, 100), font=label_font)

    # Draw example random background patch (often empty)
    draw.rectangle(
        [(rand_x, rand_y), (rand_x + patch_w, rand_y + patch_h)],
        fill=(240, 240, 240),
        outline=(100, 100, 100),
        width=3
    )

    # Add a few random dots to suggest empty/sparse background
    for _ in range(2):
        rx = np.random.randint(rand_x + 10, rand_x + patch_w - 10)
        ry = np.random.randint(rand_y + 10, rand_y + patch_h - 10)
        draw.ellipse([(rx-2, ry-2), (rx+2, ry+2)], fill=(200, 200, 200))

    # Add X marking random selection
    draw.line([(rand_x + 30, rand_y + 30), (rand_x + 70, rand_y + 70)],
              fill=(100, 100, 100), width=2)
    draw.line([(rand_x + 70, rand_y + 30), (rand_x + 30, rand_y + 70)],
              fill=(100, 100, 100), width=2)

    # Add explanation text
    explanations_rand = [
        "1. Pick random location",
        "   in image",
        "2. Extract 256×256 patch",
        "3. Often contains no particles",
        "",
        "Benefit: Model learns to",
        "reduce false positives",
        "on background regions"
    ]

    for i, text in enumerate(explanations_rand):
        draw.text((rand_x, exp_y + i * 18), text, fill=(100, 100, 100), font=small_font)

    # ============================================================
    # BOTTOM LEFT: Distribution Pie Chart
    # ============================================================

    pie_x = left_x + 100
    pie_y = 700
    pie_r = 80

    draw.text((pie_x - 40, pie_y - 120), "Per-Epoch Distribution",
              fill=(0, 0, 0), font=label_font)

    # Draw pie chart (70% green, 30% gray)
    # 70% = 252° (0.7 * 360)
    # 30% = 108° (0.3 * 360)

    # Hard mining slice (70%, green)
    draw.pieslice(
        [(pie_x - pie_r, pie_y - pie_r), (pie_x + pie_r, pie_y + pie_r)],
        start=0, end=252,
        fill=(100, 200, 100),
        outline=(0, 100, 0),
        width=2
    )

    # Random background slice (30%, gray)
    draw.pieslice(
        [(pie_x - pie_r, pie_y - pie_r), (pie_x + pie_r, pie_y + pie_r)],
        start=252, end=360,
        fill=(180, 180, 180),
        outline=(100, 100, 100),
        width=2
    )

    # Add percentage labels
    draw.text((pie_x - 50, pie_y - 20), "70%", fill=(0, 100, 0), font=label_font)
    draw.text((pie_x + 30, pie_y + 20), "30%", fill=(100, 100, 100), font=label_font)

    # Legend
    legend_y = pie_y + 120
    draw.rectangle([(pie_x - 60, legend_y), (pie_x - 45, legend_y + 15)],
                   fill=(100, 200, 100), outline=(0, 100, 0), width=1)
    draw.text((pie_x - 35, legend_y - 3), "Hard Mining (particle-centered)",
              fill=(0, 100, 0), font=small_font)

    draw.rectangle([(pie_x - 60, legend_y + 25), (pie_x - 45, legend_y + 40)],
                   fill=(180, 180, 180), outline=(100, 100, 100), width=1)
    draw.text((pie_x - 35, legend_y + 22), "Random Background",
              fill=(100, 100, 100), font=small_font)

    # ============================================================
    # BOTTOM RIGHT: Training Process
    # ============================================================

    process_x = hard_x + 50
    process_y = 650

    draw.text((process_x, process_y - 50), "Training Loop (per epoch):",
              fill=(0, 0, 0), font=label_font)

    process_steps = [
        "1. Num_patches = Num_particles × 10  (e.g., 600 patches/epoch)",
        "   (sampling with replacement)",
        "",
        "2. For each patch in epoch:",
        "   • Draw random number [0, 1]",
        "   • If < 0.7 → Hard mining step",
        "   • Else → Random background step",
        "",
        "3. Forward pass + backprop on 256×256 patch",
        "",
        "4. Repeat until convergence",
        "",
        "Result: Model learns both particle detection",
        "        and false positive reduction"
    ]

    for i, text in enumerate(process_steps):
        if text.startswith("  "):
            draw.text((process_x + 30, process_y + i * 18), text, fill=(80, 80, 80), font=small_font)
        elif text == "":
            pass
        else:
            draw.text((process_x, process_y + i * 18), text, fill=(0, 0, 0), font=small_font)

    os.makedirs("results/diagrams", exist_ok=True)
    canvas.save("results/diagrams/patch_based_mining.png")
    print("✓ Saved patch_based_mining.png")


if __name__ == "__main__":
    print("Creating patch-based mining visualization...")
    create_patch_mining_diagram()
    print("Done!")
