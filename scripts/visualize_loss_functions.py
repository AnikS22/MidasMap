"""
Visualize the actual loss functions used for training.
Shows CornerNet focal loss for heatmaps and smooth L1 for offsets.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def create_loss_visualization():
    """Create comprehensive loss function diagrams."""

    width = 1600
    height = 1000
    margin = 60

    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font

    # Title
    title = "MidasMap Loss Functions"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 20), title, fill=(0, 0, 0), font=title_font)

    # ============================================================
    # LEFT: CornerNet Focal Loss for Heatmaps
    # ============================================================

    left_x = margin
    left_y = 80
    plot_w = 650
    plot_h = 350

    # Plot background
    draw.rectangle(
        [(left_x, left_y), (left_x + plot_w, left_y + plot_h)],
        fill=(240, 245, 250),
        outline=(100, 100, 100),
        width=2
    )

    draw.text((left_x + 20, left_y - 30), "CornerNet Focal Loss (Heatmap)",
              fill=(0, 0, 0), font=label_font)

    # Generate data for focal loss
    pred = np.linspace(0.001, 0.999, 100)

    # Parameters
    alpha = 2
    beta = 4
    gt_pos = 1.0  # Particle center
    gt_neg = 0.0  # Background

    # Positive loss: -log(pred) * (1-pred)^alpha
    pos_loss = -np.log(pred) * np.power(1 - pred, alpha)

    # Negative loss: -log(1-pred) * pred^alpha * (1-gt)^beta
    # For background (gt=0): (1-0)^beta = 1.0
    neg_loss = -np.log(1 - pred + 1e-6) * np.power(pred, alpha)

    # Penalty reduction (near peaks)
    penalty_reduction = np.power(1 - gt_neg, beta)  # For bg: 1.0
    neg_loss_reduced = neg_loss * penalty_reduction

    # Scale for plotting
    scale_x = plot_w / (pred[-1] - pred[0])
    scale_y = plot_h / (max(np.max(pos_loss), np.max(neg_loss)) * 1.1)

    # Plot curves
    color_pos = (0, 150, 0)      # Green
    color_neg = (200, 0, 0)      # Red
    color_neg_reduced = (255, 150, 0)  # Orange

    # Draw positive loss
    for i in range(len(pred) - 1):
        x1 = left_x + (pred[i] - pred[0]) * scale_x
        y1 = left_y + plot_h - pos_loss[i] * scale_y
        x2 = left_x + (pred[i+1] - pred[0]) * scale_x
        y2 = left_y + plot_h - pos_loss[i+1] * scale_y
        draw.line([(x1, y1), (x2, y2)], fill=color_pos, width=3)

    # Draw negative loss
    for i in range(len(pred) - 1):
        x1 = left_x + (pred[i] - pred[0]) * scale_x
        y1 = left_y + plot_h - neg_loss[i] * scale_y
        x2 = left_x + (pred[i+1] - pred[0]) * scale_x
        y2 = left_y + plot_h - neg_loss[i+1] * scale_y
        draw.line([(x1, y1), (x2, y2)], fill=color_neg, width=2)

    # Draw negative loss reduced
    for i in range(len(pred) - 1):
        x1 = left_x + (pred[i] - pred[0]) * scale_x
        y1 = left_y + plot_h - neg_loss_reduced[i] * scale_y
        x2 = left_x + (pred[i+1] - pred[0]) * scale_x
        y2 = left_y + plot_h - neg_loss_reduced[i+1] * scale_y
        draw.line([(x1, y1), (x2, y2)], fill=color_neg_reduced, width=2)

    # Axes
    draw.line([(left_x, left_y + plot_h), (left_x + plot_w, left_y + plot_h)],
              fill=(0, 0, 0), width=2)
    draw.line([(left_x, left_y), (left_x, left_y + plot_h)],
              fill=(0, 0, 0), width=2)

    # Axis labels
    draw.text((left_x + plot_w - 50, left_y + plot_h + 10), "Prediction",
              fill=(0, 0, 0), font=small_font)
    draw.text((left_x - 40, left_y - 10), "Loss",
              fill=(0, 0, 0), font=small_font)

    # Legend for left plot
    legend_y = left_y + plot_h + 40
    draw.line([(left_x + 20, legend_y), (left_x + 50, legend_y)],
              fill=color_pos, width=3)
    draw.text((left_x + 60, legend_y - 8), "Positive (GT=1): -log(p)(1-p)²",
              fill=(0, 0, 0), font=small_font)

    draw.line([(left_x + 20, legend_y + 20), (left_x + 50, legend_y + 20)],
              fill=color_neg, width=2)
    draw.text((left_x + 60, legend_y + 12), "Negative (GT=0): -log(1-p)p²",
              fill=(0, 0, 0), font=small_font)

    draw.line([(left_x + 20, legend_y + 40), (left_x + 50, legend_y + 40)],
              fill=color_neg_reduced, width=2)
    draw.text((left_x + 60, legend_y + 32), "Negative with penalty reduction",
              fill=(0, 0, 0), font=small_font)

    # ============================================================
    # RIGHT: Smooth L1 Loss for Offsets
    # ============================================================

    right_x = left_x + plot_w + 100
    right_y = left_y

    # Plot background
    draw.rectangle(
        [(right_x, right_y), (right_x + plot_w, right_y + plot_h)],
        fill=(240, 245, 250),
        outline=(100, 100, 100),
        width=2
    )

    draw.text((right_x + 20, right_y - 30), "Smooth L1 Loss (Offset)",
              fill=(0, 0, 0), font=label_font)

    # Generate data for smooth L1
    delta = np.linspace(-2, 2, 200)
    beta_smooth = 1.0  # Standard smooth L1 beta

    smooth_l1 = np.where(
        np.abs(delta) < beta_smooth,
        0.5 * delta ** 2 / beta_smooth,
        np.abs(delta) - 0.5 * beta_smooth
    )

    l1 = np.abs(delta)
    l2 = delta ** 2

    # Scale for plotting
    scale_x_r = plot_w / (delta[-1] - delta[0])
    scale_y_r = plot_h / (np.max(l1) * 1.1)

    # Draw smooth L1 (blue)
    color_smooth = (0, 100, 255)
    for i in range(len(delta) - 1):
        x1 = right_x + (delta[i] - delta[0]) * scale_x_r
        y1 = right_y + plot_h - smooth_l1[i] * scale_y_r
        x2 = right_x + (delta[i+1] - delta[0]) * scale_x_r
        y2 = right_y + plot_h - smooth_l1[i+1] * scale_y_r
        draw.line([(x1, y1), (x2, y2)], fill=color_smooth, width=3)

    # Draw L1 (red)
    color_l1 = (200, 0, 0)
    for i in range(len(delta) - 1):
        x1 = right_x + (delta[i] - delta[0]) * scale_x_r
        y1 = right_y + plot_h - l1[i] * scale_y_r
        x2 = right_x + (delta[i+1] - delta[0]) * scale_x_r
        y2 = right_y + plot_h - l1[i+1] * scale_y_r
        draw.line([(x1, y1), (x2, y2)], fill=color_l1, width=2)

    # Axes
    draw.line([(right_x, right_y + plot_h), (right_x + plot_w, right_y + plot_h)],
              fill=(0, 0, 0), width=2)
    draw.line([(right_x, right_y), (right_x, right_y + plot_h)],
              fill=(0, 0, 0), width=2)

    # Axis labels
    draw.text((right_x + plot_w - 80, right_y + plot_h + 10), "Prediction Error",
              fill=(0, 0, 0), font=small_font)
    draw.text((right_x - 40, right_y - 10), "Loss",
              fill=(0, 0, 0), font=small_font)

    # Legend for right plot
    legend_y = right_y + plot_h + 40
    draw.line([(right_x + 20, legend_y), (right_x + 50, legend_y)],
              fill=color_smooth, width=3)
    draw.text((right_x + 60, legend_y - 8), "Smooth L1 (robust to outliers)",
              fill=(0, 0, 0), font=small_font)

    draw.line([(right_x + 20, legend_y + 20), (right_x + 50, legend_y + 20)],
              fill=color_l1, width=2)
    draw.text((right_x + 60, legend_y + 12), "L1 (less robust to outliers)",
              fill=(0, 0, 0), font=small_font)

    # ============================================================
    # BOTTOM: Loss Function Explanation
    # ============================================================

    explain_y = left_y + plot_h + 120

    explanations = [
        "HEATMAP LOSS (CornerNet Focal Loss):",
        "  • Positive: -log(pred) × (1-pred)² rewards high confidence at particle centers (GT=1)",
        "  • Negative: -log(1-pred) × pred² × (1-GT)⁴ penalizes high confidence away from particles",
        "  • Penalty reduction: (1-GT)⁴ reduces penalty near particle peaks, focuses on hard negatives",
        "  • Reason: 23,000:1 negative:positive ratio → standard BCE learns all-zeros",
        "",
        "OFFSET LOSS (Smooth L1):",
        "  • Quadratic for small errors (|error| < 1): 0.5 × error²",
        "  • Linear for large errors (|error| ≥ 1): |error| - 0.5",
        "  • Reason: Robust to outliers (unlike L2), sensitive to small errors (unlike L1)",
        "  • Achieves ±0.5 pixel sub-pixel accuracy via Gaussian peak extraction",
    ]

    for i, text in enumerate(explanations):
        if text.startswith("  "):
            draw.text((margin + 30, explain_y + i * 18), text, fill=(80, 80, 80), font=small_font)
        elif text == "":
            pass
        else:
            draw.text((margin, explain_y + i * 18), text, fill=(0, 0, 0), font=label_font)

    os.makedirs("results/diagrams", exist_ok=True)
    canvas.save("results/diagrams/19_loss_functions.png")
    print("✓ Saved 19_loss_functions.png")


if __name__ == "__main__":
    print("Creating loss function visualization...")
    create_loss_visualization()
    print("Done!")
