"""
MidasMap — Immunogold Particle Detection Dashboard

Upload a TEM image, get instant particle detections with heatmaps,
counts, confidence distributions, and exportable CSV results.

Usage:
    python app.py
    python app.py --checkpoint checkpoints/final/final_model.pth
    python app.py --share  # public link
"""

import argparse
import io
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tifffile

from src.ensemble import sliding_window_inference
from src.heatmap import extract_peaks
from src.model import ImmunogoldCenterNet
from src.postprocess import cross_class_nms


# ---------------------------------------------------------------------------
# Global model (loaded once at startup)
# ---------------------------------------------------------------------------
MODEL = None
DEVICE = None


def load_model(checkpoint_path: str):
    global MODEL, DEVICE
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    MODEL = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"Model loaded from {checkpoint_path} on {DEVICE}")


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------
def detect_particles(
    image_file,
    conf_threshold: float = 0.25,
    nms_6nm: int = 3,
    nms_12nm: int = 5,
):
    """Run detection on uploaded image. Returns visualization + data."""
    if MODEL is None:
        return None, None, None, "Model not loaded. Start app with --checkpoint"

    # Load image
    if isinstance(image_file, str):
        img = tifffile.imread(image_file)
    elif hasattr(image_file, "name"):
        img = tifffile.imread(image_file.name)
    else:
        img = np.array(image_file)

    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[2] <= 4 else img[0]
    img = img.astype(np.uint8)

    h, w = img.shape[:2]

    # Run model
    with torch.no_grad():
        hm_np, off_np = sliding_window_inference(
            MODEL, img, patch_size=512, overlap=128, device=DEVICE,
        )

    # Extract detections
    dets = extract_peaks(
        torch.from_numpy(hm_np), torch.from_numpy(off_np),
        stride=2, conf_threshold=conf_threshold,
        nms_kernel_sizes={"6nm": nms_6nm, "12nm": nms_12nm},
    )
    dets = cross_class_nms(dets, distance_threshold=8)

    n_6nm = sum(1 for d in dets if d["class"] == "6nm")
    n_12nm = sum(1 for d in dets if d["class"] == "12nm")

    # --- Generate visualizations ---

    # 1. Detection overlay
    from skimage.transform import resize
    hm6_up = resize(hm_np[0], (h, w), order=1)
    hm12_up = resize(hm_np[1], (h, w), order=1)

    fig_overlay, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray")
    for d in dets:
        color = "#00FFFF" if d["class"] == "6nm" else "#FFD700"
        radius = 8 if d["class"] == "6nm" else 14
        circle = plt.Circle(
            (d["x"], d["y"]), radius, fill=False,
            edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(circle)
    ax.set_title(
        f"Detected: {n_6nm} 6nm (cyan) + {n_12nm} 12nm (yellow) = {len(dets)} total",
        fontsize=14, pad=10,
    )
    ax.axis("off")
    plt.tight_layout()

    # Convert to numpy for Gradio
    fig_overlay.canvas.draw()
    overlay_img = np.frombuffer(fig_overlay.canvas.tostring_rgb(), dtype=np.uint8)
    overlay_img = overlay_img.reshape(fig_overlay.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_overlay)

    # 2. Heatmap visualization
    fig_hm, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(img, cmap="gray")
    axes[0].imshow(hm6_up, cmap="hot", alpha=0.6, vmin=0, vmax=max(0.3, hm6_up.max()))
    axes[0].set_title(f"6nm Heatmap ({n_6nm} particles)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(hm12_up, cmap="YlOrRd", alpha=0.6, vmin=0, vmax=max(0.3, hm12_up.max()))
    axes[1].set_title(f"12nm Heatmap ({n_12nm} particles)", fontsize=13)
    axes[1].axis("off")
    plt.tight_layout()

    fig_hm.canvas.draw()
    heatmap_img = np.frombuffer(fig_hm.canvas.tostring_rgb(), dtype=np.uint8)
    heatmap_img = heatmap_img.reshape(fig_hm.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_hm)

    # 3. Stats dashboard
    fig_stats, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confidence histogram
    if dets:
        confs_6 = [d["conf"] for d in dets if d["class"] == "6nm"]
        confs_12 = [d["conf"] for d in dets if d["class"] == "12nm"]
        if confs_6:
            axes[0].hist(confs_6, bins=20, alpha=0.7, color="#00CCCC", label=f"6nm (n={len(confs_6)})")
        if confs_12:
            axes[0].hist(confs_12, bins=20, alpha=0.7, color="#CCB300", label=f"12nm (n={len(confs_12)})")
        axes[0].axvline(conf_threshold, color="red", linestyle="--", label=f"Threshold={conf_threshold}")
        axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Detection Confidence Distribution")

    # Spatial distribution
    if dets:
        xs = [d["x"] for d in dets]
        ys = [d["y"] for d in dets]
        colors = ["#00CCCC" if d["class"] == "6nm" else "#CCB300" for d in dets]
        axes[1].scatter(xs, ys, c=colors, s=20, alpha=0.7)
        axes[1].set_xlim(0, w)
        axes[1].set_ylim(h, 0)
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    axes[1].set_title("Spatial Distribution")
    axes[1].set_aspect("equal")

    # Summary table
    axes[2].axis("off")
    table_data = [
        ["Image size", f"{w} x {h} px"],
        ["Scale", "1790 px/\u00b5m"],
        ["6nm (AMPA)", str(n_6nm)],
        ["12nm (NR1)", str(n_12nm)],
        ["Total", str(len(dets))],
        ["Threshold", f"{conf_threshold:.2f}"],
        ["Mean conf (6nm)", f"{np.mean(confs_6):.3f}" if confs_6 else "N/A"],
        ["Mean conf (12nm)", f"{np.mean(confs_12):.3f}" if confs_12 else "N/A"],
    ]
    table = axes[2].table(
        cellText=table_data, colLabels=["Metric", "Value"],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    axes[2].set_title("Detection Summary")
    plt.tight_layout()

    fig_stats.canvas.draw()
    stats_img = np.frombuffer(fig_stats.canvas.tostring_rgb(), dtype=np.uint8)
    stats_img = stats_img.reshape(fig_stats.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_stats)

    # 4. CSV export
    df = pd.DataFrame([
        {
            "particle_id": i + 1,
            "x_px": round(d["x"], 1),
            "y_px": round(d["y"], 1),
            "x_um": round(d["x"] / 1790, 4),
            "y_um": round(d["y"] / 1790, 4),
            "class": d["class"],
            "confidence": round(d["conf"], 4),
        }
        for i, d in enumerate(dets)
    ])

    csv_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(csv_path.name, index=False)

    summary = (
        f"## Results\n"
        f"- **6nm (AMPA)**: {n_6nm} particles\n"
        f"- **12nm (NR1)**: {n_12nm} particles\n"
        f"- **Total**: {len(dets)} particles\n"
        f"- **Image**: {w}x{h} px\n"
    )

    return overlay_img, heatmap_img, stats_img, csv_path.name, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks(
        title="MidasMap - Immunogold Particle Detection",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# MidasMap\n"
            "### Immunogold Particle Detection for TEM Synapse Images\n"
            "Upload an EM image (.tif) to detect 6nm (AMPA) and 12nm (NR1) gold particles."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.File(
                    label="Upload TEM Image (.tif)",
                    file_types=[".tif", ".tiff", ".png", ".jpg"],
                )
                conf_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                    label="Confidence Threshold",
                    info="Lower = more detections (more FP), Higher = fewer but more certain",
                )
                nms_6nm = gr.Slider(
                    minimum=1, maximum=9, value=3, step=2,
                    label="NMS Kernel (6nm)",
                    info="Min distance between 6nm detections (pixels at stride 2)",
                )
                nms_12nm = gr.Slider(
                    minimum=1, maximum=9, value=5, step=2,
                    label="NMS Kernel (12nm)",
                )
                detect_btn = gr.Button("Detect Particles", variant="primary", size="lg")

            with gr.Column(scale=2):
                summary_md = gr.Markdown("Upload an image to begin.")

        with gr.Tabs():
            with gr.TabItem("Detection Overlay"):
                overlay_output = gr.Image(label="Detected Particles")
            with gr.TabItem("Heatmaps"):
                heatmap_output = gr.Image(label="Class Heatmaps")
            with gr.TabItem("Statistics"):
                stats_output = gr.Image(label="Detection Statistics")
            with gr.TabItem("Export"):
                csv_output = gr.File(label="Download CSV Results")

        detect_btn.click(
            fn=detect_particles,
            inputs=[image_input, conf_slider, nms_6nm, nms_12nm],
            outputs=[overlay_output, heatmap_output, stats_output, csv_output, summary_md],
        )

        gr.Markdown(
            "---\n"
            "*MidasMap: CenterNet + CEM500K backbone, trained on 453 labeled particles "
            "across 10 synapses. LOOCV F1 = 0.94.*"
        )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="checkpoints/local_S1_v2/best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    load_model(args.checkpoint)
    app = build_app()
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
