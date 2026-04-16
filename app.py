"""
MidasMap — Immunogold particle analysis for FFRIL / TEM synapse imaging

Web UI for neuroscientists: calibrated coordinates (µm), receptor labels,
export for quantification, and clear interpretation of model limits.

Usage:
    python app.py
    python app.py --checkpoint checkpoints/final/final_model.pth
    python app.py --share
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import gradio as gr
import gradio_client.utils as _gcu

# Pydantic v2 can emit JSON Schema with additionalProperties: true (bool);
# Gradio 4.4x gradio_client assumes a dict and crashes rendering "/".
_orig_json_type = _gcu._json_schema_to_python_type


def _json_schema_to_python_type_safe(schema, defs=None):
    if schema is True or schema is False:
        return "Any"
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_type(schema, defs)


_gcu._json_schema_to_python_type = _json_schema_to_python_type_safe

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import tifffile

from src.ensemble import sliding_window_inference
from src.heatmap import extract_peaks
from src.model import ImmunogoldCenterNet
from src.postprocess import cross_class_nms


# Calibration used for training / published metrics (change in UI if your scope differs)
DEFAULT_PX_PER_UM = 1790.0

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "figure.dpi": 120,
        "savefig.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#1e293b",
        "axes.titlecolor": "#0f172a",
        "axes.grid": False,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#e2e8f0",
    }
)


MODEL = None
DEVICE = None


def load_model(checkpoint_path: str):
    global MODEL, DEVICE
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    MODEL = ImmunogoldCenterNet(
        bifpn_channels=128,
        bifpn_rounds=2,
        imagenet_encoder_fallback=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"Model loaded from {checkpoint_path} on {DEVICE}")


def _receptor_label(class_name: str) -> str:
    return "AMPA receptor" if class_name == "6nm" else "NR1 (NMDA receptor)"


def _gold_nm(class_name: str) -> int:
    return 6 if class_name == "6nm" else 12


def _pick_scale_bar_um(field_width_um: float) -> float:
    """Pick a readable scale bar (~15–30% of field width)."""
    if field_width_um <= 0:
        return 0.2
    target = field_width_um * 0.22
    candidates = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0)
    best = candidates[0]
    for c in candidates:
        if abs(c - target) < abs(best - target):
            best = c
    # Keep bar from dominating the field
    while best > 0 and best / field_width_um > 0.45:
        best = max(0.05, best / 2)
    return float(best)


def _draw_scale_bar_um(ax, w: int, h: int, px_per_um: float) -> None:
    field_um = max(w, h) / px_per_um
    bar_um = _pick_scale_bar_um(field_um)
    bar_px = bar_um * px_per_um
    margin = max(12, int(min(w, h) * 0.025))
    y_line = h - margin
    x0, x1 = margin, margin + bar_px
    for lw, color in ((5, "white"), (2, "#0f172a")):
        ax.plot([x0, x1], [y_line, y_line], color=color, linewidth=lw, solid_capstyle="butt", clip_on=False)
    t = ax.text(
        (x0 + x1) / 2,
        y_line - margin * 0.35,
        f"{bar_um:g} µm",
        ha="center",
        va="bottom",
        color="white",
        fontsize=9,
        fontweight="600",
    )
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#0f172a")])


def _export_columns() -> list[str]:
    return [
        "particle_id",
        "receptor",
        "gold_diameter_nm",
        "x_px",
        "y_px",
        "x_um",
        "y_um",
        "confidence",
        "class_model",
        "calibration_px_per_um",
    ]


def _empty_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_export_columns())


def _df_to_preview_html(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "<p class='mm-table-empty'><em>No particles above the current threshold.</em></p>"
    return df.to_html(
        classes=["mm-table"],
        index=False,
        border=0,
        justify="left",
        escape=True,
    )


def _numpy_image_to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Normalize various arrays to HxWx3 uint8 for cropping / display."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype in (np.float32, np.float64):
        mx = float(arr.max()) if arr.size else 1.0
        if mx <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def magnifier_zoom(
    store: dict,
    view: str,
    center_x_pct: float,
    center_y_pct: float,
    zoom: float,
    output_px: int,
) -> np.ndarray | None:
    """
    Crop a square region around (center_x_pct, center_y_pct) and upscale for a loupe view.
    zoom: 1 = see ~full width in loupe; larger = stronger magnification (smaller crop).
    """
    if not store or not isinstance(store, dict):
        return None
    key = {"Overlay": "overlay", "Heatmaps": "heatmap", "Summary": "stats"}.get(view, "overlay")
    img = _numpy_image_to_uint8_rgb(store.get(key))
    if img is None:
        return None
    h, w = img.shape[:2]
    cx = int(np.clip(center_x_pct / 100.0 * (w - 1), 0, w - 1))
    cy = int(np.clip(center_y_pct / 100.0 * (h - 1), 0, h - 1))
    z = max(1.0, float(zoom))
    half_w = max(1, int(w / (2.0 * z)))
    half_h = max(1, int(h / (2.0 * z)))
    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
    y0, y1 = max(0, cy - half_h), min(h, cy + half_h)
    if x1 <= x0 or y1 <= y0:
        crop = img
    else:
        crop = img[y0:y1, x0:x1]
    side = int(np.clip(output_px, 256, 1024))
    try:
        from PIL import Image as PILImage

        pil = PILImage.fromarray(crop)
        pil = pil.resize((side, side), PILImage.Resampling.LANCZOS)
        return np.asarray(pil)
    except Exception:
        from skimage.transform import resize

        up = resize(crop, (side, side), order=1, preserve_range=True)
        return np.clip(up, 0, 255).astype(np.uint8)


def run_detection(
    image_file,
    conf_threshold: float,
    nms_6nm: int,
    nms_12nm: int,
    px_per_um: float,
    progress=gr.Progress(track_tqdm=False),
):
    """Run model and return outputs plus viz state for the magnifier."""
    out = detect_particles(
        image_file,
        conf_threshold,
        nms_6nm,
        nms_12nm,
        px_per_um,
        progress=progress,
    )
    overlay, hm, stats, csvp, table, summary = out
    store = {"overlay": overlay, "heatmap": hm, "stats": stats}
    return overlay, hm, stats, csvp, table, summary, store


def detect_particles(
    image_file,
    conf_threshold: float = 0.25,
    nms_6nm: int = 3,
    nms_12nm: int = 5,
    px_per_um: float = DEFAULT_PX_PER_UM,
    progress=gr.Progress(track_tqdm=False),
):
    """Run detection; returns figures, CSV path, table HTML, and summary HTML."""
    empty_table = "<p class='mm-table-empty'><em>Run detection to populate the table.</em></p>"

    if MODEL is None:
        msg = "<p class='mm-callout mm-callout-warn'>Model not loaded. Use <code>--checkpoint</code> with a valid <code>.pth</code> file.</p>"
        return None, None, None, None, empty_table, msg

    if image_file is None:
        msg = "<p class='mm-callout'>Upload a micrograph, set calibration if needed, then run detection.</p>"
        return None, None, None, None, empty_table, msg

    try:
        px_per_um = float(px_per_um)
    except (TypeError, ValueError):
        px_per_um = DEFAULT_PX_PER_UM
    if px_per_um <= 0:
        px_per_um = DEFAULT_PX_PER_UM

    progress(0.05, desc="Loading image…")

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
    field_w_um = w / px_per_um
    field_h_um = h / px_per_um

    progress(0.15, desc="Neural network (sliding window)…")

    with torch.no_grad():
        hm_np, off_np = sliding_window_inference(
            MODEL,
            img,
            patch_size=512,
            overlap=128,
            device=DEVICE,
        )

    progress(0.72, desc="Peak extraction & NMS…")

    dets = extract_peaks(
        torch.from_numpy(hm_np),
        torch.from_numpy(off_np),
        stride=2,
        conf_threshold=conf_threshold,
        nms_kernel_sizes={"6nm": nms_6nm, "12nm": nms_12nm},
    )
    dets = cross_class_nms(dets, distance_threshold=8)

    n_6nm = sum(1 for d in dets if d["class"] == "6nm")
    n_12nm = sum(1 for d in dets if d["class"] == "12nm")
    confs_6 = [d["conf"] for d in dets if d["class"] == "6nm"]
    confs_12 = [d["conf"] for d in dets if d["class"] == "12nm"]

    progress(0.78, desc="Rendering figures…")

    from skimage.transform import resize

    hm6_up = np.clip(
        np.nan_to_num(resize(hm_np[0], (h, w), order=1), nan=0.0),
        0.0,
        1.0,
    )
    hm12_up = np.clip(
        np.nan_to_num(resize(hm_np[1], (h, w), order=1), nan=0.0),
        0.0,
        1.0,
    )

    def _heatmap_vmax(hm: np.ndarray) -> float:
        """Stable color scale: avoid invisible overlays when max is tiny or flat."""
        flat = hm.ravel()
        if flat.size == 0:
            return 0.3
        mx = float(np.max(flat))
        if mx < 1e-6:
            return 0.3
        p99 = float(np.percentile(flat, 99.0))
        return float(np.clip(max(0.12, p99 * 1.05, mx * 0.95), 0.05, 1.0))

    # --- Overlay (publication-style legend + scale bar) ---
    fig_overlay, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(img, cmap="gray", aspect="equal")
    for d in dets:
        color = "#06b6d4" if d["class"] == "6nm" else "#ca8a04"
        radius = 7 if d["class"] == "6nm" else 12
        ax.add_patch(
            plt.Circle(
                (d["x"], d["y"]),
                radius,
                fill=False,
                edgecolor=color,
                linewidth=1.8,
            )
        )
    _draw_scale_bar_um(ax, w, h, px_per_um)
    ax.set_title(
        f"Immunogold detections · AMPA (6 nm): {n_6nm} · NR1 (12 nm): {n_12nm} · Total: {len(dets)}",
        fontsize=11,
        pad=12,
    )
    ax.axis("off")
    legend_elems = [
        Patch(facecolor="none", edgecolor="#06b6d4", linewidth=2, label="6 nm gold — AMPA receptor"),
        Patch(facecolor="none", edgecolor="#ca8a04", linewidth=2, label="12 nm gold — NR1 (NMDAR)"),
    ]
    ax.legend(
        handles=legend_elems,
        loc="upper right",
        fontsize=8.5,
        title="Label class",
        title_fontsize=9,
    )
    plt.tight_layout()
    fig_overlay.canvas.draw()
    overlay_img = np.asarray(fig_overlay.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig_overlay)

    # --- Heatmaps: row1 = overlay on EM; row2 = model heat only (debug-friendly) ---
    # Training uses Gaussian GT; inference heatmaps are learned sigmoid blobs, not analytic Gaussians.
    v6, v12 = _heatmap_vmax(hm6_up), _heatmap_vmax(hm12_up)
    fig_hm, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    for ax, hm, v, cmap, title in (
        (ax00, hm6_up, v6, "magma", f"AMPA overlay · n={n_6nm} · vmax={v6:.2f}"),
        (ax01, hm12_up, v12, "inferno", f"NR1 overlay · n={n_12nm} · vmax={v12:.2f}"),
    ):
        ax.imshow(img, cmap="gray", aspect="equal", interpolation="nearest")
        ax.imshow(
            hm,
            cmap=cmap,
            alpha=0.6,
            vmin=0.0,
            vmax=v,
            interpolation="bilinear",
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax10.imshow(hm6_up, cmap="magma", vmin=0.0, vmax=v6, interpolation="nearest")
    ax10.set_title(f"AMPA heatmap only · max={float(np.max(hm6_up)):.4f}", fontsize=10)
    ax10.axis("off")

    ax11.imshow(hm12_up, cmap="inferno", vmin=0.0, vmax=v12, interpolation="nearest")
    ax11.set_title(f"NR1 heatmap only · max={float(np.max(hm12_up)):.4f}", fontsize=10)
    ax11.axis("off")

    plt.tight_layout()
    # PNG raster → uint8 RGB (reliable in Gradio vs raw canvas buffer on some setups)
    from io import BytesIO

    _buf = BytesIO()
    fig_hm.savefig(_buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig_hm)
    _buf.seek(0)
    try:
        from PIL import Image as _PILImage

        heatmap_img = np.asarray(_PILImage.open(_buf).convert("RGB"))
    except Exception:
        import matplotlib.image as _mimg

        _buf.seek(0)
        heatmap_img = (_mimg.imread(_buf)[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)

    # --- Stats (µm where helpful) ---
    fig_stats, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    if dets:
        if confs_6:
            axes[0].hist(confs_6, bins=18, alpha=0.75, color="#0891b2", label=f"AMPA (n={len(confs_6)})")
        if confs_12:
            axes[0].hist(confs_12, bins=18, alpha=0.75, color="#a16207", label=f"NR1 (n={len(confs_12)})")
        axes[0].axvline(conf_threshold, color="#be123c", linestyle="--", linewidth=1.2, label=f"Threshold = {conf_threshold:.2f}")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Confidence score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score distribution")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    if dets:
        xs_um = np.array([d["x"] for d in dets]) / px_per_um
        ys_um = np.array([d["y"] for d in dets]) / px_per_um
        colors = ["#0891b2" if d["class"] == "6nm" else "#a16207" for d in dets]
        axes[1].scatter(xs_um, ys_um, c=colors, s=22, alpha=0.75, edgecolors="none")
        axes[1].set_xlim(0, field_w_um)
        axes[1].set_ylim(field_h_um, 0)
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("y (µm)")
    axes[1].set_title("Positions (image coordinates)")
    axes[1].set_aspect("equal")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].axis("off")
    table_data = [
        ["Field of view", f"{field_w_um:.3f} × {field_h_um:.3f} µm"],
        ["Calibration", f"{px_per_um:.1f} px/µm"],
        ["AMPA (6 nm)", str(n_6nm)],
        ["NR1 (12 nm)", str(n_12nm)],
        ["Total particles", str(len(dets))],
        ["Score threshold", f"{conf_threshold:.2f}"],
        ["Mean score · AMPA", f"{float(np.mean(confs_6)):.3f}" if confs_6 else "—"],
        ["Mean score · NR1", f"{float(np.mean(confs_12)):.3f}" if confs_12 else "—"],
    ]
    tbl = axes[2].table(
        cellText=table_data,
        colLabels=["Quantity", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.05, 1.65)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="600")
            cell.set_facecolor("#e2e8f0")
    axes[2].set_title("Summary", fontsize=11, pad=12)
    plt.tight_layout()
    fig_stats.canvas.draw()
    stats_img = np.asarray(fig_stats.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig_stats)

    rows = []
    for i, d in enumerate(dets):
        rows.append(
            {
                "particle_id": i + 1,
                "receptor": _receptor_label(d["class"]),
                "gold_diameter_nm": _gold_nm(d["class"]),
                "x_px": round(d["x"], 2),
                "y_px": round(d["y"], 2),
                "x_um": round(d["x"] / px_per_um, 5),
                "y_um": round(d["y"] / px_per_um, 5),
                "confidence": round(d["conf"], 4),
                "class_model": d["class"],
                "calibration_px_per_um": round(px_per_um, 4),
            }
        )
    df = pd.DataFrame(rows, columns=_export_columns()) if rows else _empty_results_df()

    csv_f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    df.to_csv(csv_f.name, index=False)
    csv_f.close()

    progress(1.0, desc="Done")

    density_note = ""
    if field_w_um > 0 and field_h_um > 0:
        area = field_w_um * field_h_um
        density_note = f"<span class='mm-density'>Areal density (all): {len(dets) / area:.2f} particles/µm² · AMPA: {n_6nm / area:.2f} · NR1: {n_12nm / area:.2f}</span>"

    summary = f"""<div class="mm-summary">
<div class="mm-stat"><span class="mm-stat-label">AMPA · 6 nm gold</span>
<span class="mm-stat-value mm-teal">{n_6nm}</span></div>
<div class="mm-stat"><span class="mm-stat-label">NR1 · 12 nm gold</span>
<span class="mm-stat-value mm-amber">{n_12nm}</span></div>
<div class="mm-stat"><span class="mm-stat-label">Total</span>
<span class="mm-stat-value">{len(dets)}</span></div>
<div class="mm-stat mm-stat-wide"><span class="mm-stat-label">Field & calibration</span>
<span class="mm-stat-meta">{field_w_um:.3f} × {field_h_um:.3f} µm · {px_per_um:.1f} px/µm · {DEVICE}</span></div>
{density_note and f'<div class="mm-stat mm-stat-wide">{density_note}</div>'}
</div>"""

    return overlay_img, heatmap_img, stats_img, csv_f.name, _df_to_preview_html(df), summary


MM_CSS = """
@import url("https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Source+Sans+3:wght@400;600;700&display=swap");
.gradio-container { max-width: 1280px !important; margin: auto !important; padding: 1rem 0.75rem 2rem !important; }
.mm-brand-bar {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 0.5rem 1rem;
  padding: 0 0 1rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}
.mm-brand-bar span {
  font-size: 0.7rem; letter-spacing: 0.06em;
  color: var(--body-text-color-subdued); font-weight: 500;
}
.mm-hero {
  padding: 1.35rem 1.5rem;
  margin-bottom: 1.25rem;
  border-radius: 16px;
  background: linear-gradient(155deg, rgba(13, 148, 136, 0.12) 0%, rgba(15, 23, 42, 0.95) 42%, rgba(30, 27, 75, 0.15) 100%);
  border: 1px solid rgba(148, 163, 184, 0.15);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.mm-hero h1 {
  font-family: "Libre Baskerville", Georgia, serif;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0 0 0.5rem 0;
  font-size: 1.75rem;
  color: #f8fafc;
}
.mm-hero .mm-sub {
  margin: 0 0 1rem 0;
  color: #cbd5e1;
  font-size: 0.95rem;
  line-height: 1.6;
  max-width: 62ch;
}
.mm-badge-row { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.mm-badge {
  font-size: 0.62rem; letter-spacing: 0.05em; font-weight: 600;
  padding: 0.28rem 0.55rem; border-radius: 999px;
  background: rgba(45, 212, 191, 0.12); color: #5eead4;
  border: 1px solid rgba(45, 212, 191, 0.25);
}
.mm-layout { display: flex; gap: 1.5rem; align-items: flex-start; flex-wrap: wrap; }
.mm-sidebar {
  flex: 1 1 300px; max-width: 360px;
  padding: 1.25rem 1.35rem; border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background: var(--block-background-fill);
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.12);
}
.mm-main {
  flex: 1 1 480px; min-width: 0;
  padding: 0.25rem 0.15rem;
  border-radius: 16px;
}
.mm-panel-title {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--body-text-color-subdued); font-weight: 600; margin: 0 0 0.75rem 0;
}
.mm-loupe-help {
  font-size: 0.82rem; line-height: 1.45; color: var(--body-text-color-subdued);
  margin: 0 0 0.75rem 0; padding: 0.65rem 0.85rem;
  border-radius: 10px; background: rgba(30, 41, 59, 0.45);
  border: 1px solid rgba(148, 163, 184, 0.12);
}
.tabs > .tab-nav button { font-weight: 500 !important; letter-spacing: 0.01em; }
.mm-callout {
  margin: 0; padding: 0.75rem 0.9rem; border-radius: 8px;
  background: #1e293b66; border: 1px solid var(--border-color-primary);
  font-size: 0.88rem; line-height: 1.45; color: var(--body-text-color);
}
.mm-callout-warn { border-color: #f59e0b55; background: #78350f22; }
.mm-science {
  margin-top: 1rem; font-size: 0.82rem; line-height: 1.5;
  color: var(--body-text-color-subdued);
}
.mm-science h4 { margin: 0.5rem 0 0.35rem; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; }
.mm-science ul { margin: 0.25rem 0 0 1rem; padding: 0; }
.mm-summary { display: flex; flex-wrap: wrap; gap: 0.65rem; margin: 0 0 1rem 0; }
.mm-stat {
  flex: 1 1 118px; padding: 0.75rem 0.95rem; border-radius: 8px;
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
}
.mm-stat-wide { flex: 1 1 100%; }
.mm-stat-label {
  display: block; font-size: 0.68rem; text-transform: uppercase;
  letter-spacing: 0.06em; opacity: 0.72; margin-bottom: 0.2rem;
}
.mm-stat-value { font-size: 1.4rem; font-weight: 700; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.mm-stat-value.mm-teal { color: #2dd4bf; }
.mm-stat-value.mm-amber { color: #fbbf24; }
.mm-stat-meta { font-size: 0.84rem; opacity: 0.92; line-height: 1.35; }
.mm-density { font-size: 0.84rem; opacity: 0.9; }
table.mm-table {
  width: 100%; border-collapse: collapse; font-size: 0.82rem;
  margin: 0.25rem 0 0.75rem 0;
}
table.mm-table th {
  text-align: left; padding: 0.45rem 0.5rem;
  border-bottom: 1px solid var(--border-color-primary);
  color: var(--body-text-color-subdued); font-weight: 600;
}
table.mm-table td { padding: 0.35rem 0.5rem; border-bottom: 1px solid #33415544; }
.mm-table-empty { margin: 0.5rem 0; opacity: 0.75; font-size: 0.9rem; }
.mm-foot {
  margin-top: 2rem; padding-top: 1rem;
  border-top: 1px solid var(--border-color-primary);
  font-size: 0.78rem; line-height: 1.45;
  color: var(--body-text-color-subdued);
}
.mm-foot code { font-size: 0.76rem; }
"""


def build_app():
    # Use named hues only (no custom Color dicts): avoids Gradio/Jinja template bugs on some stacks (e.g. HF Spaces + Py3.13).
    theme = gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=("Source Sans 3", "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=("IBM Plex Mono", "ui-monospace", "monospace"),
    ).set(
        body_background_fill_dark="*neutral_950",
        block_background_fill_dark="*neutral_900",
        border_color_primary="*neutral_700",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        block_label_text_size="*text_sm",
    )

    with gr.Blocks(
        title="MidasMap — Immunogold analysis",
        theme=theme,
        css=MM_CSS,
    ) as app:
        gr.HTML(
            """
            <div class="mm-brand-bar">
              <span>MidasMap · immunogold on TEM synapses</span>
              <span>For research — verify important counts by eye</span>
            </div>
            <div class="mm-hero">
              <h1>MidasMap</h1>
              <p class="mm-sub">
                Find <strong>6 nm</strong> (AMPA) and <strong>12 nm</strong> (NR1) gold particles in
                <strong>FFRIL</strong> micrographs. Set <strong>calibration</strong> so exports are in µm.
                Use the <strong>magnifying glass</strong> below to inspect beads and heatmaps up close.
              </p>
              <div class="mm-badge-row">
                <span class="mm-badge">FFRIL</span>
                <span class="mm-badge">CenterNet</span>
                <span class="mm-badge">CEM500K</span>
                <span class="mm-badge">F1 ≈ 0.94 LOOCV</span>
              </div>
            </div>
            """
        )

        viz_state = gr.State({"overlay": None, "heatmap": None, "stats": None})

        with gr.Row(elem_classes=["mm-layout"]):
            with gr.Column(elem_classes=["mm-sidebar"]):
                gr.HTML('<p class="mm-panel-title">1 · Upload & settings</p>')
                image_input = gr.File(
                    label="Micrograph",
                    file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                )
                px_per_um_in = gr.Number(
                    value=DEFAULT_PX_PER_UM,
                    label="Pixels per µm",
                    info=f"Default {DEFAULT_PX_PER_UM:.0f} matches the training corpus. Change if your scale differs.",
                    minimum=1,
                    maximum=1e6,
                )
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    label="Confidence",
                    info="Higher = stricter (fewer hits). Lower = more sensitive.",
                )
                with gr.Accordion("Advanced · peak spacing (NMS)", open=False):
                    nms_6nm = gr.Slider(
                        minimum=1,
                        maximum=9,
                        value=3,
                        step=2,
                        label="Spacing · 6 nm channel",
                        info="Minimum gap between AMPA peaks on the model grid.",
                    )
                    nms_12nm = gr.Slider(
                        minimum=1,
                        maximum=9,
                        value=5,
                        step=2,
                        label="Spacing · 12 nm channel",
                    )
                detect_btn = gr.Button("Run detection", variant="primary", size="lg")

                with gr.Accordion("Magnifying glass", open=True):
                    gr.HTML(
                        """<p class="mm-loupe-help" style="margin-top:0">
                        After you run detection, pick which result to inspect and adjust the sliders.
                        <strong>Magnification</strong> zooms in (smaller crop, upscaled). Use the fullscreen icon on any image for a larger view.
                        </p>"""
                    )
                    mag_view = gr.Radio(
                        choices=["Overlay", "Heatmaps", "Summary"],
                        value="Overlay",
                        label="Source image",
                    )
                    mag_cx = gr.Slider(
                        0, 100, value=50, step=0.5,
                        label="Pan left ↔ right (%)",
                    )
                    mag_cy = gr.Slider(
                        0, 100, value=50, step=0.5,
                        label="Pan up ↔ down (%)",
                    )
                    mag_zoom = gr.Slider(
                        1, 10, value=2.5, step=0.25,
                        label="Magnification",
                        info="Higher = stronger zoom (smaller region).",
                    )
                    mag_out = gr.Slider(
                        256, 768, value=480, step=64,
                        label="Loupe window (px)",
                    )
                    mag_out_img = gr.Image(
                        label="Loupe preview",
                        type="numpy",
                        height=380,
                        show_fullscreen_button=True,
                    )

                with gr.Accordion("Notes for scientists", open=False):
                    gr.Markdown(
                        """
#### What the model outputs
- **Circles** mark predicted gold centers; **scores** are CNN confidences, not p-values.
- **AMPA** = 6 nm class; **NR1** = 12 nm class (NMDA receptor subunit). Verify ambiguous sites on the raw image.

#### When to trust it
- Trained on **10 FFRIL synapse images** (453 hand-placed particles). Expect best performance on **similar prep, contrast, and magnification**.
- **Always spot-check** counts used for publication, especially near membranes and dense clusters.

#### Coordinates & CSV
- **x, y** follow image pixel order (origin top-left). **µm** columns use your calibration above.
- CSV includes **receptor**, **gold diameter**, and **calibration** used for provenance.

#### Citation
Sahai, A. (2026). *MidasMap* (software). https://github.com/AnikS22/MidasMap
                        """
                    )

            with gr.Column(elem_classes=["mm-main"]):
                gr.HTML('<p class="mm-panel-title">2 · Results</p>')
                summary_md = gr.HTML(
                    value="<p class='mm-callout'>Upload a micrograph and tap <strong>Run detection</strong>. Set pixels/µm before exporting if your scale differs.</p>"
                )
                with gr.Tabs():
                    with gr.Tab("Overlay"):
                        overlay_output = gr.Image(
                            label="Detections + scale bar",
                            type="numpy",
                            height=540,
                            show_fullscreen_button=True,
                        )
                    with gr.Tab("Heatmaps"):
                        heatmap_output = gr.Image(
                            label="Class-specific maps",
                            type="numpy",
                            height=540,
                            show_fullscreen_button=True,
                        )
                    with gr.Tab("Summary"):
                        stats_output = gr.Image(
                            label="Counts & distributions",
                            type="numpy",
                            height=440,
                            show_fullscreen_button=True,
                        )
                    with gr.Tab("Table & export"):
                        table_output = gr.HTML(
                            label="Detections (preview)",
                            value="<p class='mm-table-empty'><em>Results appear here after detection.</em></p>",
                        )
                        csv_output = gr.File(label="Download CSV")

        gr.HTML(
            f"""
            <div class="mm-foot">
              <strong>Training context:</strong> LOOCV mean F1 ≈ 0.94 on eight well-annotated folds;
              raw grayscale input (avoid heavy filtering). Not a clinical device.
              Model weights: <code>checkpoints/final/final_model.pth</code> or
              <a href="https://huggingface.co/AnikS22/MidasMap" target="_blank" rel="noopener">Hugging Face</a>.
            </div>
            """
        )

        mag_inputs = [viz_state, mag_view, mag_cx, mag_cy, mag_zoom, mag_out]

        detect_btn.click(
            fn=run_detection,
            inputs=[image_input, conf_slider, nms_6nm, nms_12nm, px_per_um_in],
            outputs=[
                overlay_output,
                heatmap_output,
                stats_output,
                csv_output,
                table_output,
                summary_md,
                viz_state,
            ],
        ).then(magnifier_zoom, mag_inputs, mag_out_img)

        for _ctrl in (mag_view, mag_cx, mag_cy, mag_zoom, mag_out):
            _ctrl.change(magnifier_zoom, mag_inputs, mag_out_img)

    return app


def _running_on_hf_space() -> bool:
    """Hugging Face Spaces injects these env vars; Gradio must bind 0.0.0.0 and never use share=True."""
    # Gradio's own detector uses SYSTEM=spaces + SPACE_ID; keep checks in sync so launch() binds correctly.
    if os.environ.get("SYSTEM") == "spaces":
        return True
    return bool(
        os.environ.get("SPACE_REPO_NAME")
        or os.environ.get("SPACE_AUTHOR_NAME")
        or os.environ.get("SPACE_ID")
    )


def _resolve_checkpoint(ckpt: Path) -> Path:
    """Use local .pth if present; on HF Space fetch from the Hub model repo if missing (smaller Space uploads)."""
    if ckpt.is_file():
        return ckpt
    if _running_on_hf_space():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise SystemExit(
                "huggingface_hub is required on the Space to download weights. "
                "Add it to requirements.txt or bundle checkpoints/final/final_model.pth in the Space."
            ) from e
        repo_id = os.environ.get("MIDASMAP_HF_WEIGHTS_REPO", "AnikS22/MidasMap").strip()
        filename = os.environ.get(
            "MIDASMAP_HF_WEIGHTS_FILE", "checkpoints/final/final_model.pth"
        ).strip()
        print(f"Checkpoint not found at {ckpt}; downloading {filename} from model repo {repo_id} ...")
        cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        return Path(cached)
    raise SystemExit(
        f"Checkpoint not found: {ckpt}\n"
        "Train with train_final.py or download from Hugging Face:\n"
        "  huggingface-cli download AnikS22/MidasMap checkpoints/final/final_model.pth "
        "--local-dir . --repo-type model"
    )


def main():
    parser = argparse.ArgumentParser(description="MidasMap web dashboard")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final/final_model.pth",
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument("--share", action="store_true", help="Gradio public share link (use if localhost is blocked)")
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        metavar="HOST",
        help='Bind address, e.g. 0.0.0.0 for LAN (default: 127.0.0.1)',
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"):
        args.share = True

    if _running_on_hf_space():
        args.share = False
        if not args.server_name:
            args.server_name = "0.0.0.0"

    ckpt = _resolve_checkpoint(Path(args.checkpoint))

    load_model(str(ckpt))
    demo = build_app()
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", str(args.port))))
    launch_kw = dict(
        share=args.share,
        server_port=port,
        server_name=args.server_name,
        show_api=False,
        inbrowser=False,
    )
    demo.launch(**launch_kw)


if __name__ == "__main__":
    main()
