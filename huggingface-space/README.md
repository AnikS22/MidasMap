---
title: MidasMap
emoji: 🔬
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# MidasMap: Precision Immunogold Detection

> **Detect receptor-specific nanosized particles (6 nm AMPA, 12 nm NMDA) in electron microscopy synapses at 94.3% accuracy.** Upload your TEM synapse image and get particle locations in seconds—no manual annotation required.

**Key Results**: F1=0.943 LOOCV | 6nm: 0.944 | 12nm: 0.909 | Sub-pixel accuracy (±0.5 px)

This is a live Gradio demo for **[MidasMap](https://github.com/AnikS22/MidasMap)** — an automated detector for immunogold particles in TEM synapse images.

## 🎯 How to Use

1. **Upload** your 2048×2048 px TEM image (TIFF or PNG)
2. **Adjust** confidence threshold to filter detections
3. **View** particles overlaid on the image (6nm in one color, 12nm in another)
4. **Download** detections as CSV (x, y, class, confidence)

## 🚀 Deploy from Your Laptop

From the **MidasMap** repo root (recommended: skip checkpoint upload to avoid LFS issues):

```bash
export HF_TOKEN=hf_...  # Your Hugging Face write token
export HF_SPACE_SKIP_CHECKPOINT=1
./scripts/upload_hf_space.sh
```

**Alternative** (git + LFS, often more reliable):
```bash
brew install git-lfs && git lfs install  # once per machine
export HF_TOKEN=hf_...
./scripts/push_hf_space_git.sh
```

**First time?** Create the Space:
```bash
huggingface-cli repo create MidasMap --type space --space_sdk gradio -y
```

**Model weights**: Automatically downloaded from the Hub (`AnikS22/MidasMap` repo) on first boot. Override with env vars: `MIDASMAP_HF_WEIGHTS_REPO`, `MIDASMAP_HF_WEIGHTS_FILE`.

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **401 / Not authenticated** | Set `HF_TOKEN=hf_...` with write access, or run `huggingface-cli login` |
| **LFS / upload stuck** | Use `HF_SPACE_SKIP_CHECKPOINT=1` to skip the checkpoint; model downloads from Hub on boot |
| **Space doesn't exist** | Create via HF web UI or run `huggingface-cli repo create MidasMap --type space --space_sdk gradio` |
| **Model not found** | Add `HF_TOKEN` as a Space **secret** if model repo is private |
| **Still failing** | Try `export HF_HUB_ENABLE_HF_TRANSFER=1` and install `pip install hf_transfer` for faster uploads |

## 📚 Learn More

- **Full documentation**: [github.com/AnikS22/MidasMap](https://github.com/AnikS22/MidasMap)
- **Architecture details**: Heatmap + offset heads, BiFPN feature fusion, sub-pixel accuracy
- **Training strategy**: 3-phase progressive unfreezing, hard mining, focal loss for extreme imbalance

## 🔗 Embed This Space

Embed in your website:  
`https://yoursite.vercel.app/?embed=https://huggingface.co/spaces/YOUR_USER/midasmap`
