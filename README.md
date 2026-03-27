---
license: mit
tags:
  - immunogold
  - particle-detection
  - electron-microscopy
  - TEM
  - neuroscience
  - CenterNet
  - CEM500K
  - synapse
datasets:
  - custom
metrics:
  - f1
model-index:
  - name: MidasMap
    results:
      - task:
          type: object-detection
          name: Immunogold Particle Detection
        metrics:
          - type: f1
            value: 0.943
            name: LOOCV Mean F1 (8 annotated folds)
---

# MidasMap: Immunogold Particle Detection for TEM Synapse Images

MidasMap automatically detects **6nm** (AMPA receptor) and **12nm** (NR1/NMDA receptor) immunogold particles in freeze-fracture replica immunolabeling (FFRIL) transmission electron microscopy images.

## Performance

| Metric | Value |
|--------|-------|
| **LOOCV Mean F1** | **0.943** (8 folds with sufficient annotations) |
| 6nm (AMPA) F1 | 0.944 (100% recall) |
| 12nm (NR1) F1 | 0.909 (100% recall) |
| Parameters | 24.4M |
| Inference | ~10s per image (GPU) |

Validated on 453 labeled particles across 10 synapse images via leave-one-image-out cross-validation with 5 random seeds per fold.

## Quick Start

```python
import torch
from src.model import ImmunogoldCenterNet
from src.ensemble import sliding_window_inference
from src.heatmap import extract_peaks
from src.postprocess import cross_class_nms
import tifffile

# Load model
model = ImmunogoldCenterNet(bifpn_channels=128, bifpn_rounds=2)
ckpt = torch.load("checkpoints/final/final_model.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Run on any TEM image
img = tifffile.imread("your_image.tif")
if img.ndim == 3:
    img = img[:, :, 0]

with torch.no_grad():
    hm, off = sliding_window_inference(model, img, patch_size=512, overlap=128)

dets = extract_peaks(torch.from_numpy(hm), torch.from_numpy(off),
                     stride=2, conf_threshold=0.25)
dets = cross_class_nms(dets, 8)

for d in dets:
    print(f"{d['class']} at ({d['x']:.1f}, {d['y']:.1f}) conf={d['conf']:.3f}")
```

## Web Dashboard

```bash
pip install gradio
python app.py --checkpoint checkpoints/final/final_model.pth
# Opens at http://localhost:7860
```

Upload TIF images, adjust confidence threshold, view heatmaps, and export CSV results.

## Architecture

```
Raw TEM Image (any size)
    |
[Sliding window: 512x512, 128px overlap]
    |
ResNet-50 (CEM500K pretrained on 500K EM images)
    |
BiFPN (bidirectional feature pyramid, 2 rounds, 128ch)
    |
Transposed Conv → stride-2 output (H/2 x W/2)
    |
+--Heatmap Head (2ch sigmoid: 6nm + 12nm)
+--Offset Head (2ch: sub-pixel x,y correction)
    |
Peak extraction (max-pool NMS) → detections
```

### Key Design Choices

- **CEM500K backbone**: Pretrained on 500,000 electron microscopy images. Reaches F1=0.93 in just 5 training epochs because it already understands EM structures.
- **Stride-2 output**: Standard CenterNet uses stride 4, but 6nm beads (4-6px radius) collapse to 1px at that resolution. Stride 2 preserves 2-3px per bead.
- **CornerNet focal loss**: Handles the extreme class imbalance (positive:negative pixel ratio ~1:23,000).
- **Raw image input**: No preprocessing — CEM500K was trained on raw EM, so any heavy filtering creates a domain gap.

## Training

### 3-Phase Strategy
1. **Phase 1** (40 epochs): Freeze encoder, train BiFPN + heads at lr=1e-3
2. **Phase 2** (40 epochs): Unfreeze layer3+4 at lr=1e-5 to 5e-4
3. **Phase 3** (60 epochs): Full fine-tune with discriminative LRs (1e-6 to 2e-4)

### Data Augmentation
- Random 90-degree rotations, flips
- Conservative brightness/contrast (+-8%)
- Gaussian noise, mild blur
- Copy-paste: real bead crops blended onto training patches
- 70% hard mining (patches centered on particles)

### Overfitting Prevention
- RNG reseeded per sample (unique patches every epoch)
- Early stopping (patience=20, monitoring val F1)
- Weight decay 1e-4

### Train Final Model
```bash
python train_final.py --config config/config.yaml --device cuda:0
```

### HPC (SLURM)
```bash
sbatch slurm/05_train_final.sh
```

## LOOCV Results (per fold)

| Fold | Avg F1 | Best F1 | # Particles |
|------|--------|---------|-------------|
| S27 | 0.990 | 0.994 | 45 |
| S8 | 0.981 | 0.988 | 70 |
| S25 | 0.972 | 0.977 | 41 |
| S29 | 0.956 | 0.966 | 36 |
| S1 | 0.930 | 0.940 | 22 |
| S4 | 0.919 | 0.972 | 113 |
| S22 | 0.907 | 0.938 | 102 |
| S13 | 0.890 | 0.912 | 20 |
| S7* | 0.799 | 1.000 | 3 |
| S15* | 0.633 | 0.667 | 1 |

*S7 and S15 have insufficient annotations for reliable evaluation (3 and 1 particles respectively).

## Dataset

- 10 FFRIL synapse images (2048x2115 pixels)
- 403 labeled 6nm particles (AMPA receptors)
- 50 labeled 12nm particles (NR1 receptors)
- Annotations in microns, converted at 1790 px/micron

## Critical Implementation Notes

1. **Coordinate conversion**: CSV "XY in microns" values are actual microns, not normalized coordinates. Multiply by 1790 to get pixels.
2. **Heatmap peaks**: Must be exactly 1.0 at integer grid centers. The CornerNet focal loss uses `pos_mask = (gt == 1.0)`.
3. **Patch diversity**: RNG must be reseeded per `__getitem__` call to prevent memorizing fixed patches.

## Citation

If you use MidasMap in your research, please cite:

```bibtex
@software{midasmap2026,
  title={MidasMap: Automated Immunogold Particle Detection for TEM Synapse Images},
  author={Sahai, Anik},
  year={2026},
  url={https://github.com/AnikS22/MidasMap}
}
```

## Dependencies

- PyTorch >= 2.0
- torchvision
- albumentations
- scikit-image
- tifffile
- CEM500K weights (download: `python scripts/download_cem500k.py`)
