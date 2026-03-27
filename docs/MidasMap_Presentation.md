# MidasMap: Automated Immunogold Particle Detection for TEM Synapse Images

---

## The Problem

Neuroscientists use **immunogold labeling** to visualize receptor proteins at synapses in transmission electron microscopy (TEM) images.

- **6nm gold beads** label AMPA receptors (panAMPA)
- **12nm gold beads** label NR1 (NMDA) receptors
- **18nm gold beads** label vGlut2 (vesicular glutamate transporter)

**Manual counting is slow and subjective.** Each image takes 30-60 minutes to annotate. With hundreds of synapses per experiment, this becomes a bottleneck.

### The Challenge
- Particles are **tiny** (4-10 pixels radius) on 2048x2115 images
- Contrast delta is only **11-39 intensity units** on a 0-255 scale
- Large dark vesicles look similar to gold particles to naive detectors
- Only **453 labeled particles** across 10 training images

---

## Previous Approaches (GoldDigger et al.)

| Approach | Result |
|----------|--------|
| CenterNet (initial attempt) | "Detection quality remained poor" |
| U-Net heatmap | Macro F1 = 0.005-0.017 |
| GoldDigger/cGAN | "No durable breakthrough" |
| Aggressive filtering | "FP dropped but TP dropped harder" |

**Core issue:** Previous systems failed due to:
1. Incorrect coordinate conversion (microns treated as normalized values)
2. Broken loss function (heatmap peaks not exactly 1.0)
3. Overfitting to fixed training patches

---

## MidasMap Architecture

```
Input: Raw TEM Image (any size)
         |
    [Sliding Window → 512x512 patches]
         |
    ResNet-50 Encoder (pretrained on CEM500K: 500K EM images)
         |
    BiFPN Neck (bidirectional feature pyramid, 2 rounds, 128ch)
         |
    Transposed Conv Decoder → stride-2 output
         |
    +------------------+-------------------+
    |                  |                   |
    Heatmap Head       Offset Head
    (2ch sigmoid)      (2ch regression)
    6nm channel        sub-pixel x,y
    12nm channel       correction
    |                  |
    +------------------+-------------------+
         |
    Peak Extraction (max-pool NMS)
         |
    Cross-class NMS + Mask Filter
         |
    Output: [(x, y, class, confidence), ...]
```

### Key Design Decisions

**CEM500K Backbone:** ResNet-50 pretrained on 500,000 electron microscopy images via self-supervised learning. The backbone already understands EM structures (membranes, vesicles, organelles) before seeing any gold particles. This is why the model reaches F1=0.93 in just 5 epochs.

**Stride-2 Output:** Standard CenterNet uses stride 4. At stride 4, a 6nm bead (4-6px radius) collapses to 1 pixel — too small to detect reliably. At stride 2, the same bead occupies 2-3 pixels, enough for Gaussian peak detection.

**CornerNet Focal Loss:** With positive:negative pixel ratio of 1:23,000, standard BCE would learn to predict all zeros. The focal loss uses `(1-p)^alpha` weighting to focus on hard examples and `(1-gt)^beta` penalty reduction near peaks.

**Raw Image Input:** No preprocessing. The CEM500K backbone was trained on raw EM images. Any heavy preprocessing (top-hat, CLAHE) creates a domain gap and hurts performance. The model learns to distinguish particles from vesicles through training data, not handcrafted filters.

---

## Training Strategy

### 3-Phase Training with Discriminative Learning Rates

| Phase | Epochs | What's Trainable | Learning Rate |
|-------|--------|-------------------|---------------|
| **1. Warm-up** | 40 | BiFPN + heads only | 1e-3 |
| **2. Deep unfreeze** | 40 | + layer3 + layer4 | 1e-5 to 5e-4 |
| **3. Full fine-tune** | 60 | All layers | 1e-6 to 2e-4 |

```
Loss Curve (final model):

Phase 1          Phase 2          Phase 3
|                |                |
1.4 |\           |                |
    | \          |                |
1.0 |  \         |                |
    |   ----     |                |
0.8 |       \    |                |
    |        \   |                |
0.6 |         \--+---             |
    |            |   \            |
0.4 |            |    \---        |
    |            |        \-------+---
0.2 |            |                |
    +---+---+----+---+---+----+---+---+--> Epoch
    0   10  20   40  50  60   80  100 140
```

### Data Augmentation
- Random 90-degree rotations (EM is rotation-invariant)
- Horizontal/vertical flips
- Conservative brightness/contrast (+-8% — preserves the subtle particle signal)
- Gaussian noise (simulates shot noise)
- **Copy-paste augmentation**: real bead crops blended onto training patches
- **70% hard mining**: patches centered on particles, 30% random

### Overfitting Prevention
- **Unique patches every epoch**: RNG reseeded per sample so the model never sees the same patch twice
- **Early stopping**: patience=20 epochs, monitoring validation F1
- **Weight decay**: 1e-4 on all parameters

---

## Critical Bugs Found and Fixed

### Bug 1: Coordinate Conversion
**Problem:** CSV files labeled "XY in microns" were assumed to be normalized [0,1] coordinates. They were actual micron values.

**Effect:** All particle annotations were offset by 50-80 pixels from the real locations. The model was learning to detect particles where none existed.

**Fix:** Multiply by 1790 px/micron (verified against researcher's color overlay TIFs across 7 synapses).

### Bug 2: Heatmap Peak Values
**Problem:** Gaussian peaks were centered at float coordinates, producing peak values of 0.78-0.93 instead of exactly 1.0.

**Effect:** The CornerNet focal loss uses `pos_mask = (gt == 1.0)` to identify positive pixels. With no pixels at exactly 1.0, the model had **zero positive training signal**. It literally could not learn.

**Fix:** Center Gaussians at the integer grid point (always produces 1.0). Sub-pixel precision is handled by the offset regression head.

### Bug 3: Overfitting on Fixed Patches
**Problem:** The dataset generated 200 random patches once at initialization. Every epoch replayed the same patches.

**Effect:** On fast CUDA GPUs, the model memorized all patches in ~17 epochs (loss crashed from 1.6 to 0.002). Validation F1 peaked at 0.66 and degraded.

**Fix:** Reseed RNG per `__getitem__` call so every patch is unique.

---

## Results

### Leave-One-Image-Out Cross-Validation (10 folds, 5 seeds each)

| Fold | Avg F1 | Best F1 | Notes |
|------|--------|---------|-------|
| S27 | **0.990** | 0.994 | |
| S8 | **0.981** | 0.988 | |
| S25 | **0.972** | 0.977 | |
| S29 | **0.956** | 0.966 | |
| S1 | **0.930** | 0.940 | |
| S4 | **0.919** | 0.972 | |
| S22 | **0.907** | 0.938 | |
| S13 | **0.890** | 0.912 | |
| S7 | 0.799 | 1.000 | Only 3 particles (noisy metric) |
| S15 | 0.633 | 0.667 | Only 1 particle (noisy metric) |

**Mean F1 = 0.943** (8 folds with sufficient annotations)

### Per-class Performance (S1 fold, best threshold)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 6nm (AMPA) | 0.895 | **1.000** | **0.944** |
| 12nm (NR1) | 0.833 | **1.000** | **0.909** |

**100% recall** on both classes — every particle is found. Only errors are a few false positives.

### Generalization to Unseen Images

Tested on 15 completely unseen images from a different imaging session. Detections land correctly on particles with no manual tuning. The model successfully detects both 6nm and 12nm particles on:
- Wild-type (Wt2) samples
- Heterozygous (Het1) samples
- Different synapse regions (D1, E3, S1, S10, S12, S18)

---

## System Components

```
MidasMap/
  config/config.yaml        # All hyperparameters
  src/
    preprocessing.py        # Data loading (10 synapses, 453 particles)
    model.py                # CenterNet: ResNet-50 + BiFPN + heads (24.4M params)
    loss.py                 # CornerNet focal loss + offset regression
    heatmap.py              # GT generation + peak extraction + NMS
    dataset.py              # Patch sampling, augmentation, copy-paste
    postprocess.py          # Mask filter, cross-class NMS
    ensemble.py             # D4 TTA + sliding window inference
    evaluate.py             # Hungarian matching, F1/precision/recall
    visualize.py            # Overlay visualizations
  train.py                  # LOOCV training (--fold, --seed)
  train_final.py            # Final deployable model (all data)
  predict.py                # Inference on new images
  evaluate_loocv.py         # Full evaluation runner
  app.py                    # Gradio web dashboard
  slurm/                    # HPC job scripts
  tests/                    # 36 unit tests
```

---

## Dashboard

MidasMap includes a web-based dashboard (Gradio) for interactive use:

1. **Upload** any TEM image (.tif)
2. **Adjust** confidence threshold and NMS parameters
3. **View** detections overlaid on the image
4. **Inspect** per-class heatmaps
5. **Analyze** confidence distributions and spatial patterns
6. **Export** results as CSV (particle_id, x_px, y_px, x_um, y_um, class, confidence)

```
python app.py --checkpoint checkpoints/final/final_model.pth
# Opens at http://localhost:7860
```

---

## Future Directions

1. **Spatial analytics**: distance to synaptic cleft, nearest-neighbor analysis, Ripley's K-function
2. **Size regression head**: predict actual bead diameter instead of binary classification
3. **18nm detection**: extend to vGlut2 particles (3-class model)
4. **Active learning**: flag low-confidence detections for human review
5. **Cross-protocol generalization**: fine-tune on cryo-EM or different staining protocols

---

## Technical Summary

- **Model**: CenterNet with CEM500K-pretrained ResNet-50, BiFPN neck, stride-2 output
- **Training**: 3-phase with discriminative LRs, 140 epochs, 453 particles / 10 images
- **Evaluation**: Leave-one-image-out CV, Hungarian matching, F1 = 0.943
- **Inference**: Sliding window (512x512, 128px overlap), ~10s per image on GPU
- **Output**: Per-particle (x, y, class, confidence) with optional heatmap visualization
