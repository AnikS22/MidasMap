# Pipeline Visualization Guide

## Overview

This guide explains how to generate and view visualizations of the **MidasMap pipeline**—showing exactly what the neural network "sees" at each step of processing.

## Quick Start

### 1. Generate Synthetic Demo Data

First, create synthetic TEM-like images with simulated gold particles:

```bash
python scripts/generate_demo_data.py --output demo_images/ --n-images 3
```

This creates:
- `synapse_00_original.png` — Synthetic TEM image
- `synapse_00_heatmap_6nm.png` — Training target for 6nm particles
- `synapse_00_heatmap_12nm.png` — Training target for 12nm particles
- `synapse_00_aug_*.png` — Augmented versions
- `synapse_00_annotations.txt` — Ground truth particle coordinates

### 2. Visualize the Pipeline

See what the model "sees" at each stage:

```bash
python scripts/visualize_pipeline.py --image demo_images/synapse_00_original.png
```

This generates `synapse_00_pipeline_viz.png` showing:

#### Stage 1: Input
- **Original Image** — Grayscale TEM capture (0-255)
- **Preprocessed** — Normalized to [0, 1]

#### Stage 2: Feature Extraction (ResNet-50 Encoder)
- **Shallow Features** — Edges, corners (Layer 1)
- **Middle Features** — Blobs, textures (Layer 2)
- **Deep Features** — Particle likelihood (Layer 3-4)

#### Stage 3: Model Outputs (Detection Heads)
- **Heatmap Output** — "Where are particles?" (2 channels: 6nm & 12nm)
- **Offset Output** — "Exact sub-pixel location?" (2 channels: dx, dy)
- **Final Detections** — Combined prediction + NMS

### 3. View Augmentations

See how D4 TTA (Test-Time Augmentation) works:

```bash
python scripts/visualize_pipeline.py --image demo_images/synapse_00_original.png --augmentation
```

This creates `synapse_00_augmentation_viz.png` showing:
- 4 rotations (0°, 90°, 180°, 270°)
- 2 flips (horizontal, vertical)
- 3 brightness variations (×0.9, ×1.0, ×1.1)
- Why averaging 10 views improves predictions

---

## What Each Stage Means

### Stage 1: Preprocessing
```
Original:    grayscale image (0-255 uint8)
             └─> Normalize to [0, 1]
             └─> Pad to multiple of 32
             └─> Ready for neural network
```

**Why?** Neural networks expect normalized inputs. Padding helps with stride calculations.

---

### Stage 2: Feature Extraction (Encoder)

The ResNet-50 encoder learns to recognize patterns at different levels:

#### Shallow (Early Layers)
- Detects: Edges, corners, simple shapes
- Receptive field: ~11×11 pixels
- Example: "Is this pixel near a boundary?"

#### Middle (Intermediate Layers)
- Detects: Textures, blobs, local structures
- Receptive field: ~51×51 pixels
- Example: "Is this a bright circular region?"

#### Deep (Late Layers)
- Detects: High-level objects (particles vs background)
- Receptive field: ~195×195 pixels
- Example: "Is this likely a gold particle?"

**Why this hierarchy?** Small details matter for localization, but large context matters for classification.

---

### Stage 3: BiFPN Neck

The BiFPN (Bidirectional Feature Pyramid Network) combines information from all levels:

```
Deep features (coarse, semantic)
    ↓
    [Bidirectional fusion]
    ↑
Shallow features (fine, spatial detail)
```

**Why?** A particle is both:
- A semantic object (gold, not background)
- A fine spatial detail (where exactly is it?)

BiFPN uses both simultaneously.

---

### Stage 4: Detection Heads

Two parallel heads process the fused features:

#### Heatmap Head
- **Input:** Fused features
- **Output:** 2 channels (6nm probability, 12nm probability)
- **Resolution:** Stride-2 (H/2 × W/2)
- **Activation:** Sigmoid (0-1 confidence)

```
Feature map (256×256 for 512×512 input)
    ↓
Conv → ReLU → Conv → Sigmoid
    ↓
Heatmap (256×256, 2 channels)
    ↓
"Pixel (100, 100) is 95% likely a 6nm particle"
```

#### Offset Head
- **Input:** Same fused features
- **Output:** 2 channels (Δx, Δy)
- **Range:** (-0.5, 0.5) pixels
- **Purpose:** Sub-pixel refinement

```
Feature map (256×256)
    ↓
Conv → ReLU → Conv → Linear
    ↓
Offset (256×256, 2 channels)
    ↓
"Refine center by (+0.3, -0.2) pixels"
```

---

### Stage 5: Post-Processing

The outputs are converted to final detections:

```
Heatmap (H/2, W/2)
    ↓
Find local maxima (peaks)
    ↓
Threshold by confidence (e.g., > 0.3)
    ↓
Apply offsets for sub-pixel refinement
    ↓
Cross-class NMS (remove overlapping 6nm/12nm)
    ↓
Final detections: [(x, y, class, confidence), ...]
```

---

## Understanding the Visualizations

### Pipeline Figure (8 panels)

```
┌─────────────┬──────────────┬──────────┐
│ Original    │ Preprocessed │ Info     │
├─────────────┼──────────────┼──────────┤
│ Shallow     │ Middle       │ Deep     │  ← Feature levels
├─────────────┼──────────────┼──────────┤
│ Heatmap     │ Offset       │ Final    │  ← Model outputs
└─────────────┴──────────────┴──────────┘
```

**Color interpretation:**

- **Grayscale panels:** Raw image data (bright = high intensity)
- **Hot colormap panels:** Detected features (red = strong feature, blue = weak)
- **Viridis colormap panels:** Confidence scores (yellow = high confidence, purple = low)

**Detection visualization:**
- White circle = detected particle
- Color = confidence (green = high, red = low)
- Size = fixed for visibility

---

### Augmentation Figure (9 panels)

```
Row 1: 0°, 90°, 180° rotations
Row 2: Horizontal flip, Vertical flip, Original (reference)
Row 3: Brightness ×0.9, ×1.0, ×1.1
```

**Why it matters:**
- Gold particles are rotationally symmetric (look same at any angle)
- Averaging predictions from all 9 views reduces noise
- Typical F1 gain: +1-3%

---

## Interpreting Results

### Perfect Detection
```
Original image
    └─> Heatmap shows strong peak at particle center
    └─> Offset points inward (refinement)
    └─> Final detection: Correct location, high confidence
```

### False Positive (Fake Alarm)
```
Empty region
    └─> Heatmap shows spurious peak (noise)
    └─> Final detection: Wrong location, medium confidence
    └─> Solution: User can adjust confidence threshold
```

### False Negative (Missed)
```
Actual particle
    └─> Heatmap is weak or multimodal (nearby texture)
    └─> Feature extraction didn't recognize it
    └─> Solution: Requires model retraining on more data
```

---

## Generating Custom Visualizations

### With Your Own Data

If you have real TEM images:

```bash
python scripts/visualize_pipeline.py --image path/to/your/synapse.tif \
                                      --output custom_pipeline.png \
                                      --augmentation
```

### Batch Processing

```bash
for image in demo_images/*_original.png; do
    echo "Processing $image..."
    python scripts/visualize_pipeline.py --image "$image"
done
```

---

## Technical Details for AI Researchers

### Why Stride-2?

Standard CNNs use stride-4 or stride-8 (faster, simpler).

For gold particles (4-6 pixels):
- **Stride-4:** Particle → 1-2 pixels at feature map → hard to localize
- **Stride-2:** Particle → 2-3 pixels at feature map → sufficient signal

At stride-2, a 6nm particle spans 2-3 feature map pixels, allowing:
- Peak detection (local maximum)
- Sub-pixel refinement (offset regression)
- Proper gradient flow for training

### ResNet-50 Pretrained Weights

The visualized "feature extraction" is approximated. Real extraction:

```python
# Real code (src/model.py)
x0 = self.stem(x)        # stride 4, 64 channels
p2 = self.layer1(x0)     # stride 4, 256 channels
p3 = self.layer2(p2)     # stride 8, 512 channels
p4 = self.layer3(p3)     # stride 16, 1024 channels
p5 = self.layer4(p4)     # stride 32, 2048 channels

# BiFPN fuses all levels
features = self.bifpn([p2, p3, p4, p5])

# Upsample P2 to stride 2
x_up = self.upsample(features[0])  # stride 2, 64 channels

# Dual heads
heatmap = self.heatmap_head(x_up)
offsets = self.offset_head(x_up)
```

### Heatmap Head Training

During training, we generate Gaussian targets:

```python
# Ground truth heatmap (from annotation at (x, y))
yy, xx = np.mgrid[0:H, 0:W]
gt_heatmap = np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * sigma**2))
# sigma = 1.5 pixels (slightly larger than particle)

# Loss: CornerNet focal loss
loss = cornernet_focal_loss(pred_heatmap, gt_heatmap)
```

The model learns to output high confidence near particle centers.

---

## Common Issues & Solutions

### "Pipeline visualization shows peaks everywhere"
**Cause:** Image is too noisy or has many texture features
**Solution:** 
- Increase confidence threshold (only keep detections > 0.5)
- Check if microscope settings need adjustment
- Apply denoising to image preprocessing

### "Offset head seems random"
**Cause:** May be (it's hard to visualize offset vectors)
**Solution:**
- Check final detections (offset impact on accuracy)
- Offset is only used when confidence > 0.3
- Magnitude should be small (±0.5 px)

### "Augmentation images look identical"
**Cause:** Rotations at 90° intervals are symmetric for this synthetic image
**Solution:**
- Try different demo images
- Check with real TEM images (have more structure)
- Augmentations are subtle but improve 1-3% F1

---

## Questions to Ask While Looking at Visualizations

1. **Preprocessing:** Is the normalized image still high-contrast enough?
2. **Features:** Do shallow features match image boundaries? Do deep features highlight particles?
3. **Heatmap:** Are peaks centered on particles? Any spurious peaks?
4. **Offset:** Do vectors point toward particle centers? Magnitudes < 0.5 px?
5. **Final:** Are detections reasonable? What would you adjust?

---

## Next Steps

- Try with real TEM images (place in a directory, run batch)
- Adjust confidence thresholds in `app.py` interactively
- Train the full model on your own data
- Contribute improvements to the visualization

---

**For questions:** See the main README or GitHub issues.
