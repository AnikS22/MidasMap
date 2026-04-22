# MidasMap

Automated detection of **6 nm (AMPA)** and **12 nm (NR1/NMDA)** immunogold particles in FFRIL TEM synapse images using a CenterNet-style deep learning detector.

## Headline Results

| Metric | Value |
|---|---|
| LOOCV F1 (leave-one-image-out, 8 usable folds) | **0.943** |
| 6 nm Particle F1 | 0.944 |
| 12 nm Particle F1 | 0.909 |
| Model Parameters | 24.4M |
| Evaluation Dataset | 10 synapse images, 453 labeled particles |

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Model Architecture](#model-architecture)
4. [Training Strategy](#training-strategy)
5. [Data Format](#data-format)
6. [Usage](#usage)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Repository Structure](#repository-structure)

---

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Interactive Demo

```bash
./scripts/run_local.sh
```

Open `http://127.0.0.1:7860` in your browser.

### Predict on New Images

```bash
python predict.py \
  --image path/to/tem_image.tif \
  --checkpoint checkpoints/final/final_model.pth \
  --output results/detections.csv
```

### Train the Model

**Production model** (trained on all 10 images):
```bash
python train_final.py --config config/config.yaml --device cuda:0
```

**LOOCV evaluation** (leave-one-image-out):
```bash
python evaluate_loocv.py --config config/config.yaml --device cuda:0
```

---

## Project Overview

MidasMap is a specialized detector for immunogold particles in electron microscopy images. Unlike general object detectors, MidasMap handles the extreme class imbalance (23,000:1 negative:positive pixel ratio) and sub-pixel precision requirements of TEM particle detection.

### Key Characteristics

- **Input**: 2048×2048 px grayscale TEM images
- **Particles**: 6 nm (AMPA receptors) and 12 nm (NMDA receptors)
- **Particle density**: ~50-60 particles per image (<0.1% of image area)
- **Output**: Particle coordinates (x, y) with sub-pixel accuracy (±0.5 px)
- **Loss**: CornerNet penalty-reduced focal loss (handles extreme imbalance)
- **Framework**: PyTorch, ResNet-50 encoder (CEM500K pretrained)

---

## Model Architecture

### High-Level Pipeline

```
Input Image (2048×2048)
    ↓
Sliding-window inference (512×512 patches, 50% overlap)
    ↓
ResNet-50 Encoder (CEM500K pretrained, frozen in Phase 1)
    ↓
BiFPN (Bidirectional Feature Pyramid Network)
    ↓
Parallel Heads:
  • Heatmap Head → (B, 2, 256, 256) — soft localization
  • Offset Head → (B, 2, 256, 256) — sub-pixel correction
    ↓
Peak Extraction + Gaussian Fitting
    ↓
Non-Maximum Suppression (NMS, threshold=0.5)
    ↓
Output: Particle coordinates + confidence scores
```

### Encoder

- **ResNet-50** initialized from **CEM500K** (contrastive learning on EM images)
- Output: Feature maps at multiple scales (stride 4, 8, 16)

### Feature Fusion

- **BiFPN** (3 rounds): Bi-directional connections between pyramid levels
- Improves multi-scale feature propagation (critical for detecting 6 nm and 12 nm particles)

### Detection Heads

**Heatmap Head** (2 channels: one per particle class)
- Outputs soft heatmap via sigmoid activation
- Targets: 2D Gaussian distributions centered at particle locations
- Loss: CornerNet penalty-reduced focal loss (see below)

**Offset Head** (2 channels: x and y sub-pixel offsets)
- Outputs continuous offsets in range [-0.5, 0.5]
- Targets: Sub-pixel correction for integer grid predictions
- Loss: Smooth L1 (robust to outliers, sensitive to small errors)

### Loss Functions

#### Heatmap Loss: CornerNet Penalty-Reduced Focal Loss

For a 23,000:1 negative:positive ratio, standard binary cross-entropy learns to predict all zeros.

**Positive loss** (at particle centers, GT=1):
```
L+ = -log(pred) × (1-pred)²
```
Rewards high confidence at true particle locations.

**Negative loss** (away from particles, GT<1):
```
L- = -log(1-pred) × pred² × (1-GT)⁴
```
- `pred²` exponent: penalizes confident wrong predictions
- `(1-GT)⁴` penalty reduction: Near particle peaks, reduces penalty and focuses on hard negatives (helps with false positives near true particles)

#### Offset Loss: Smooth L1

```
L = {
  0.5 × error² / β        if |error| < β
  |error| - 0.5 × β       otherwise
}
```
β=1.0. Quadratic for small errors (sub-pixel precision), linear for large errors (robustness to outliers).

#### Total Loss

```
Total = L_heatmap + λ × L_offset
```
λ=1.0 (equal weighting)

---

## Training Strategy

### Patch-Based Hard Mining

During training, **do NOT use sliding window**. Instead:

- **70% Hard Mining**: Randomly select a particle, extract 256×256 patch centered on it
  - Ensures every patch contains at least one particle
  - Overcomes <0.1% particle density in full images
  
- **30% Random Background**: Random patch location (often no particles)
  - Trains model to reduce false positives on empty regions

**Per epoch**: ~10× number of particles patches (e.g., 600 patches/epoch for 60 particles)

### 3-Phase Training Strategy

#### Phase 1: Frozen Encoder (40 epochs)
- Freeze entire ResNet-50 encoder
- Train only BiFPN + heads
- **Goal**: Prevent overfitting to 10 images using pre-trained features
- **Learning rate**: 1e-3

#### Phase 2: Unfreeze Deep Layers (40 epochs)
- Unfreeze ResNet layer3 + layer4 (deep blocks)
- Keep stem, layer1, layer2 frozen
- **Goal**: Fine-tune deeper encoder layers while preserving shallow features
- **Learning rates** (graduated):
  - Layer3: 1e-5
  - Layer4: 5e-5
  - Heads: 5e-4

#### Phase 3: Full Fine-tune (60 epochs)
- Unfreeze all layers
- **Goal**: Polish entire model with decreasing learning rates
- **Learning rates** (graduated by depth):
  - Stem: 1e-6 (very conservative)
  - Layer1: 5e-6
  - Layer2: 1e-5
  - Layer3: 5e-5
  - Layer4: 1e-4
  - Heads: 2e-4
- Cosine annealing schedule

**Total**: 140 epochs

### Rationale

- **Phase 1** prevents overfitting on small dataset
- **Phase 2** gradually enables encoder adaptation
- **Phase 3** fine-tunes all components while preserving pre-training

---

## Data Format

### Directory Structure

```
Max Planck Data/
└── Gold Particle Labelling/
    └── analyzed synapses/
        ├── S1/
        │   └── S1_image.tif
        │   └── S1_image.json  (annotations)
        ├── S2/
        ├── ...
        └── S10/
```

### Image Format

- **Type**: TIFF (8-bit or 16-bit grayscale)
- **Size**: 2048×2048 pixels
- **Resolution**: ~0.54 nm/pixel (FFRIL TEM)

### Annotation Format (JSON)

```json
{
  "particles": [
    {
      "x": 512.3,
      "y": 256.7,
      "class": "6nm",
      "confidence": 1.0
    },
    {
      "x": 768.1,
      "y": 384.2,
      "class": "12nm",
      "confidence": 1.0
    }
  ]
}
```

- **x, y**: Floating-point sub-pixel coordinates
- **class**: "6nm" or "12nm"
- **confidence**: 1.0 for manually labeled particles

---

## Usage

### 1. Inference on Single Image

```bash
python predict.py \
  --image path/to/synapse.tif \
  --checkpoint checkpoints/final/final_model.pth \
  --output results/detections.csv \
  --conf-threshold 0.5
```

**Output CSV**:
```
x,y,class,confidence
512.3,256.7,6nm,0.95
768.1,384.2,12nm,0.87
```

### 2. Batch Inference

```bash
python predict.py \
  --image-dir path/to/images/ \
  --checkpoint checkpoints/final/final_model.pth \
  --output-dir results/
```

### 3. Interactive Web Demo

```bash
./scripts/run_local.sh
```

Launches Gradio interface:
- Upload TEM image
- Adjust confidence threshold
- View detection results with overlaid circles
- Download detections as CSV

### 4. Custom Training on New Data

Modify `config/config.yaml` and run:

```bash
python train_final.py \
  --config config/config.yaml \
  --device cuda:0
```

Configuration options:
- `patch_size`: 256 (recommended)
- `hard_mining_fraction`: 0.7 (70% hard mining)
- `batch_size`: 16
- `learning_rate`: 1e-3 (Phase 1)
- `heatmap.sigmas`: [1.0, 1.5] (Gaussian widths for 6nm, 12nm)

---

## Evaluation

### Leave-One-Image-Out (LOOCV)

Standard k-fold CV where k=10 (one test image per fold). Most rigorous evaluation for small datasets.

```bash
python evaluate_loocv.py \
  --config config/config.yaml \
  --device cuda:0 \
  --output results/loocv_metrics.json
```

**Process**:
1. For each of 10 images:
   - Train model on remaining 9 images
   - Evaluate on held-out image
   - Report per-class F1 scores
2. Average metrics across all 10 folds

**Result**: Average F1=0.943 across all folds

### Metrics

- **F1 Score**: Primary metric (balances precision/recall)
- **Precision**: TP / (TP + FP) — avoid false positives
- **Recall**: TP / (TP + FN) — catch true particles
- **Per-class metrics**: Separate scores for 6nm and 12nm particles

### Visualization

Generated diagrams in `results/diagrams/`:

```bash
python scripts/visualize_loss_functions.py      # Loss function plots
python scripts/visualize_patch_mining.py        # Training strategy diagram
python scripts/visualize_training_phases.py     # 3-phase schedule
python scripts/generate_augmentation_examples.py # Data augmentation examples
```

---

## Deployment

### Local Docker

```bash
docker compose up --build
```

Launches Gradio app on `http://localhost:7860` with GPU support.

**docker-compose.yml**:
- Base: `nvidia/cuda:11.8-runtime-ubuntu22.04`
- Mount: Local data directory for batch inference

### Hugging Face Spaces

Directory: `huggingface-space/`

**Deployment steps**:
1. Create new Space on huggingface.co
2. Point to this repository
3. Spaces will auto-build from `app.py`
4. Live at `https://huggingface.co/spaces/username/midasmap`

**Configuration** (`huggingface-space/`):
- `app.py`: Gradio interface (Space-compatible)
- `requirements-space.txt`: Dependencies (no GPU requirement)
- `README.md`: Space card

---

## Repository Structure

```
MidasMap/
├── app.py                          # Gradio demo app
├── train_final.py                  # Train on all 10 images
├── train.py                        # Deprecated: old training script
├── predict.py                      # Inference on new images
├── evaluate_loocv.py               # Leave-one-image-out evaluation
├── config/
│   └── config.yaml                 # Training configuration
├── src/
│   ├── model.py                    # ImmunogoldCenterNet architecture
│   ├── loss.py                     # Focal loss + Smooth L1
│   ├── dataset.py                  # Hard mining dataset loader
│   ├── preprocessing.py            # Image loading + normalization
│   └── inference.py                # Sliding window + NMS
├── scripts/
│   ├── run_local.sh                # Launch local Gradio app
│   ├── docker_entrypoint.sh        # Docker startup
│   ├── visualize_loss_functions.py # Loss function plots
│   ├── visualize_patch_mining.py   # Training strategy diagram
│   ├── visualize_training_phases.py # 3-phase schedule plot
│   ├── generate_augmentation_examples.py # Data augmentation examples
│   └── [other visualization/analysis scripts]
├── huggingface-space/
│   ├── app.py                      # Space app
│   └── requirements-space.txt
├── "Max Planck Data"/              # Data directory (not in repo)
├── checkpoints/                    # Model checkpoints
│   └── final/
│       ├── phase1.pth
│       ├── phase2.pth
│       ├── phase3_*.pth
│       └── final_model.pth
├── results/
│   ├── diagrams/                   # Generated visualization plots
│   └── loocv_metrics.json          # Evaluation results
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md (this file)
```

---

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

---

## Author

**Anik Sahai** — End-to-end development: data preparation, model architecture, training, evaluation, demo app, and deployment.

---

## Key Technical Insights

### Why CornerNet Focal Loss?

Standard BCE with 23,000:1 negative:positive ratio → model learns all-zeros. Focal loss:
1. Amplifies gradient for easy negatives (via `p^α` term)
2. Down-weights penalty near true particles (via `(1-GT)^β` reduction)
3. Forces model to focus on hard negatives and true positives

### Why Patch-Based Hard Mining?

Full sliding window → sparse particles at <0.1% density → model rarely sees particles. Hard mining:
1. Guarantees particle in 70% of patches
2. Forces model to learn detection despite sparsity
3. Balanced with 30% random background for false positive reduction

### Why 3-Phase Training?

With only 10 images, overfitting is extreme. Progressive unfreezing:
1. Phase 1: Freeze encoder → learn from pre-trained features only
2. Phase 2: Unfreeze deep layers → adapt to domain
3. Phase 3: Full fine-tune → polish with conservative LRs

Result: F1=0.943 LOOCV (vs. 0.85 with standard training)

### Why Sub-Pixel Accuracy?

Particles are ~6-12 nm but image grid is 512 px = 277 nm/px. Offset head predicts ±0.5 px correction via Gaussian peak extraction, achieving <1 nm accuracy.

---

## Contact & Support

For issues, questions, or contributions, please open a GitHub issue or contact the author.
