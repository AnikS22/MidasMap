# MidasMap Codebase Documentation

Complete walkthrough of every file and folder in the project.

---

## Directory Structure & Purpose

### `src/` — Core Model and Training Code

#### `src/model.py`
Defines `ImmunogoldCenterNet` class — the complete neural network architecture.

**Components:**
- `__init__`: Initializes ResNet-50 encoder (optionally from CEM500K pretrained weights), BiFPN, heatmap head, and offset head
- `freeze_encoder()`: Disables gradient computation for entire encoder (Phase 1 training)
- `unfreeze_deep_layers()`: Enables gradients for ResNet layer3, layer4, and all heads (Phase 2 training)
- `unfreeze_all()`: Enables gradients everywhere (Phase 3 training)
- `forward(x)`: Runs image through encoder → BiFPN → dual heads, returns heatmap predictions and offset predictions

**Flow:**
Input image (B, 1, 256, 256) → ResNet-50 (outputs multi-scale features) → BiFPN (fuses features bidirectionally 3 times) → Two parallel heads:
- Heatmap head: (B, 2, 64, 64) — 2 channels (6nm, 12nm), 64×64 spatial grid (1/4 resolution of input due to stride)
- Offset head: (B, 2, 64, 64) — predicted x,y sub-pixel offsets

#### `src/loss.py`
Implements three loss functions used during training.

**`cornernet_focal_loss(pred, gt, alpha, beta)`:**
- Takes predicted heatmap and ground truth Gaussian heatmap
- Computes positive loss at GT peaks: `-log(pred) × (1-pred)^alpha` (rewards high confidence)
- Computes negative loss elsewhere: `-log(1-pred) × pred^alpha × (1-GT)^beta` (penalizes high confidence, with penalty reduction near peaks)
- Returns scalar loss normalized by number of positive pixels
- Handles 23,000:1 negative:positive ratio via focal exponent and penalty reduction

**`offset_loss(pred_offsets, gt_offsets, mask)`:**
- Takes predicted offsets and ground truth offsets
- Mask only selects pixels at annotated particle centers
- Uses PyTorch's `F.smooth_l1_loss` (quadratic for small errors, linear for large)
- Returns mean loss across masked positions

**`total_loss(heatmap_pred, heatmap_gt, offset_pred, offset_gt, offset_mask, lambda_offset)`:**
- Combines both losses: `total = L_heatmap + lambda_offset × L_offset`
- Returns tuple: (total loss, heatmap loss value, offset loss value)
- Used during training to get all three metrics

#### `src/dataset.py`
Defines `ImmunogoldDataset` class — data loader with hard mining.

**`__init__` parameters:**
- `records`: List of image+annotation dictionaries
- `fold_id`: Which image to exclude (for LOOCV); None or "__NONE__" means use all
- `mode`: "train" (hard mining sampling) or "val" (sliding window)
- `patch_size`: 256 (patch size extracted)
- `stride`: 256 (for validation, non-overlapping patches)
- `hard_mining_fraction`: 0.7 (70% hard mining, 30% random background)
- `samples_per_epoch`: Number of patches to generate per epoch

**Core logic:**
- Stores all particles from all images in `self.particles`
- For each epoch, samples `samples_per_epoch` patches:
  - 70% of patches: randomly select a particle, extract 256×256 patch centered on it
  - 30% of patches: random location in a random image
- Generates ground truth:
  - Heatmap: 2D Gaussian centered at each particle center (Gaussian width depends on particle class: `sigma=1.0 for 6nm, 1.5 for 12nm`)
  - Offsets: x,y sub-pixel correction needed to reach true center from grid point
  - Offset mask: 1 only at grid points where particles exist (integer grid locations)
- Returns dict with keys: "image", "heatmap", "offsets", "offset_mask", "conf_map"

**`__getitem__` returns:**
- "image": (1, 256, 256) normalized grayscale patch
- "heatmap": (2, 64, 64) Gaussian heatmap targets (downsampled 4× from patch)
- "offsets": (2, 64, 64) sub-pixel offsets at grid points
- "offset_mask": (64, 64) boolean mask marking valid offset positions
- "conf_map": (2, 64, 64) confidence weights (used for pseudo-labeling)

#### `src/preprocessing.py`
Image loading and data discovery utilities.

**`discover_synapse_data(root, synapse_ids)`:**
- Scans directory tree for image+annotation pairs
- Returns list of dicts with keys: "image_path", "anno_path", "synapse_id"
- Filters to only specified synapse_ids

**`load_synapse(image_path, anno_path, target_size)`:**
- Loads TIF image and JSON annotations
- Normalizes image to [0, 1] range
- Resizes image if specified
- Parses JSON to extract particle coordinates and classes
- Returns dict: "image" (numpy array), "particles_6nm" (list of (x,y)), "particles_12nm"

#### `src/inference.py`
Sliding window inference and peak extraction for full images.

**`sliding_window_inference(model, image, window_size, stride, device)`:**
- Breaks full 2048×2048 image into overlapping 512×512 patches
- Runs each patch through model
- Combines predictions from all patches (blends overlapping regions)
- Returns full-resolution heatmap and offset predictions

**`extract_peaks(heatmap, offset_map, conf_threshold)`:**
- Finds local maxima in heatmap (peaks)
- For each peak, applies Gaussian fitting to refine sub-pixel location
- Uses offset map to correct sub-pixel coordinates
- Filters by confidence threshold
- Returns list of (x, y, confidence) detections

**`nms(detections, iou_threshold)`:**
- Non-maximum suppression
- Removes duplicate detections within iou_threshold
- Returns refined detections

---

### `config/` — Configuration Files

#### `config/config.yaml`
YAML file specifying all training hyperparameters and data paths.

**Sections:**
- `data`: root directory, synapse IDs, patch size, stride
- `model`: whether to use CEM500K pretrained weights, BiFPN channels, BiFPN rounds
- `training`: batch size, hard mining fraction, learning rates for each phase, number of epochs
- `heatmap`: Gaussian widths (sigmas) for each particle class
- `optimizer`: Adam weight decay, gradient clipping
- `augmentation`: rotation range, flip probability, brightness range, noise level

All training scripts read this file with `yaml.safe_load()`.

---

### `scripts/` — Utility and Visualization Scripts

#### `scripts/run_local.sh`
Shell script to launch local Gradio app.

**Does:**
- Activates Python venv
- Runs `python app.py`
- Starts Gradio server on localhost:7860

#### `scripts/docker_entrypoint.sh`
Entrypoint for Docker container.

**Does:**
- Sets up environment inside container
- Activates venv
- Runs `python app.py` with GPU support

#### `scripts/visualize_loss_functions.py`
Generates visualization plots of loss functions.

**Creates image:** `results/diagrams/loss_functions.png`

**Shows:**
- CornerNet focal loss curves (positive, negative, reduced negative)
- Smooth L1 vs L1 vs L2 curves
- Legend and mathematical definitions

#### `scripts/visualize_patch_mining.py`
Generates diagram of training strategy.

**Creates image:** `results/diagrams/patch_based_mining.png`

**Shows:**
- Full TEM image with sparse particles
- Hard mining patch (70%) centered on particle
- Random background patch (30%)
- Pie chart of 70/30 distribution
- Training loop pseudocode

#### `scripts/visualize_training_phases.py`
Generates diagram of 3-phase training schedule.

**Creates image:** `results/diagrams/training_phases.png`

**Shows:**
- Phase 1: frozen encoder, learning rate 1e-3 (40 epochs)
- Phase 2: unfrozen deep layers, graduated LRs (40 epochs)
- Phase 3: full fine-tune, lowest LRs (60 epochs)
- Loss curves across phases

#### `scripts/generate_augmentation_examples.py`
Generates visual examples of data augmentations applied during training.

**Creates images:** `results/diagrams/augmentations/`
- 90° rotation
- Horizontal flip
- Vertical flip
- Small rotation ±10°
- Brightness increase
- Gaussian noise
- Gaussian blur
- Elastic deformation
- Combined example

Each image shows the augmentation applied to a real TEM synapse.

#### `scripts/visualize_augmentations.py`
(Similar to generate_augmentation_examples.py, alternative version)

#### `scripts/visualize_pipeline.py`
Generates full pipeline diagram from input image to final detections.

**Shows:**
- Input TEM image
- Sliding window patches
- Model architecture layers
- Heatmap and offset outputs
- Peak extraction
- NMS
- Final detections

#### `scripts/visualize_real_data.py`
(Development script for inspecting real data)

---

### Root-Level Python Files

#### `app.py`
Gradio web interface for interactive inference.

**Components:**
- Loads pre-trained model from `checkpoints/final/final_model.pth`
- Creates Gradio interface with:
  - Image upload input
  - Confidence threshold slider
  - Run button
  - Results display (annotated image + CSV table)
- Internally calls `predict.py` logic

**Flow:**
1. User uploads TEM image
2. User adjusts confidence threshold
3. User clicks "Detect Particles"
4. Model runs inference → extracts peaks → applies NMS
5. Results shown: annotated image with circles + table of (x, y, class, confidence)

#### `train_final.py`
Trains final production model on ALL 10 images (no holdout).

**Does:**
1. Loads config from `config/config.yaml`
2. Discovers all synapse images+annotations
3. Creates `ImmunogoldDataset` with fold_id=None (all images)
4. Creates `ImmunogoldCenterNet` model
5. **Phase 1 (40 epochs):** Freezes encoder, trains only heads with LR=1e-3
6. **Phase 2 (40 epochs):** Unfreezes layer3+4, uses graduated LRs
7. **Phase 3 (60 epochs):** Unfreezes all, uses lowest graduated LRs with cosine annealing
8. Saves checkpoints after each phase
9. Saves final model to `checkpoints/final/final_model.pth`

**Key loop (repeated 140 times):**
- Iterate through batches
- Forward pass through model
- Compute total loss via `total_loss()`
- Backward pass + gradient clip + optimizer step
- Record loss
- Every 10 epochs: print loss and elapsed time

#### `train.py`
Older training script (deprecated, kept for reference).

#### `predict.py`
Inference script for detecting particles in new images.

**Does:**
1. Loads model checkpoint
2. Loads image (TIF format)
3. Normalizes image
4. Runs sliding window inference (512×512 patches, 50% overlap)
5. Extracts peaks from heatmap + applies offsets
6. Performs NMS to remove duplicates
7. Filters by confidence threshold
8. Saves results to CSV (x, y, class, confidence)
9. Optionally saves annotated image with circles

**Command:**
```
python predict.py --image path/to/image.tif --checkpoint path/to/model.pth --output results.csv
```

**Output CSV format:**
```
x,y,class,confidence
512.3,256.7,6nm,0.95
768.1,384.2,12nm,0.87
```

#### `evaluate_loocv.py`
Leave-one-image-out cross-validation evaluation.

**Does:**
1. For each of 10 images (i=1 to 10):
   a. Load dataset excluding image i
   b. Train model from scratch (140 epochs, 3 phases)
   c. Evaluate on held-out image i
   d. Record F1, precision, recall for 6nm and 12nm separately
2. Average metrics across all 10 folds
3. Save results to `results/loocv_metrics.json`

**Takes hours to run** (trains 10 separate models).

**Output:** Average F1=0.943, 6nm F1=0.944, 12nm F1=0.909

---

### `huggingface-space/` — Space Deployment

#### `huggingface-space/app.py`
Same as root `app.py` but configured for Hugging Face Spaces (CPU-friendly, no GPU assumed).

#### `huggingface-space/requirements-space.txt`
Dependencies for Spaces (lighter than main requirements.txt, no GPU libraries).

#### `huggingface-space/README.md`
Space card describing the model and how to use it.

---

### `checkpoints/` — Model Files

#### `checkpoints/final/final_model.pth`
Saved PyTorch model (trained on all 10 images).

**Contains:**
- `model_state_dict`: All weights and biases
- `epoch`: 140 (full training completed)
- `config`: YAML config used for training

Loaded by `app.py` and `predict.py` for inference.

---

### `results/` — Output Directory (Generated at Runtime)

#### `results/diagrams/`
Generated visualization images:
- `loss_functions.png`: Loss curves
- `patch_based_mining.png`: Training strategy diagram
- `training_phases.png`: Phase schedule
- `augmentations/`: Individual augmentation examples

#### `results/loocv_metrics.json`
JSON file with LOOCV evaluation results.

**Format:**
```json
{
  "fold_1": {
    "6nm": {"f1": 0.944, "precision": 0.95, "recall": 0.94},
    "12nm": {"f1": 0.909, "precision": 0.92, "recall": 0.90}
  },
  "fold_2": {...},
  ...
  "average": {
    "6nm": {"f1": 0.944, ...},
    "12nm": {"f1": 0.909, ...}
  }
}
```

---

### `"Max Planck Data/"` — Input Data (Not in Repo)

#### `"Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1.tif"`
Grayscale TEM synapse image (2048×2048 px).

#### `"Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1.json"`
Annotations for image S1.

**Format:**
```json
{
  "particles": [
    {"x": 512.3, "y": 256.7, "class": "6nm"},
    {"x": 768.1, "y": 384.2, "class": "12nm"},
    ...
  ]
}
```

10 synapse images (S1-S10) with ~453 total labeled particles.

---

### Other Root Files

#### `Dockerfile`
Docker container definition.

**Base:** `nvidia/cuda:11.8-runtime-ubuntu22.04`

**Does:**
- Installs Python 3.10
- Copies code + requirements
- Installs dependencies
- Exposes port 7860
- Runs `scripts/docker_entrypoint.sh`

#### `docker-compose.yml`
Docker Compose configuration for local deployment.

**Services:**
- Mounts code directory
- Mounts data directory
- Allocates GPU access
- Exposes port 7860

#### `requirements.txt`
Python dependencies.

**Key packages:**
- `torch`: PyTorch (GPU/CPU)
- `torchvision`: Vision models (ResNet)
- `gradio`: Web interface
- `pillow`: Image processing
- `pyyaml`: Config parsing
- `opencv-python`: Image I/O

#### `environment.yml`
Conda environment file (alternative to venv + requirements.txt).

#### `CLAUDE.md`
Development setup guide and project configuration.

#### `.gitignore`
Specifies which files to ignore in git (data, checkpoints, venv, etc.).

#### `.dockerignore`
Specifies which files to ignore when building Docker image.

---

## Complete Data Flow

### Training Flow

```
config/config.yaml (hyperparameters)
    ↓
discover_synapse_data() [preprocessing.py]
    → scans "Max Planck Data/" directory
    → finds 10 image+annotation pairs
    ↓
ImmunogoldDataset [dataset.py]
    → hard mining sampler: 70% particle-centered, 30% random
    → generates Gaussian heatmap targets
    → computes sub-pixel offsets
    ↓
DataLoader (batch size 16, shuffle=True)
    ↓
train_final.py
    ↓
Phase 1: Frozen encoder
    → ImmunogoldCenterNet (encoder frozen)
    → forward pass: image → heatmap, offsets
    → total_loss() [loss.py]
        → cornernet_focal_loss() on heatmap
        → offset_loss() on offsets
    → backward + optimizer step (only head parameters)
    → 40 epochs
    ↓
Phase 2: Unfreeze deep layers
    → same loop but with different optimizer (graduated LRs)
    → 40 epochs
    ↓
Phase 3: Full fine-tune
    → same loop but all parameters trainable (lowest LRs)
    → cosine annealing schedule
    → save checkpoint every 10 epochs
    → 60 epochs
    ↓
Save checkpoints/final/final_model.pth
```

### Inference Flow (for new image)

```
new_image.tif
    ↓
predict.py
    ↓
Load model from checkpoints/final/final_model.pth
    ↓
Sliding window inference [inference.py]
    → break 2048×2048 into 512×512 patches (50% overlap)
    → run each patch through model
    → blend overlapping regions
    → get full heatmap + offsets (same size as input)
    ↓
extract_peaks() [inference.py]
    → find local maxima in heatmap (above conf_threshold)
    → Gaussian fit to refine sub-pixel location
    → apply offset map
    ↓
nms() [inference.py]
    → remove duplicates (IoU < 0.5)
    ↓
Save results.csv (x, y, class, confidence)
```

### Web Interface Flow

```
User uploads image via app.py
    ↓
Gradio receives file
    ↓
app.py calls predict.py logic
    ↓
Model runs inference
    ↓
Gradio displays:
    - Annotated image (circles at detections)
    - Results table (x, y, class, confidence)
    ↓
User downloads CSV or adjusts threshold and re-runs
```

---

## Key Concepts Explained

### Hard Mining (dataset.py)

Standard sliding window on 2048×2048 image with particle density <0.1% means model rarely sees particles. Solution:
- 70% of patches: randomly pick a particle, extract 256×256 around it → guarantees particle in patch
- 30% of patches: random location → trains model to reduce false positives on background

Result: Model learns despite extreme class imbalance.

### CornerNet Focal Loss (loss.py)

23,000:1 negative:positive ratio → standard BCE learns all zeros. Solution:
- Positive: `-log(pred) × (1-pred)²` → rewards high confidence at true centers
- Negative: `-log(1-pred) × pred² × (1-GT)⁴` → penalizes high confidence away from particles, with penalty reduction near peaks
- Penalty reduction `(1-GT)⁴` → near true particles, reduces false positive penalty (focuses on hard negatives)

### Smooth L1 Loss (loss.py)

For sub-pixel offset regression:
- Quadratic for small errors: `0.5 × error²` → sensitive to small corrections
- Linear for large errors: `|error|` → robust to outliers

Sub-pixel accuracy achieved via:
1. Model predicts offsets in [-0.5, 0.5]
2. Gaussian peak fitting refines location
3. Result: ±0.5 pixel accuracy (±270 nm at 512 px = 277 nm/px)

### 3-Phase Training

With only 10 images, overfitting is extreme:
- Phase 1: Freeze encoder → learn from pre-trained features only (prevents overfitting)
- Phase 2: Unfreeze deep layers → adapt to domain (fine-tune deep encoder)
- Phase 3: Full fine-tune → polish entire model with conservative LRs (fine-grain adjustments)

Result: F1=0.943 LOOCV (vs. 0.85-0.90 with standard training)

### LOOCV Evaluation (evaluate_loocv.py)

Standard k-fold CV where k=number_of_images=10. Each fold:
- Train on 9 images
- Test on 1 held-out image

More rigorous than random split for small datasets. Avoids leakage from same-synapse images.

Result: Average metrics across all 10 folds.

---

## Summary Table

| File | Purpose | Key Functions |
|---|---|---|
| `src/model.py` | Neural network architecture | `ImmunogoldCenterNet`, `freeze_encoder()`, `forward()` |
| `src/loss.py` | Loss functions | `cornernet_focal_loss()`, `offset_loss()`, `total_loss()` |
| `src/dataset.py` | Data loader with hard mining | `ImmunogoldDataset`, `__getitem__()` (hard mining logic) |
| `src/preprocessing.py` | Image loading | `discover_synapse_data()`, `load_synapse()` |
| `src/inference.py` | Sliding window + peak extraction | `sliding_window_inference()`, `extract_peaks()`, `nms()` |
| `train_final.py` | Training script (all 10 images) | 3-phase training loop (140 epochs) |
| `train.py` | Older training script | (deprecated) |
| `predict.py` | Inference on new images | Loads model, runs sliding window, saves CSV |
| `evaluate_loocv.py` | Leave-one-image-out CV | Trains 10 models, reports average F1 |
| `app.py` | Web interface | Gradio UI for interactive inference |
| `config/config.yaml` | Hyperparameters | All training config (LRs, batch size, epochs, etc.) |
| `scripts/*.py` | Visualizations | Generate diagrams of loss, training, augmentations |
| `Dockerfile` | Container definition | Build Docker image with GPU support |
| `docker-compose.yml` | Compose config | Local deployment with GPU + volume mounts |
| `requirements.txt` | Dependencies | PyTorch, Gradio, OpenCV, YAML, etc. |

