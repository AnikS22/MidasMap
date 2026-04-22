# Project Setup & Development Guide

## Development Environment

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

See `requirements.txt` for complete list. Key packages:
- PyTorch (GPU/CPU)
- Torchvision (ResNet)
- Gradio (Web UI)
- OpenCV, Pillow, NumPy (Image processing)
- PyYAML (Config parsing)

## Project Structure

```
src/              Model, losses, dataset, preprocessing, inference
config/           Training configuration (YAML)
scripts/          Utility scripts (visualization, deployment)
huggingface-space/ Web app for HF Spaces deployment
```

See `CODEBASE.md` for detailed documentation of every file.

## Training

### 1. Configuration
Edit `config/config.yaml` with desired hyperparameters.

### 2. Run Training
```bash
python train_final.py --config config/config.yaml --device cuda:0
```

### 3. Checkpoints
Models saved to `checkpoints/final/`:
- `phase1.pth` (40 epochs)
- `phase2.pth` (40 epochs)
- `phase3_*.pth` (intermediate checkpoints)
- `final_model.pth` (140 epochs, ready for inference)

## Evaluation

Leave-one-image-out cross-validation:
```bash
python evaluate_loocv.py --config config/config.yaml --device cuda:0
```

Results saved to `results/loocv_metrics.json`.

## Inference

### Command Line
```bash
python predict.py \
  --image path/to/image.tif \
  --checkpoint checkpoints/final/final_model.pth \
  --output detections.csv
```

### Web Interface
```bash
./scripts/run_local.sh
```

Open `http://127.0.0.1:7860`.

### Docker
```bash
docker compose up --build
```

## Code Quality

- No comments in code (self-documenting variable names)
- Type hints on public APIs
- Deterministic random seeding (seed=42)
- Gradient clipping for stability

## Data Format

### Images
- Format: 8-bit or 16-bit grayscale TIFF
- Size: 2048×2048 pixels
- Directory: `"Max Planck Data/Gold Particle Labelling/analyzed synapses/"`

### Annotations
- Format: JSON with particle coordinates
- One JSON per image (same name, .json extension)
- Structure: `{"particles": [{"x": float, "y": float, "class": "6nm"|"12nm"}, ...]}`

## Key Files to Understand

1. **src/model.py** — Neural network architecture (ResNet-50 + BiFPN + heads)
2. **src/loss.py** — CornerNet focal loss + Smooth L1 offset loss
3. **src/dataset.py** — Hard mining data loader (70% particle-centered, 30% random)
4. **train_final.py** — 3-phase training (frozen → deep → full fine-tune)
5. **predict.py** — Inference pipeline (sliding window + peak extraction + NMS)

See `CODEBASE.md` for comprehensive documentation of all files.

## Performance

Target: F1 score ~0.94 on LOOCV evaluation
- 6 nm particles: ~0.94 F1
- 12 nm particles: ~0.91 F1
- 453 total labeled particles across 10 synapse images

## References

- CenterNet: https://arxiv.org/abs/1904.07850
- BiFPN: https://arxiv.org/abs/1912.03768
- CEM500K: https://www.biorxiv.org/content/10.1101/2022.03.17.484749v1

