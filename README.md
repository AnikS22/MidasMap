# MidasMap

MidasMap detects **6 nm (AMPA)** and **12 nm (NR1/NMDA)** immunogold particles in FFRIL TEM synapse images using a CenterNet-style detector with a CEM500K-pretrained encoder.

## Author

This repository reflects end-to-end solo work by **Anik Sahai**: data prep, model design, training, evaluation, demo app, and deployment tooling.

## Headline Results

| Metric | Value |
|---|---|
| LOOCV mean F1 (8 usable folds) | **0.943** |
| 6 nm F1 | 0.944 |
| 12 nm F1 | 0.909 |
| Parameters | 24.4M |

Evaluation is leave-one-image-out CV on 10 synapse images (453 labeled particles total).

## Repo Layout

- `src/` model, losses, training/inference utilities
- `app.py` Gradio inference app
- `train.py` and `train_final.py` training entry points
- `predict.py` and `evaluate_loocv.py` evaluation/inference scripts
- `config/` training config
- `scripts/` local run + deployment helpers
- `huggingface-space/` Space-ready app package

## Quick Start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run local app

```bash
./scripts/run_local.sh
```

Open `http://127.0.0.1:7860`.

### 3) Run prediction script

```bash
python predict.py --help
```

## Model Architecture (high level)

1. Sliding-window inference (`512x512`, overlap) on full TEM images
2. ResNet-50 encoder initialized from CEM500K
3. BiFPN feature fusion
4. Heatmap + offset heads (CenterNet-style keypoint detection)
5. Peak extraction + NMS for final detections

## Training

Final training run:

```bash
python train_final.py --config config/config.yaml --device cuda:0
```

LOOCV evaluation:

```bash
python evaluate_loocv.py --help
```

## Deploy

- Local container: `docker compose up --build`
- Hugging Face Space instructions: `docs/DEPLOY.md`

## Citation

```bibtex
@software{midasmap2026,
  title={MidasMap: Automated Immunogold Particle Detection for TEM Synapse Images},
  author={Sahai, Anik},
  year={2026},
  url={https://github.com/AnikS22/MidasMap}
}
```
