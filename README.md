# MidasMap

Detects immunogold particles (6 nm and 12 nm) in transmission electron microscopy (TEM) images of synapses.

In immunogold labeling, gold beads of a given size are attached to a specific receptor (here 6 nm for AMPA and 12 nm for NMDA receptors), and someone counts them by hand on each TEM image. MidasMap finds the particles automatically and returns a coordinate, class, and confidence for each one.

Demo: [huggingface.co/spaces/AnikS22/MidasMap](https://huggingface.co/spaces/AnikS22/MidasMap) (Gradio). Weights: [huggingface.co/AnikS22/MidasMap](https://huggingface.co/AnikS22/MidasMap). As of October 2026 the Space shows a runtime error on Hugging Face's free CPU tier; the local app below works.

## Status

| Part | State |
|---|---|
| Model, training, LOOCV evaluation code | Done |
| Trained weights | On Hugging Face (`checkpoints/final/final_model.pth`) |
| Local Gradio app (`app.py`) | Works with the weights above |
| Hugging Face Space | Deployed, currently failing to start |
| Unit tests | `tests/` covers heatmaps, loss, model shapes, matching |
| Training data | Not in this repo (lab data, not mine to publish) |

## Method

- Backbone: ResNet-50 initialized from CEM500K (self-supervised pretraining on electron microscopy images) instead of ImageNet.
- Neck: BiFPN to fuse features across scales, since the two particle sizes differ by 2x.
- Heads, CenterNet-style: a 2-channel heatmap (one per particle class) and a 2-channel sub-pixel offset.
- Loss: CornerNet penalty-reduced focal loss on the heatmap, smooth L1 on the offset. Particles are well under 0.1% of pixels, so plain BCE collapses to predicting background.
- Sampling: 70% of training patches are centered on a particle, 30% are random background.
- Training in three phases (see `config/config.yaml`): frozen encoder (60 epochs), unfreeze layer3/layer4 (60 epochs), full fine-tune with per-layer learning rates (80 epochs).
- Inference: sliding 512x512 windows, peak extraction, cross-class NMS, optional D4 test-time augmentation.

## Results

Dataset: 10 annotated TEM synapse images (2048x2048), 453 hand-labeled particles. Evaluation is leave-one-image-out: train on 9 images, test on the held-out one, repeat 10 times. A prediction counts as correct if Hungarian matching pairs it with a ground-truth particle within 9 px (6 nm) or 15 px (12 nm).

| | F1 |
|---|---|
| 6 nm (AMPA) | 0.944 |
| 12 nm (NMDA) | 0.909 |
| Overall | 0.943 |

These numbers come from my write-up of the LOOCV run. The raw `loocv_metrics.json` from that run is not in the repo, and an earlier summary figure reports 0.928 overall on 429 particles, so treat the overall number as approximate until it is re-run with `evaluate_loocv.py`. Ten images from one source is a small test set; I have not tested on images from other microscopes or labs.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# get the trained weights
huggingface-cli download AnikS22/MidasMap checkpoints/final/final_model.pth \
    --local-dir . --repo-type model

# browser app at http://127.0.0.1:7860
./scripts/run_local.sh

# command line, one image
python predict.py --image path/to/synapse.tif \
    --checkpoint checkpoints/final/final_model.pth \
    --output-dir results/predictions --conf-threshold 0.3
```

Training and evaluation need the annotated data. Set `data.root` in `config/config.yaml`, then:

```bash
python scripts/download_cem500k.py                       # backbone weights
python evaluate_loocv.py --config config/config.yaml     # 10-fold LOOCV
python train_final.py --config config/config.yaml        # final model on all 10 images
pytest tests/
```

`slurm/` has the same steps as SLURM jobs for an HPC cluster. `Dockerfile` and `docker-compose.yml` run the app on a CUDA machine.

## Layout

| Path | What it is |
|---|---|
| `src/model.py` | ResNet-50 + BiFPN + heatmap/offset heads |
| `src/loss.py` | focal loss and offset loss |
| `src/dataset.py` | patch sampling with particle-centered crops |
| `src/ensemble.py`, `src/postprocess.py` | sliding-window inference, TTA, NMS |
| `src/evaluate.py` | Hungarian matching and per-class F1 |
| `train_final.py`, `evaluate_loocv.py`, `predict.py` | entry points |
| `app.py` | Gradio app (local and Hugging Face Space) |
| `scripts/visualize_*.py` | explanatory diagrams; several plot hardcoded illustrative values, not training logs |
| `CODEBASE.md` | file-by-file notes |

## Credits

Built by Anik Sahai with A.D. Henderson / FAU High. Particle annotations were done by hand by neuroscientists who provided the images. MIT licensed, see [LICENSE](LICENSE).
