#!/bin/bash
#SBATCH --job-name=immunogold_eval
#SBATCH --partition=gpu          # TODO: adjust
#SBATCH --gres=gpu:4             # load multiple models in parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-9              # one job per test fold
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate immunogold

SYNAPSE_IDS=("S1" "S4" "S7" "S8" "S13" "S15" "S22" "S25" "S27" "S29")
FOLD_NAME=${SYNAPSE_IDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Ensemble evaluation ==="
echo "Date: $(date)"
echo "Test fold: ${FOLD_NAME}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

# Run evaluation for this fold
# The evaluate_loocv.py script handles loading all ensemble members
python -c "
import yaml
import torch
import numpy as np
from pathlib import Path
from src.preprocessing import discover_synapse_data, load_synapse
from src.model import ImmunogoldCenterNet
from src.ensemble import ensemble_predict
from src.heatmap import extract_peaks
from src.postprocess import apply_structural_mask_filter, cross_class_nms, sweep_confidence_threshold
from src.evaluate import match_detections_to_gt
from src.visualize import overlay_annotations

with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda')
fold_id = '${FOLD_NAME}'
records = discover_synapse_data(cfg['data']['root'], cfg['data']['synapse_ids'])
match_radii = {k: float(v) for k, v in cfg['evaluation']['match_radii_px'].items()}

# Load test data
test_record = [r for r in records if r.synapse_id == fold_id][0]
test_data = load_synapse(test_record)

# Load val data for threshold tuning
synapse_ids = cfg['data']['synapse_ids']
test_idx = synapse_ids.index(fold_id)
val_idx = (test_idx + 1) % len(synapse_ids)
val_record = [r for r in records if r.synapse_id == synapse_ids[val_idx]][0]
val_data = load_synapse(val_record)

# Load all ensemble models
models = []
for seed_idx in range(cfg['training']['n_seeds']):
    seed = seed_idx + 42
    fold_dir = Path('checkpoints') / f'fold_{fold_id}_seed{seed}'
    for epoch in cfg['training']['n_snapshot_epochs']:
        ckpt_path = fold_dir / f'phase3_{epoch}.pth'
        if not ckpt_path.exists():
            ckpt_path = fold_dir / 'phase3_best.pth'
        if not ckpt_path.exists():
            continue
        model = ImmunogoldCenterNet(
            bifpn_channels=cfg['model']['bifpn_channels'],
            bifpn_rounds=cfg['model']['bifpn_rounds'],
        )
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        models.append(model)

print(f'Loaded {len(models)} ensemble members for fold {fold_id}')

# Ensemble inference with TTA
test_hm, test_off = ensemble_predict(models, test_data['image'], device, use_tta=True)

# Extract and post-process
hm_t = torch.from_numpy(test_hm)
off_t = torch.from_numpy(test_off)
dets = extract_peaks(hm_t, off_t, stride=2, conf_threshold=0.1)
if test_data['mask'] is not None:
    dets = apply_structural_mask_filter(dets, test_data['mask'])
dets = cross_class_nms(dets, cfg['postprocessing']['cross_class_nms_distance_px'])

# Evaluate
has_6nm = fold_id not in cfg['data'].get('incomplete_6nm', [])
results = match_detections_to_gt(
    dets,
    test_data['annotations'].get('6nm', np.empty((0, 2))),
    test_data['annotations'].get('12nm', np.empty((0, 2))),
    match_radii,
)

for cls in ['6nm', '12nm', 'overall']:
    r = results[cls]
    note = ' (N/A)' if cls == '6nm' and not has_6nm else ''
    print(f'{cls}: F1={r[\"f1\"]:.3f}, P={r[\"precision\"]:.3f}, R={r[\"recall\"]:.3f}{note}')

# Save visualization
overlay_annotations(
    test_data['image'], test_data['annotations'],
    title=f'Fold {fold_id} — Mean F1={results[\"mean_f1\"]:.3f}',
    save_path=Path('results/per_fold_predictions') / f'{fold_id}_ensemble.png',
    predictions=dets,
)
print(f'Mean F1: {results[\"mean_f1\"]:.3f}')
"

echo "=== Evaluation complete for fold ${FOLD_NAME} ==="
