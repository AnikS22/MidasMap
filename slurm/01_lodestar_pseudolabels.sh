#!/bin/bash
#SBATCH --job-name=immunogold_lodestar
#SBATCH --partition=gpu          # TODO: adjust to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1              # 0=6nm, 1=12nm
#SBATCH --output=logs/lodestar_%A_%a.out
#SBATCH --error=logs/lodestar_%A_%a.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate immunogold

BEAD_CLASSES=("6nm" "12nm")
BEAD_CLASS=${BEAD_CLASSES[$SLURM_ARRAY_TASK_ID]}

echo "=== LodeStar pseudo-label generation ==="
echo "Date: $(date)"
echo "Bead class: ${BEAD_CLASS}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"

python -c "
import yaml
from src.preprocessing import discover_synapse_data, load_synapse
from src.lodestar_pipeline import run_lodestar_pipeline

with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)

records = discover_synapse_data(cfg['data']['root'], cfg['data']['synapse_ids'])
synapse_data = [load_synapse(r) for r in records]

run_lodestar_pipeline(
    synapse_data=synapse_data,
    bead_class='${BEAD_CLASS}',
    output_dir=cfg['data']['merged_annotation_dir'],
    tophat_radii=cfg['preprocessing']['tophat_radii'],
    crop_sizes=cfg['lodestar']['crop_sizes'],
    confidence_percentile=cfg['lodestar']['confidence_percentile'],
    confirmation_radius=cfg['lodestar']['confirmation_radius_px'],
    discovery_radius=cfg['lodestar']['discovery_radius_px'],
)
"

echo "=== LodeStar ${BEAD_CLASS} complete ==="
