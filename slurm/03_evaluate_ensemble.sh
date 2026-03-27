#!/bin/bash
#SBATCH --job-name=immunogold_eval
#SBATCH --partition=gpu          # TODO: adjust
#SBATCH --gres=gpu:1             # single GPU per fold (model loaded sequentially)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9              # one job per test fold
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

set -euo pipefail
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate immunogold

SYNAPSE_IDS=("S1" "S4" "S7" "S8" "S13" "S15" "S22" "S25" "S27" "S29")
FOLD_NAME=${SYNAPSE_IDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Ensemble evaluation ==="
echo "Date: $(date)"
echo "Test fold: ${FOLD_NAME}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Evaluate this single fold using the same script as local evaluation
# evaluate_loocv.py handles loading all ensemble members per fold
python evaluate_loocv.py \
    --config config/config.yaml \
    --ensemble-dir checkpoints \
    --device cuda:0 \
    --use-tta \
    --fold "${FOLD_NAME}" \
    --output "results/per_fold_predictions/${FOLD_NAME}_metrics.csv"

echo "=== Evaluation complete for fold ${FOLD_NAME} ==="
