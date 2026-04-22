#!/bin/bash
#SBATCH --job-name=immunogold_train
#SBATCH --partition=gpu          # TODO: adjust to your GPU partition
#SBATCH --gres=gpu:1             # single A100 or V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-49             # 10 folds x 5 seeds = 50 jobs
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate immunogold

# Map array task ID to fold and seed
SYNAPSE_IDS=("S1" "S4" "S7" "S8" "S13" "S15" "S22" "S25" "S27" "S29")
FOLD_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((SEED_IDX + 42))
FOLD_NAME=${SYNAPSE_IDS[$FOLD_IDX]}

echo "=== Training immunogold CenterNet ==="
echo "Date: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Fold: ${FOLD_NAME} (idx ${FOLD_IDX})"
echo "Seed: ${SEED} (idx ${SEED_IDX})"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python train.py \
    --fold "${FOLD_NAME}" \
    --seed "${SEED}" \
    --config config/config.yaml \
    --device cuda:0

echo "=== Training complete for fold=${FOLD_NAME} seed=${SEED} ==="
