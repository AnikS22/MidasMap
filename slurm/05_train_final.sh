#!/bin/bash
#SBATCH --job-name=immunogold_final
#SBATCH --partition=shortq7-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_final_%j.out
#SBATCH --error=logs/train_final_%j.err

set -euo pipefail
mkdir -p logs

module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-bxjolrw
eval "$(conda shell.bash hook)"
conda activate immunogold

echo "=== Training final deployable model ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python train_final.py --config config/config.yaml --device cuda:0

echo "=== Final model training complete ==="
