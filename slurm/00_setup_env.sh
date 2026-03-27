#!/bin/bash
#SBATCH --job-name=immunogold_setup
#SBATCH --partition=cpu          # TODO: change to your CPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -euo pipefail

echo "=== Setting up immunogold detection environment ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"

# Create logs directory
mkdir -p logs checkpoints results weights

# --- Conda environment ---
if ! conda info --envs | grep -q "immunogold"; then
    echo "Creating conda environment..."
    conda env create -f environment.yml
else
    echo "Updating existing environment..."
    conda env update -f environment.yml --prune
fi

# Activate
source activate immunogold || conda activate immunogold

# Verify key packages
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import albumentations; print(f'Albumentations {albumentations.__version__}')"
python -c "import skimage; print(f'scikit-image {skimage.__version__}')"
python -c "import tifffile; print(f'tifffile {tifffile.__version__}')"

# --- Download CEM500K weights ---
WEIGHTS_DIR="weights"
CEM500K_FILE="${WEIGHTS_DIR}/cem500k_mocov2_resnet50.pth.tar"
CEM500K_URL="https://zenodo.org/records/6453140/files/cem500k_mocov2_resnet50_200ep.pth.tar?download=1"

if [ ! -f "${CEM500K_FILE}" ]; then
    echo "Downloading CEM500K pretrained weights from Zenodo..."
    wget -O "${CEM500K_FILE}" "${CEM500K_URL}"
    echo "Download complete: $(ls -lh ${CEM500K_FILE})"
else
    echo "CEM500K weights already exist: $(ls -lh ${CEM500K_FILE})"
fi

# --- Verify data access ---
DATA_ROOT="/Users/mpcr/Downloads/Max Planck Data/Gold Particle Labelling"
# TODO: Update DATA_ROOT to match your HPC data location, e.g.:
# DATA_ROOT="/scratch/${USER}/immunogold_data/Gold Particle Labelling"

if [ -d "${DATA_ROOT}" ]; then
    echo "Data directory accessible: ${DATA_ROOT}"
    echo "Synapse folders: $(ls "${DATA_ROOT}/analyzed synapses/")"
else
    echo "WARNING: Data directory not found at ${DATA_ROOT}"
    echo "Please update DATA_ROOT in config/config.yaml and this script"
fi

# --- Quick validation ---
echo ""
echo "Running data validation..."
python -c "
from src.preprocessing import discover_synapse_data
import yaml
with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)
records = discover_synapse_data(cfg['data']['root'], cfg['data']['synapse_ids'])
total_6nm = sum(len(r.csv_6nm_paths) for r in records)
total_12nm = sum(len(r.csv_12nm_paths) for r in records)
print(f'Discovered {len(records)} synapses, {total_6nm} 6nm CSVs, {total_12nm} 12nm CSVs')
for r in records:
    print(f'  {r.synapse_id}: image={r.image_path.name}, 6nm={r.has_6nm}, 12nm={r.has_12nm}')
"

echo ""
echo "=== Setup complete ==="
