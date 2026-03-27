#!/bin/bash
# Master submission script for the full immunogold detection pipeline.
# Submits all stages with dependency chaining so they run sequentially.
#
# Usage:
#   bash slurm/04_full_pipeline.sh

set -euo pipefail

echo "=== Immunogold Detection Pipeline ==="
echo "Date: $(date)"
echo "Submitting all pipeline stages..."

# Create log directory
mkdir -p logs

# Stage 0: Environment setup (single job)
JOB0=$(sbatch --parsable slurm/00_setup_env.sh)
echo "Stage 0 (setup):     Job ${JOB0}"

# Stage 1: LodeStar pseudo-labels (array job 0-1, depends on setup)
# afterok: on a single job waits for that one job
JOB1=$(sbatch --parsable --dependency=afterok:${JOB0} slurm/01_lodestar_pseudolabels.sh)
echo "Stage 1 (lodestar):  Job ${JOB1} (array 0-1, depends on ${JOB0})"

# Stage 2: Training — 50 parallel jobs (depends on ALL lodestar tasks)
# aftercorr: waits for all tasks in the array job to complete successfully
JOB2=$(sbatch --parsable --dependency=aftercorr:${JOB1} slurm/02_train_single_fold.sh)
echo "Stage 2 (training):  Job ${JOB2} (array 0-49, depends on ${JOB1})"

# Stage 3: Ensemble evaluation — 10 parallel jobs (depends on ALL training tasks)
JOB3=$(sbatch --parsable --dependency=aftercorr:${JOB2} slurm/03_evaluate_ensemble.sh)
echo "Stage 3 (evaluate):  Job ${JOB3} (array 0-9, depends on ${JOB2})"

echo ""
echo "Pipeline submitted successfully!"
echo "Final results job: ${JOB3}"
echo ""
echo "Monitor with:"
echo "  squeue -u \${USER}"
echo "  sacct -j ${JOB0},${JOB1},${JOB2},${JOB3}"
echo ""
echo "Results will be in:"
echo "  results/loocv_metrics.csv"
echo "  results/per_fold_predictions/"
