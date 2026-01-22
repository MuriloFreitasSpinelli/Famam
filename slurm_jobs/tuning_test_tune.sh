#!/bin/bash
#SBATCH --job-name=tuning_test
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=C:\Users\pupul\Desktop\Famam\slurm_logs/%x_%j.out
#SBATCH --error=C:\Users\pupul\Desktop\Famam\slurm_logs/%x_%j.err

#SBATCH --mail-user=murilodefreitasspinelli@gmail.com
#SBATCH --mail-type=END,FAIL

# ============================================================
# FAMAM Hyperparameter Tuning Job - tuning_test
# Generated: 2026-01-22 17:53:17
# ============================================================

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules
module load devel/cuda/12.1
module load devel/python/3.11

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate famam_gpu

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Navigate to project directory
cd C:\Users\pupul\Desktop\Famam

# Run tuning
echo "Starting hyperparameter tuning..."
python -m src.user_interface.experiment_cli.run_tuning \
    --tuning-config C:\Users\pupul\Desktop\Famam\configs\model_tuning\quick_tuning.json \
    --training-config C:\Users\pupul\Desktop\Famam\configs\model_training\small_model.json \
    --dataset C:\Users\pupul\Desktop\Famam\data\datasets\default_dataset.h5 \
    --max-samples 1000

echo "Job completed at $(date)"
