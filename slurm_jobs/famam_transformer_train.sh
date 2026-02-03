#!/bin/bash
#SBATCH --job-name=famam_transformer
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

#SBATCH --mail-user=murilodefreitasspinelli@gmail.com
#SBATCH --mail-type=END,FAIL

# ============================================================
# FAMAM Training Job - famam_transformer
# Generated: 2026-01-29 11:30:59
# Strategy: none
# GPUs: 1
# ============================================================

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $SLURM_GPUS"

# Load required modules
# No modules specified

# ============================================================
# Activate conda environment
# ============================================================
echo "Setting up conda environment..."

# Initialize conda from user installation
CONDA_FOUND=false
for CONDA_DIR in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3"; do
    if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
        echo "  Found conda at: $CONDA_DIR"
        source "$CONDA_DIR/etc/profile.d/conda.sh"
        CONDA_FOUND=true
        break
    fi
done

if [ "$CONDA_FOUND" = false ]; then
    echo "ERROR: Cannot find conda installation"
    exit 1
fi

# Activate the environment
conda activate famam_gpu || { echo "ERROR: Failed to activate famam_gpu"; conda env list; exit 1; }

echo "  Active environment: $CONDA_DEFAULT_ENV"
echo "  Python: $(which python)"

# Verify packages are available
echo "  Checking packages..."
python -c "import numpy; print(f'    numpy {numpy.__version__}')" || exit 1
python -c "import tensorflow as tf; print(f'    tensorflow {tf.__version__}')" || exit 1

echo "Checking GPU..."
nvidia-smi
python -c "import tensorflow as tf; print(f'GPUs available: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered for SLURM logs

# Navigate to project directory
PROJECT_DIR="$HOME/Famam"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# Add project to PYTHONPATH so Python can find src module
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# ============================================================
# Configuration - EDIT THESE FOR YOUR RUN
# ============================================================
DATASET_SRC="data/datasets/multitrack_rock_cluster.h5"
TRAINING_CONFIG="configs/model_training/multitrack_rock_cluster.json"

# ============================================================
# Copy dataset to local SSD ($TMPDIR) for faster I/O
# ============================================================
DATASET_FILENAME=$(basename "$DATASET_SRC")
DATASET_LOCAL="$TMPDIR/$DATASET_FILENAME"

echo "Copying dataset to local SSD..."
echo "  Source: $DATASET_SRC"
echo "  Destination: $DATASET_LOCAL"
cp "$DATASET_SRC" "$DATASET_LOCAL"
echo "  Copy complete. Size: $(du -h $DATASET_LOCAL | cut -f1)"

# Run training with local dataset copy
echo "Starting training..."
echo "  Dataset: $DATASET_LOCAL"
echo "  Config: $TRAINING_CONFIG"

python -m src_v4.client.cli train from-config \
    "$DATASET_LOCAL" \
    "$TRAINING_CONFIG" \
    --val-split 0.1

# Capture exit status
EXIT_STATUS=$?

# Cleanup (optional - $TMPDIR is auto-cleaned after job)
echo "Cleaning up local dataset copy..."
rm -f "$DATASET_LOCAL"

echo "=============================================="
echo "Job completed at $(date)"
echo "Exit status: $EXIT_STATUS"
echo "=============================================="

exit $EXIT_STATUS
