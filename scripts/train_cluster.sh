#!/bin/bash
#=============================================================================
# SLURM Job Script for Music Generation Model Training
# Cluster: bwUniCluster 3.0
# Documentation: https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs
#=============================================================================
#
# USAGE:
#   sbatch train_cluster.sh                    # Use defaults
#   sbatch train_cluster.sh my_dataset.h5 my_config.json   # Custom configs
#
# MONITORING:
#   squeue -u $USER                            # Check job status
#   scontrol show job <JOBID>                  # Job details
#   scancel <JOBID>                            # Cancel job
#   tail -f slurm-<JOBID>.out                  # Watch output
#
#=============================================================================

#-----------------------------------------------------------------------------
# SLURM RESOURCE CONFIGURATION
#-----------------------------------------------------------------------------

# Job identification
#SBATCH --job-name=music-gen-train      # Job name (shows in squeue)
#SBATCH --output=logs/slurm-%j.out      # Standard output (%j = job ID)
#SBATCH --error=logs/slurm-%j.err       # Standard error

# Partition selection (choose ONE):
#   gpu_h100       - NVIDIA H100 (4x/node), 72h max, best performance
#   gpu_a100_il    - NVIDIA A100 Ice Lake, 48h max
#   gpu_a100_short - Quick jobs, 30min max (for testing)
#   dev_gpu_h100   - Development, 30min max, 1 job limit
#SBATCH --partition=gpu_h100

# Time limit (format: HH:MM:SS or D-HH:MM:SS)
# Be conservative - jobs are killed when time runs out!
#SBATCH --time=24:00:00

# Node and task configuration
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Total tasks (1 for single-GPU training)
#SBATCH --cpus-per-task=8               # CPU cores for data loading

# GPU configuration
#SBATCH --gres=gpu:1                    # GPUs per node (1-4 for H100 nodes)

# Memory (per CPU core)
#SBATCH --mem-per-cpu=8G                # Memory per CPU core

# Email notifications (optional - uncomment and set your email)
# #SBATCH --mail-user=your.email@example.com
# #SBATCH --mail-type=BEGIN,END,FAIL

#-----------------------------------------------------------------------------
# ENVIRONMENT SETUP
#-----------------------------------------------------------------------------

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust versions as needed)
echo "Loading modules..."
module purge                            # Clear any loaded modules
module load devel/cuda/12.1             # CUDA toolkit
module load devel/cudnn/8.9             # cuDNN for deep learning
module load lib/hdf5/1.14               # HDF5 for dataset files

# Activate conda environment
# Option 1: If using conda
source ~/.bashrc
conda activate famam_gpu                # Your conda environment name

# Option 2: If using venv (uncomment if needed)
# source ~/venvs/famam/bin/activate

# Verify GPU is available
echo "Checking GPU..."
nvidia-smi
python -c "import tensorflow as tf; print(f'GPUs available: {len(tf.config.list_physical_devices(\"GPU\"))}')"

#-----------------------------------------------------------------------------
# TRAINING CONFIGURATION
#-----------------------------------------------------------------------------

# Project directory (adjust to your path on the cluster)
PROJECT_DIR="${HOME}/Famam"
cd "$PROJECT_DIR" || exit 1

# Dataset and config files (can be overridden via command line args)
DATASET_FILE="${1:-data/datasets/multitrack_rock.h5}"
TRAINING_CONFIG="${2:-configs/model_training/multitrack_rock.json}"

# Validate files exist
if [ ! -f "$DATASET_FILE" ]; then
    echo "ERROR: Dataset not found: $DATASET_FILE"
    exit 1
fi

if [ ! -f "$TRAINING_CONFIG" ]; then
    echo "ERROR: Training config not found: $TRAINING_CONFIG"
    exit 1
fi

echo ""
echo "Training Configuration:"
echo "  Dataset: $DATASET_FILE"
echo "  Config: $TRAINING_CONFIG"
echo ""

#-----------------------------------------------------------------------------
# RUN TRAINING
#-----------------------------------------------------------------------------

echo "Starting training..."
echo "=============================================="

# Run the training command
python -m src_v4.client.cli train from-config \
    "$DATASET_FILE" \
    "$TRAINING_CONFIG" \
    --val-split 0.1

# Capture exit status
EXIT_STATUS=$?

#-----------------------------------------------------------------------------
# CLEANUP AND REPORTING
#-----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "Job finished at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "=============================================="

# Optional: Copy results to a persistent location
# cp -r models/* /path/to/persistent/storage/

exit $EXIT_STATUS
