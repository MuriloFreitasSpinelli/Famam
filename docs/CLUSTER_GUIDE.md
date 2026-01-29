# Cluster Training Guide (bwUniCluster 3.0)

## Quick Start

```bash
# 1. Upload project to cluster
rsync -avz --exclude '.git' --exclude 'data/midi' . username@bwunicluster.scc.kit.edu:~/Famam/

# 2. Upload your dataset separately (it's large)
rsync -avz data/datasets/*.h5 username@bwunicluster.scc.kit.edu:~/Famam/data/datasets/

# 3. SSH to cluster
ssh username@bwunicluster.scc.kit.edu

# 4. Submit job
cd ~/Famam
sbatch scripts/train_cluster.sh
```

## Available GPU Partitions

| Partition | GPU | Max Time | Use Case |
|-----------|-----|----------|----------|
| `gpu_h100` | NVIDIA H100 (80GB) | 72 hours | Best for large models |
| `gpu_a100_il` | NVIDIA A100 (40/80GB) | 48 hours | Good alternative |
| `gpu_a100_short` | NVIDIA A100 | 30 min | Quick tests |
| `dev_gpu_h100` | NVIDIA H100 | 30 min | Development only |

## Job Submission

### Basic submission
```bash
sbatch scripts/train_cluster.sh
```

### With custom dataset/config
```bash
sbatch scripts/train_cluster.sh data/datasets/my_data.h5 configs/model_training/my_config.json
```

### Override resources
```bash
sbatch --time=48:00:00 --gres=gpu:2 scripts/train_cluster.sh
```

## Monitoring Jobs

```bash
# Check your jobs
squeue -u $USER

# Detailed job info
scontrol show job <JOBID>

# Watch output in real-time
tail -f logs/slurm-<JOBID>.out

# Cancel a job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER
```

## Configuration Recommendations

### For H100 (80GB VRAM)
```json
{
  "max_seq_length": 2048,
  "d_model": 512,
  "num_layers": 8,
  "num_heads": 8,
  "d_ff": 2048,
  "batch_size": 32
}
```

### For A100 (40GB VRAM)
```json
{
  "max_seq_length": 2048,
  "d_model": 384,
  "num_layers": 6,
  "num_heads": 6,
  "d_ff": 1536,
  "batch_size": 16
}
```

### For Local (8GB VRAM)
```json
{
  "max_seq_length": 1024,
  "d_model": 256,
  "num_layers": 4,
  "num_heads": 4,
  "d_ff": 1024,
  "batch_size": 4
}
```

## File Transfer

### Upload to cluster
```bash
# Sync code (exclude large files)
rsync -avz --exclude '.git' --exclude 'data/midi' --exclude '*.h5' \
    . username@bwunicluster.scc.kit.edu:~/Famam/

# Upload dataset
rsync -avz data/datasets/*.h5 username@bwunicluster.scc.kit.edu:~/Famam/data/datasets/
```

### Download results
```bash
# Download trained model
rsync -avz username@bwunicluster.scc.kit.edu:~/Famam/models/ ./models/

# Download logs
rsync -avz username@bwunicluster.scc.kit.edu:~/Famam/logs/ ./logs/
```

## Troubleshooting

### Job pending forever
- Check partition limits: `sinfo -p gpu_h100`
- Try a different partition or fewer resources
- Check your quota: `squeue -u $USER`

### Out of memory
- Reduce `batch_size`
- Reduce `max_seq_length`
- Reduce `d_model` or `num_layers`

### Module not found
- Check available modules: `module avail cuda`
- Load correct versions in script

### Job killed (timeout)
- Increase `--time` in SBATCH header
- Enable checkpointing in training config

## Environment Setup (First Time)

```bash
# Load modules
module load devel/cuda/12.1
module load devel/cudnn/8.9

# Create conda environment
conda create -n famam_gpu python=3.10
conda activate famam_gpu

# Install dependencies
pip install tensorflow[and-cuda] muspy h5py numpy

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
