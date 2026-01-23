"""
SLURM batch script generator for distributed training on bwUniCluster 3.0.

Generates SBATCH scripts for:
- Single GPU training
- Multi-GPU training (MirroredStrategy)
- Multi-node distributed training (MultiWorkerMirroredStrategy)
"""

from pathlib import Path, PurePosixPath
from typing import Optional
from datetime import datetime

from .configs.slurm_config import SlurmConfig


def to_posix_path(path_str: str) -> str:
    """Convert a path string to POSIX format (forward slashes) for Linux compatibility."""
    return str(PurePosixPath(Path(path_str)))


class SlurmScriptGenerator:
    """
    Generator for SLURM batch scripts.

    Creates properly formatted SBATCH scripts for bwUniCluster 3.0
    with support for various distributed training configurations.
    """

    # Template for single-node training (single or multi-GPU)
    SINGLE_NODE_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory_gb}G
#SBATCH --time={time_limit}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{exclusive_line}
{email_lines}

# ============================================================
# FAMAM Training Job - {job_name}
# Generated: {timestamp}
# Strategy: {strategy}
# GPUs: {gpus_per_node}
# ============================================================

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $SLURM_GPUS"

# Load required modules
{module_loads}

# Activate conda environment
{conda_activation}

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Navigate to project directory
# Derive project root from script location (script is in slurm_jobs/)
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# ============================================================
# Copy dataset to local SSD ($TMPDIR) for faster I/O
# ============================================================
DATASET_SRC="{dataset_path}"
DATASET_FILENAME=$(basename "$DATASET_SRC")
DATASET_LOCAL="$TMPDIR/$DATASET_FILENAME"

echo "Copying dataset to local SSD..."
echo "  Source: $DATASET_SRC"
echo "  Destination: $DATASET_LOCAL"
cp "$DATASET_SRC" "$DATASET_LOCAL"
echo "  Copy complete. Size: $(du -h $DATASET_LOCAL | cut -f1)"

# Run training with local dataset copy
echo "Starting training..."
python -m src.user_interface.experiment_cli.run_training \\
    --config {config_path} \\
    --dataset "$DATASET_LOCAL" \\
    --strategy {strategy}

# Cleanup (optional - $TMPDIR is auto-cleaned after job)
echo "Cleaning up local dataset copy..."
rm -f "$DATASET_LOCAL"

echo "Job completed at $(date)"
'''

    # Template for multi-node distributed training
    MULTI_NODE_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory_gb}G
#SBATCH --time={time_limit}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{exclusive_line}
{email_lines}

# ============================================================
# FAMAM Distributed Training Job - {job_name}
# Generated: {timestamp}
# Strategy: multi_worker_mirrored
# Nodes: {nodes}
# GPUs per Node: {gpus_per_node}
# Total GPUs: {total_gpus}
# ============================================================

echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Node ID: $SLURM_NODEID"

# Load required modules
{module_loads}

# Activate conda environment
{conda_activation}

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Navigate to project directory
# Derive project root from script location (script is in slurm_jobs/)
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "Node $SLURM_NODEID: Project directory: $PROJECT_DIR"

# ============================================================
# Copy dataset to local SSD ($TMPDIR) for faster I/O
# Each node copies the dataset to its own local storage
# ============================================================
DATASET_SRC="{dataset_path}"
DATASET_FILENAME=$(basename "$DATASET_SRC")
DATASET_LOCAL="$TMPDIR/$DATASET_FILENAME"

echo "Node $SLURM_NODEID: Copying dataset to local SSD..."
echo "  Source: $DATASET_SRC"
echo "  Destination: $DATASET_LOCAL"
cp "$DATASET_SRC" "$DATASET_LOCAL"
echo "  Copy complete. Size: $(du -h $DATASET_LOCAL | cut -f1)"

# ============================================================
# Setup TF_CONFIG for MultiWorkerMirroredStrategy
# ============================================================

# Get list of all nodes in the job
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES_ARRAY=($NODES)
NUM_NODES=${{#NODES_ARRAY[@]}}

# Build worker list with port
WORKER_PORT=12345
WORKERS=""
for node in ${{NODES_ARRAY[@]}}; do
    if [ -z "$WORKERS" ]; then
        WORKERS="\\"$node:$WORKER_PORT\\""
    else
        WORKERS="$WORKERS,\\"$node:$WORKER_PORT\\""
    fi
done

# Set TF_CONFIG for this worker
export TF_CONFIG='{{"cluster":{{"worker":['$WORKERS']}},"task":{{"type":"worker","index":'$SLURM_NODEID'}}}}'

echo "TF_CONFIG: $TF_CONFIG"

# Run training with srun (using local dataset copy)
echo "Starting distributed training..."
srun python -m src.user_interface.experiment_cli.run_training \\
    --config {config_path} \\
    --dataset "$DATASET_LOCAL" \\
    --strategy multi_worker_mirrored

# Cleanup
echo "Cleaning up local dataset copy..."
rm -f "$DATASET_LOCAL"

echo "Job completed at $(date)"
'''

    # Template for tuning jobs
    TUNING_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory_gb}G
#SBATCH --time={time_limit}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{exclusive_line}
{email_lines}

# ============================================================
# FAMAM Hyperparameter Tuning Job - {job_name}
# Generated: {timestamp}
# ============================================================

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules
{module_loads}

# Activate conda environment
{conda_activation}

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Navigate to project directory
# Derive project root from script location (script is in slurm_jobs/)
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# ============================================================
# Copy dataset to local SSD ($TMPDIR) for faster I/O
# ============================================================
DATASET_SRC="{dataset_path}"
DATASET_FILENAME=$(basename "$DATASET_SRC")
DATASET_LOCAL="$TMPDIR/$DATASET_FILENAME"

echo "Copying dataset to local SSD..."
echo "  Source: $DATASET_SRC"
echo "  Destination: $DATASET_LOCAL"
cp "$DATASET_SRC" "$DATASET_LOCAL"
echo "  Copy complete. Size: $(du -h $DATASET_LOCAL | cut -f1)"

# Run tuning with local dataset copy
echo "Starting hyperparameter tuning..."
python -m src.user_interface.experiment_cli.run_tuning \\
    --tuning-config {tuning_config_path} \\
    --training-config {training_config_path} \\
    --dataset "$DATASET_LOCAL" \\
    --max-samples {max_samples}

# Cleanup
echo "Cleaning up local dataset copy..."
rm -f "$DATASET_LOCAL"

echo "Job completed at $(date)"
'''

    def __init__(self, slurm_config: SlurmConfig):
        """
        Initialize the SLURM script generator.

        Args:
            slurm_config: SLURM job configuration
        """
        self.config = slurm_config

    def _get_module_loads(self) -> str:
        """Generate module load commands."""
        if not self.config.module_loads:
            return "# No modules specified"
        return "\n".join(f"module load {mod}" for mod in self.config.module_loads)

    def _get_conda_activation(self) -> str:
        """Generate conda activation command."""
        if not self.config.conda_env:
            return "# No conda environment specified"
        return f"source $(conda info --base)/etc/profile.d/conda.sh\nconda activate {self.config.conda_env}"

    def _get_exclusive_line(self) -> str:
        """Get exclusive node directive if requested."""
        return "#SBATCH --exclusive" if self.config.exclusive else ""

    def _get_email_lines(self) -> str:
        """Get email notification directives if configured."""
        if not self.config.email:
            return ""
        return f"#SBATCH --mail-user={self.config.email}\n#SBATCH --mail-type={self.config.email_type}"

    def generate_training_script(
        self,
        config_path: str,
        dataset_path: str,
        strategy: str = 'none',
        project_dir: Optional[str] = None,
    ) -> str:
        """
        Generate a SLURM script for training.

        Args:
            config_path: Path to training config JSON
            dataset_path: Path to dataset HDF5 file
            strategy: Distribution strategy ('none', 'mirrored', 'multi_worker_mirrored')
            project_dir: Project root directory (defaults to current working directory)

        Returns:
            Generated SLURM script content
        """
        if project_dir is None:
            project_dir = str(Path.cwd())

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Choose template based on configuration
        if self.config.is_multi_node or strategy == 'multi_worker_mirrored':
            template = self.MULTI_NODE_TEMPLATE
            effective_strategy = 'multi_worker_mirrored'
        else:
            template = self.SINGLE_NODE_TEMPLATE
            effective_strategy = strategy if strategy else 'none'
            if self.config.gpus_per_node > 1 and strategy == 'none':
                effective_strategy = 'mirrored'

        script = template.format(
            job_name=self.config.job_name,
            partition=self.config.partition,
            nodes=self.config.nodes,
            ntasks_per_node=self.config.ntasks_per_node,
            gpus_per_node=self.config.gpus_per_node,
            cpus_per_task=self.config.cpus_per_task,
            memory_gb=self.config.memory_gb,
            time_limit=self.config.time_limit,
            log_dir=to_posix_path(self.config.log_dir),
            total_gpus=self.config.total_gpus,
            exclusive_line=self._get_exclusive_line(),
            email_lines=self._get_email_lines(),
            timestamp=timestamp,
            strategy=effective_strategy,
            module_loads=self._get_module_loads(),
            conda_activation=self._get_conda_activation(),
            project_dir=to_posix_path(project_dir),
            config_path=to_posix_path(config_path),
            dataset_path=to_posix_path(dataset_path),
        )

        return script

    def generate_tuning_script(
        self,
        tuning_config_path: str,
        training_config_path: str,
        dataset_path: str,
        max_samples: int = 1000,
        project_dir: Optional[str] = None,
    ) -> str:
        """
        Generate a SLURM script for hyperparameter tuning.

        Args:
            tuning_config_path: Path to tuning config JSON
            training_config_path: Path to base training config JSON
            dataset_path: Path to dataset HDF5 file
            max_samples: Maximum samples for tuning
            project_dir: Project root directory

        Returns:
            Generated SLURM script content
        """
        if project_dir is None:
            project_dir = str(Path.cwd())

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        script = self.TUNING_TEMPLATE.format(
            job_name=self.config.job_name,
            partition=self.config.partition,
            gpus_per_node=self.config.gpus_per_node,
            cpus_per_task=self.config.cpus_per_task,
            memory_gb=self.config.memory_gb,
            time_limit=self.config.time_limit,
            log_dir=to_posix_path(self.config.log_dir),
            exclusive_line=self._get_exclusive_line(),
            email_lines=self._get_email_lines(),
            timestamp=timestamp,
            module_loads=self._get_module_loads(),
            conda_activation=self._get_conda_activation(),
            project_dir=to_posix_path(project_dir),
            tuning_config_path=to_posix_path(tuning_config_path),
            training_config_path=to_posix_path(training_config_path),
            dataset_path=to_posix_path(dataset_path),
            max_samples=max_samples,
        )

        return script

    def save_training_script(
        self,
        config_path: str,
        dataset_path: str,
        strategy: str = 'none',
        output_name: Optional[str] = None,
        project_dir: Optional[str] = None,
    ) -> Path:
        """
        Generate and save a training SLURM script.

        Args:
            config_path: Path to training config JSON
            dataset_path: Path to dataset HDF5 file
            strategy: Distribution strategy
            output_name: Output script filename (without extension)
            project_dir: Project root directory

        Returns:
            Path to saved script
        """
        script = self.generate_training_script(
            config_path=config_path,
            dataset_path=dataset_path,
            strategy=strategy,
            project_dir=project_dir,
        )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Also ensure log directory exists
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if output_name is None:
            output_name = f"{self.config.job_name}_train"

        output_path = output_dir / f"{output_name}.sh"

        with open(output_path, 'w', newline='\n') as f:
            f.write(script)

        posix_output = to_posix_path(str(output_path))
        print(f"SLURM script saved to: {output_path}")
        print(f"Submit with: sbatch {posix_output}")

        return output_path

    def save_tuning_script(
        self,
        tuning_config_path: str,
        training_config_path: str,
        dataset_path: str,
        max_samples: int = 1000,
        output_name: Optional[str] = None,
        project_dir: Optional[str] = None,
    ) -> Path:
        """
        Generate and save a tuning SLURM script.

        Args:
            tuning_config_path: Path to tuning config JSON
            training_config_path: Path to base training config JSON
            dataset_path: Path to dataset HDF5 file
            max_samples: Maximum samples for tuning
            output_name: Output script filename
            project_dir: Project root directory

        Returns:
            Path to saved script
        """
        script = self.generate_tuning_script(
            tuning_config_path=tuning_config_path,
            training_config_path=training_config_path,
            dataset_path=dataset_path,
            max_samples=max_samples,
            project_dir=project_dir,
        )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if output_name is None:
            output_name = f"{self.config.job_name}_tune"

        output_path = output_dir / f"{output_name}.sh"

        with open(output_path, 'w', newline='\n') as f:
            f.write(script)

        posix_output = to_posix_path(str(output_path))
        print(f"SLURM script saved to: {output_path}")
        print(f"Submit with: sbatch {posix_output}")

        return output_path


def generate_quick_script(
    job_name: str,
    config_path: str,
    dataset_path: str,
    partition: str = 'gpu_a100_il',
    gpus: int = 1,
    time_limit: str = '24:00:00',
    conda_env: Optional[str] = None,
) -> Path:
    """
    Quick function to generate a basic training SLURM script.

    Args:
        job_name: Name for the SLURM job
        config_path: Path to training config
        dataset_path: Path to dataset
        partition: SLURM partition
        gpus: Number of GPUs
        time_limit: Job time limit
        conda_env: Conda environment name

    Returns:
        Path to generated script
    """
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        gpus_per_node=gpus,
        cpus_per_task=gpus * 8,
        memory_gb=gpus * 32,
        time_limit=time_limit,
        conda_env=conda_env,
    )

    generator = SlurmScriptGenerator(slurm_config)

    strategy = 'mirrored' if gpus > 1 else 'none'
    return generator.save_training_script(
        config_path=config_path,
        dataset_path=dataset_path,
        strategy=strategy,
    )
