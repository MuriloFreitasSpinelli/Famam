"""
SLURM job configuration for distributed training on bwUniCluster 3.0.

Provides dataclass for configuring SLURM batch job parameters including
partition selection, resource allocation, and environment setup.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import ClassVar, Optional, List


@dataclass
class SlurmConfig:
    """
    Configuration for SLURM batch job submission.

    Supports bwUniCluster 3.0 GPU partitions:
    - gpu_h100: NVIDIA H100 GPUs (high performance)
    - gpu_a100_il: NVIDIA A100 GPUs (general purpose)
    - gpu_a100_short: NVIDIA A100 GPUs (short jobs, max 30 min)
    """

    VALID_PARTITIONS: ClassVar[set[str]] = {
        'gpu_h100', 'gpu_a100_il', 'gpu_a100_short'
    }

    # ============ Job Identification ============
    job_name: str = 'famam_training'

    # ============ Resource Allocation ============
    partition: str = 'gpu_a100_il'
    nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    memory_gb: int = 64
    time_limit: str = '24:00:00'  # HH:MM:SS format

    # ============ Environment Setup ============
    conda_env: Optional[str] = None
    module_loads: List[str] = field(default_factory=lambda: ['devel/cuda/12.1', 'devel/python/3.11'])

    # ============ Multi-Node Settings ============
    ntasks_per_node: int = 1  # For multi-node: one task per node

    # ============ Output Settings ============
    output_dir: str = './slurm_jobs'
    log_dir: str = './slurm_logs'

    # ============ Advanced Options ============
    exclusive: bool = False  # Request exclusive node access
    email: Optional[str] = None  # Email for job notifications
    email_type: str = 'END,FAIL'  # When to send emails: BEGIN,END,FAIL,ALL

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate partition
        if self.partition not in self.VALID_PARTITIONS:
            raise ValueError(
                f"Invalid partition '{self.partition}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_PARTITIONS))}"
            )

        # Validate nodes
        if self.nodes < 1:
            raise ValueError(f"nodes must be at least 1, got {self.nodes}")

        # Validate GPUs per node
        if self.gpus_per_node < 1:
            raise ValueError(f"gpus_per_node must be at least 1, got {self.gpus_per_node}")

        # Validate CPUs per task
        if self.cpus_per_task < 1:
            raise ValueError(f"cpus_per_task must be at least 1, got {self.cpus_per_task}")

        # Validate memory
        if self.memory_gb < 1:
            raise ValueError(f"memory_gb must be at least 1, got {self.memory_gb}")

        # Validate time limit format (HH:MM:SS)
        parts = self.time_limit.split(':')
        if len(parts) != 3:
            raise ValueError(
                f"Invalid time_limit format '{self.time_limit}'. "
                "Must be HH:MM:SS format"
            )
        try:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            if hours < 0 or minutes < 0 or minutes > 59 or seconds < 0 or seconds > 59:
                raise ValueError("Invalid time values")
        except ValueError:
            raise ValueError(
                f"Invalid time_limit format '{self.time_limit}'. "
                "Must be HH:MM:SS format with valid numbers"
            )

        # Check partition time limits
        if self.partition == 'gpu_a100_short':
            total_minutes = hours * 60 + minutes + seconds / 60
            if total_minutes > 30:
                raise ValueError(
                    f"gpu_a100_short partition has a 30-minute time limit. "
                    f"Requested: {self.time_limit}"
                )

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs across all nodes."""
        return self.nodes * self.gpus_per_node

    @property
    def is_multi_node(self) -> bool:
        """Whether this is a multi-node job."""
        return self.nodes > 1

    def get_recommended_cpus(self) -> int:
        """Get recommended CPUs based on GPU count."""
        # Typical ratio: 8 CPUs per GPU
        return self.gpus_per_node * 8

    def get_recommended_memory(self) -> int:
        """Get recommended memory (GB) based on GPU count."""
        # Typical ratio: 32GB per GPU
        return self.gpus_per_node * 32

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"SLURM configuration saved to: {output_path}")

    @classmethod
    def load(cls, path: str) -> 'SlurmConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        summary_lines = [
            "=" * 60,
            "SLURM Job Configuration Summary",
            "=" * 60,
            "",
            "Job:",
            f"  Name: {self.job_name}",
            f"  Partition: {self.partition}",
            f"  Time Limit: {self.time_limit}",
            "",
            "Resources:",
            f"  Nodes: {self.nodes}",
            f"  GPUs per Node: {self.gpus_per_node}",
            f"  Total GPUs: {self.total_gpus}",
            f"  CPUs per Task: {self.cpus_per_task}",
            f"  Memory: {self.memory_gb} GB",
            "",
            "Environment:",
            f"  Modules: {', '.join(self.module_loads)}",
            f"  Conda Env: {self.conda_env or 'None'}",
            "",
            "Output:",
            f"  Scripts: {self.output_dir}",
            f"  Logs: {self.log_dir}",
            "=" * 60,
        ]

        return "\n".join(summary_lines)


# Preset configurations for common use cases
def get_single_gpu_config(job_name: str = 'famam_train') -> SlurmConfig:
    """Get configuration for single GPU training."""
    return SlurmConfig(
        job_name=job_name,
        partition='gpu_a100_il',
        nodes=1,
        gpus_per_node=1,
        cpus_per_task=8,
        memory_gb=32,
        time_limit='24:00:00',
    )


def get_multi_gpu_config(
    job_name: str = 'famam_train_multi',
    gpus: int = 4,
) -> SlurmConfig:
    """Get configuration for multi-GPU (single node) training."""
    return SlurmConfig(
        job_name=job_name,
        partition='gpu_a100_il',
        nodes=1,
        gpus_per_node=gpus,
        cpus_per_task=gpus * 8,
        memory_gb=gpus * 32,
        time_limit='24:00:00',
    )


def get_multi_node_config(
    job_name: str = 'famam_train_distributed',
    nodes: int = 2,
    gpus_per_node: int = 4,
) -> SlurmConfig:
    """Get configuration for multi-node distributed training."""
    return SlurmConfig(
        job_name=job_name,
        partition='gpu_a100_il',
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_task=gpus_per_node * 8,
        memory_gb=gpus_per_node * 32,
        time_limit='24:00:00',
        ntasks_per_node=1,
    )


def get_dev_config(job_name: str = 'famam_dev') -> SlurmConfig:
    """Get configuration for quick development/testing on short partition."""
    return SlurmConfig(
        job_name=job_name,
        partition='gpu_a100_short',
        nodes=1,
        gpus_per_node=1,
        cpus_per_task=8,
        memory_gb=32,
        time_limit='00:30:00',
    )
