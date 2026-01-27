#!/usr/bin/env python3
"""
Generate SLURM batch script for transformer training on cluster.

Usage:
    python scripts/generate_slurm_job.py \
        --training-config configs/transformer_training/transform_v1.json \
        --dataset data/datasets/rock_dataset.h5 \
        --slurm-config configs/slurm/transformer_training.json

    # Or with defaults:
    python scripts/generate_slurm_job.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training.slurm_generator import SlurmScriptGenerator
from src.model_training.configs.slurm_config import SlurmConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate SLURM batch script for training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--training-config', '-c',
        type=str,
        default='configs/transformer_training/transform_v1.json',
        help='Path to training configuration JSON',
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='data/datasets/rock_dataset.h5',
        help='Path to dataset HDF5 file',
    )

    parser.add_argument(
        '--slurm-config', '-s',
        type=str,
        default='configs/slurm/transformer_training.json',
        help='Path to SLURM configuration JSON',
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default='none',
        choices=['none', 'mirrored', 'multi_worker_mirrored'],
        help='Distribution strategy',
    )

    parser.add_argument(
        '--output-name', '-o',
        type=str,
        default=None,
        help='Output script name (without .sh extension)',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SLURM Job Generator")
    print("=" * 60)

    # Load SLURM config
    slurm_config_path = Path(args.slurm_config)
    if slurm_config_path.exists():
        print(f"Loading SLURM config: {slurm_config_path}")
        slurm_config = SlurmConfig.load(str(slurm_config_path))
    else:
        print(f"SLURM config not found, using defaults")
        slurm_config = SlurmConfig(
            job_name='famam_transformer',
            partition='gpu_a100_il',
            gpus_per_node=1,
            cpus_per_task=8,
            memory_gb=64,
            time_limit='12:00:00',
            conda_env='famam',
        )

    print(slurm_config.summary())

    # Validate paths
    training_config = Path(args.training_config)
    dataset = Path(args.dataset)

    if not training_config.exists():
        print(f"Warning: Training config not found: {training_config}")
        print("  Make sure to copy it to the cluster before running")

    if not dataset.exists():
        print(f"Warning: Dataset not found: {dataset}")
        print("  Make sure to copy it to the cluster before running")

    # Generate script
    generator = SlurmScriptGenerator(slurm_config)

    # Determine strategy based on GPU count
    strategy = args.strategy
    if strategy == 'none' and slurm_config.gpus_per_node > 1:
        strategy = 'mirrored'
        print(f"\nAuto-selecting strategy: {strategy} (multiple GPUs)")

    output_name = args.output_name or f"{slurm_config.job_name}_train"

    script_path = generator.save_training_script(
        config_path=str(training_config),
        dataset_path=str(dataset),
        strategy=strategy,
        output_name=output_name,
    )

    print("\n" + "=" * 60)
    print("SLURM script generated!")
    print("=" * 60)
    print(f"\nTo submit on cluster:")
    print(f"  sbatch {script_path}")
    print("\nMake sure these files exist on the cluster:")
    print(f"  - {training_config}")
    print(f"  - {dataset}")


if __name__ == '__main__':
    main()
