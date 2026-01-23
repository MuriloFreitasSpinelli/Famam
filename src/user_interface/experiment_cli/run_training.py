"""
Non-interactive entry point for model training.

Designed for use in SLURM batch jobs and automated pipelines.
Supports distributed training via TensorFlow distribution strategies.

Usage:
    python -m src.user_interface.experiment_cli.run_training \
        --config configs/model_training/my_model.json \
        --dataset data/datasets/my_dataset.h5 \
        --strategy mirrored
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a music generation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to training configuration JSON file',
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to dataset HDF5 file',
    )

    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='none',
        choices=['none', 'mirrored', 'multi_worker_mirrored', 'tpu'],
        help='Distribution strategy for training',
    )

    parser.add_argument(
        '--batch-size-per-replica',
        type=int,
        default=None,
        help='Batch size per GPU/replica (overrides config batch_size)',
    )

    parser.add_argument(
        '--splits',
        type=str,
        default='0.8,0.1,0.1',
        help='Train/val/test split ratios (comma-separated)',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )

    parser.add_argument(
        '--save-bundle',
        action='store_true',
        default=True,
        help='Save model bundle after training',
    )

    parser.add_argument(
        '--no-save-bundle',
        action='store_false',
        dest='save_bundle',
        help='Do not save model bundle after training',
    )

    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()

    print("=" * 70)
    print("FAMAM Model Training")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.strategy}")
    print("=" * 70)

    # Validate paths
    config_path = Path(args.config)
    dataset_path = Path(args.dataset)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Import after path setup
    from src.core import MusicDataset
    from src.model_training import ModelTrainingConfig, train_from_music_dataset

    # Load configuration
    print("\nLoading configuration...")
    config = ModelTrainingConfig.load(str(config_path))

    # Override distribution settings from command line
    config.distribution_strategy = args.strategy
    if args.batch_size_per_replica is not None:
        config.batch_size_per_replica = args.batch_size_per_replica

    if args.seed is not None:
        config.random_seed = args.seed

    print(config.summary())

    # Load dataset
    print("\nLoading dataset...")
    dataset = MusicDataset.load(str(dataset_path))
    print(f"  Entries: {len(dataset)}")
    print(f"  Tracks: {dataset.count_tracks()}")
    print(f"  Genres: {list(dataset.vocabulary.genre_to_id.keys())}")

    # Ensure max_time_steps matches the dataset
    if config.max_time_steps != dataset.max_time_steps:
        print(f"\nNote: Updating config max_time_steps from {config.max_time_steps} "
              f"to {dataset.max_time_steps} to match dataset.")
        config.max_time_steps = dataset.max_time_steps

    # Parse splits
    splits = tuple(float(x.strip()) for x in args.splits.split(','))
    if len(splits) != 3 or abs(sum(splits) - 1.0) > 0.001:
        print(f"Error: Invalid splits '{args.splits}'. Must be 3 values summing to 1.0")
        sys.exit(1)

    # Convert to TensorFlow datasets
    print(f"\nSplitting dataset: {splits}")
    datasets = dataset.to_tensorflow_dataset(splits=splits, random_state=args.seed)

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    try:
        model, history, trainer = train_from_music_dataset(
            datasets=datasets,
            config=config,
            vocabulary=dataset.vocabulary,
        )

        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)

        # Print final metrics
        if history:
            final_loss = history.get('loss', [])[-1] if history.get('loss') else 'N/A'
            final_val_loss = history.get('val_loss', [])[-1] if history.get('val_loss') else 'N/A'
            print(f"Final loss: {final_loss}")
            print(f"Final val_loss: {final_val_loss}")

        # Save model bundle
        if args.save_bundle:
            output_dir = Path(config.output_dir) / config.model_name
            bundle_path = output_dir / "model_bundle"
            print(f"\nSaving model bundle to: {bundle_path}.h5")
            trainer.save_bundle(str(bundle_path), dataset.vocabulary)

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
