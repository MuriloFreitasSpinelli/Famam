"""
Non-interactive entry point for hyperparameter tuning.

Designed for use in SLURM batch jobs and automated pipelines.

Usage:
    python -m src.user_interface.experiment_cli.run_tuning \
        --tuning-config configs/model_tuning/my_tuning.json \
        --training-config configs/model_training/my_model.json \
        --dataset data/datasets/my_dataset.h5 \
        --max-samples 1000
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
        description='Run hyperparameter tuning for music generation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--tuning-config', '-t',
        type=str,
        required=True,
        help='Path to tuning configuration JSON file',
    )

    parser.add_argument(
        '--training-config', '-c',
        type=str,
        default=None,
        help='Path to base training configuration JSON file (optional)',
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to dataset HDF5 file',
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples to use for tuning (limits memory usage)',
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel jobs (-1 for all CPUs, overrides config)',
    )

    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Name for output tuning results (default: based on timestamp)',
    )

    parser.add_argument(
        '--create-training-config',
        action='store_true',
        default=True,
        help='Create training config from best parameters',
    )

    parser.add_argument(
        '--no-create-training-config',
        action='store_false',
        dest='create_training_config',
        help='Do not create training config from best parameters',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )

    return parser.parse_args()


def main():
    """Main entry point for tuning."""
    import sys
    args = parse_args()

    print("=" * 70)
    print("FAMAM Hyperparameter Tuning")
    print("=" * 70)
    print(f"Tuning config: {args.tuning_config}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 70)
    sys.stdout.flush()  # Ensure output is visible in SLURM logs

    # Validate paths
    tuning_config_path = Path(args.tuning_config)
    dataset_path = Path(args.dataset)

    if not tuning_config_path.exists():
        print(f"Error: Tuning config file not found: {tuning_config_path}")
        sys.exit(1)

    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    if args.training_config:
        training_config_path = Path(args.training_config)
        if not training_config_path.exists():
            print(f"Error: Training config file not found: {training_config_path}")
            sys.exit(1)

    # Import after path setup
    from src.core import MusicDataset
    from src.model_tuning import ModelTuningConfig, tune_from_music_dataset

    # Load tuning configuration
    print("\nLoading tuning configuration...")
    sys.stdout.flush()
    tuning_config = ModelTuningConfig.load(str(tuning_config_path))

    # Override n_jobs if specified
    if args.n_jobs is not None:
        tuning_config.tuning_n_jobs = args.n_jobs

    print(f"  Method: {tuning_config.tuning_method}")
    print(f"  CV folds: {tuning_config.tuning_cv_folds}")
    print(f"  N jobs: {tuning_config.tuning_n_jobs}")
    sys.stdout.flush()

    # Load dataset
    print("\nLoading dataset...")
    sys.stdout.flush()
    dataset = MusicDataset.load(str(dataset_path))
    print(f"  Entries: {len(dataset)}")
    print(f"  Tracks: {dataset.count_tracks()}")
    print(f"  Genres: {list(dataset.vocabulary.genre_to_id.keys())}")
    sys.stdout.flush()

    # Convert to TensorFlow dataset
    print("\nConverting dataset to TensorFlow format...")
    sys.stdout.flush()
    tf_dataset = dataset.to_tensorflow_dataset()

    # Determine input shape
    input_shape = (
        dataset.config.num_pitches if dataset.config else 128,
        dataset.max_time_steps
    )
    print(f"  Input shape: {input_shape}")
    sys.stdout.flush()

    # Run tuning
    print("\n" + "=" * 70)
    print("Starting hyperparameter tuning...")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    try:
        search, best_params = tune_from_music_dataset(
            train_dataset=tf_dataset,
            config=tuning_config,
            num_genres=dataset.vocabulary.num_genres,
            input_shape=input_shape,
            max_samples=args.max_samples,
            num_instruments=dataset.vocabulary.num_instruments,
        )

        print("\n" + "=" * 70)
        print("Tuning complete!")
        print("=" * 70)
        print(f"Best score: {search.best_score_:.4f}")
        print(f"Best parameters: {best_params}")

        # Generate output name
        if args.output_name:
            output_name = args.output_name
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"tuning_{timestamp}"

        # Save results
        results_path = tuning_config.save_tuning_results(search, output_name)
        print(f"\nTuning results saved to: {results_path}")

        # Create training config from best parameters
        if args.create_training_config:
            from src.model_training import ModelTrainingConfig

            model_name = f"{output_name}_tuned"
            print(f"\nCreating training config: {model_name}")

            training_config = ModelTuningConfig.from_tuning_results(
                results_path,
                model_name=model_name,
                num_pitches=input_shape[0],
                max_time_steps=input_shape[1],
            )

            # Save training config
            config_dir = PROJECT_ROOT / "configs" / "model_training"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_save_path = config_dir / f"{model_name}.json"
            training_config.save(str(config_save_path))
            print(f"Training config saved to: {config_save_path}")

        print("\nTuning completed successfully!")
        print("\nSaved artifacts:")
        print(f"  - Tuning results: {results_path}")
        if args.create_training_config:
            print(f"  - Training config: {config_save_path}")

    except Exception as e:
        print(f"\nError during tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
