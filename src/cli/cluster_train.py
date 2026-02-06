"""
Cluster Training Script - Non-interactive training for SLURM/cluster jobs.

Usage:
    python -m src.cli.cluster_train <dataset_path> <config_path> [options]

Examples:
    python -m src.cli.cluster_train data/datasets/rock.h5 configs/model_training/rock.json
    python -m src.cli.cluster_train data/datasets/rock.h5 configs/model_training/rock.json --val-split 0.1

Author: Murilo de Freitas Spinelli
"""

import os
import sys
import argparse
from pathlib import Path

# Disable TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a music generation model (non-interactive, for cluster jobs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.cli.cluster_train data/datasets/rock.h5 configs/model_training/rock.json
    python -m src.cli.cluster_train $TMPDIR/rock.h5 configs/model_training/rock.json --val-split 0.1
        """
    )

    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset file (.h5)"
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the training config file (.json)"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )

    parser.add_argument(
        "--min-tracks",
        type=int,
        default=2,
        help="Minimum tracks per sample (default: 2)"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("  FAMAM Cluster Training")
    print("=" * 60)
    print(f"\nDataset: {args.dataset_path}")
    print(f"Config: {args.config_path}")
    print(f"Validation split: {args.val_split}")
    print(f"Min tracks: {args.min_tracks}")
    print(f"Random seed: {args.random_seed}")
    print()

    # Validate paths
    dataset_path = Path(args.dataset_path)
    config_path = Path(args.config_path)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    # Import after argument parsing to avoid slow imports on --help
    print("Loading modules...")
    from ..config import MusicDatasetConfig, TrainingConfig
    from ..data import MusicDataset, MultiTrackEncoder
    from ..training import Trainer
    from ..models import ModelBundle

    try:
        # Load dataset
        print(f"\nLoading dataset: {dataset_path}")
        dataset = MusicDataset.load(str(dataset_path))
        print(f"  Entries: {len(dataset)}")
        print(f"  Tracks: {dataset.count_tracks()}")
        print(f"  Genres: {dataset.vocabulary.num_genres}")
        print(f"  Resolution: {dataset.resolution}")

        # Load training config
        print(f"\nLoading training config: {config_path}")
        config = TrainingConfig.load(str(config_path))

        # Override output dir if specified
        if args.output_dir:
            config.output_dir = args.output_dir
            print(f"  Output dir overridden: {args.output_dir}")

        print(config.summary())

        # Load dataset config for encoder settings
        dataset_config_path = dataset_path.with_suffix('.config.json')
        if dataset_config_path.exists():
            print(f"\nLoading dataset config: {dataset_config_path}")
            dataset_config = MusicDatasetConfig.load(str(dataset_config_path))
            resolution = dataset_config.resolution
            positions_per_bar = dataset_config.positions_per_bar
        else:
            print("\nNo dataset config found, using defaults")
            resolution = dataset.resolution
            positions_per_bar = 16

        # Create encoder
        print("\nCreating encoder...")
        encoder = MultiTrackEncoder(
            num_genres=max(1, dataset.vocabulary.num_genres),
            resolution=resolution,
            positions_per_bar=positions_per_bar,
        )
        print(f"  Encoder: MultiTrackEncoder")
        print(f"  Vocab size: {encoder.vocab_size}")

        # Prepare data
        print("\nPreparing data splits...")
        print(f"  Train: {1.0 - args.val_split:.0%}")
        print(f"  Validation: {args.val_split:.0%}")

        datasets = dataset.to_multitrack_dataset(
            encoder=encoder,
            splits=(1.0 - args.val_split, args.val_split, 0.0),
            random_state=args.random_seed,
            min_tracks=args.min_tracks,
        )

        print(f"  Train samples: {len(datasets['train'])}")
        print(f"  Validation samples: {len(datasets['validation'])}")

        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = Trainer(config, encoder)
        trainer.build_model()

        # Train
        print("\n" + "=" * 60)
        print("  Starting Training")
        print("=" * 60 + "\n")

        model, history = trainer.train(datasets['train'], datasets['validation'])

        # Save model bundle
        print("\n" + "=" * 60)
        print("  Saving Model Bundle")
        print("=" * 60)

        output_dir = Path(config.output_dir) / config.model_name
        bundle_path = output_dir / "checkpoints" / "model_bundle.h5"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = ModelBundle(
            model=trainer.model,
            encoder=encoder,
            config=config,
            model_name=config.model_name,
            vocabulary=dataset.vocabulary,
        )
        bundle.save(str(bundle_path))

        print(f"\nModel bundle saved: {bundle_path}")
        print(bundle.summary())

        # Print final metrics
        print("\n" + "=" * 60)
        print("  Training Complete")
        print("=" * 60)

        if history and hasattr(history, 'history'):
            final_loss = history.history.get('loss', [None])[-1]
            final_val_loss = history.history.get('val_loss', [None])[-1]
            final_acc = history.history.get('accuracy', history.history.get('acc', [None]))[-1]
            final_val_acc = history.history.get('val_accuracy', history.history.get('val_acc', [None]))[-1]

            print(f"\nFinal Metrics:")
            if final_loss is not None:
                print(f"  Loss: {final_loss:.4f}")
            if final_val_loss is not None:
                print(f"  Val Loss: {final_val_loss:.4f}")
            if final_acc is not None:
                print(f"  Accuracy: {final_acc:.4f}")
            if final_val_acc is not None:
                print(f"  Val Accuracy: {final_val_acc:.4f}")

        print(f"\nOutput directory: {output_dir}")
        print("\nDone!")

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
