from typing import Dict
import tensorflow as tf
import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

from data.enhanced_music_dataset import EnhancedMusicDataset
from data.configs.tensorflow_dataset_config import TensorflowDatasetConfig
from data.tensorflow_dataset import load_tensors, tensors_exist, save_tensors


def generate_tensorflow_dataset(
    config: TensorflowDatasetConfig,
    dataset: EnhancedMusicDataset,
    save: bool = True
) -> Dict[str, tf.data.Dataset]:
    """
    Generate TensorFlow datasets with music representation and metadata.

    Args:
        config: Configuration for tensor generation
        dataset: EnhancedMusicDataset to convert
        save: Whether to save tensors to ./data/tensors/

    Returns:
        Dict with 'train', 'validation', 'test' TF datasets
    """
    # Select the appropriate method based on tensor_type
    if config.tensor_type == 'music-only':
        tf_datasets = dataset.to_tensorflow_dataset(
            representation=config.representation_type,
            splits=[config.train_split, config.val_split, config.test_split],
            random_state=getattr(config, 'random_state', None)
        )
    elif config.tensor_type == 'music-genre':
        tf_datasets = dataset.to_tensorflow_dataset_with_genre(
            representation=config.representation_type,
            splits=[config.train_split, config.val_split, config.test_split],
            random_state=getattr(config, 'random_state', None)
        )
    elif config.tensor_type == 'music-instrument':
        tf_datasets = dataset.to_tensorflow_dataset_with_instruments(
            representation=config.representation_type,
            splits=[config.train_split, config.val_split, config.test_split],
            random_state=getattr(config, 'random_state', None)
        )
    elif config.tensor_type == 'full':
        tf_datasets = dataset.to_tensorflow_dataset_with_metadata(
            representation=config.representation_type,
            splits=[config.train_split, config.val_split, config.test_split],
            random_state=getattr(config, 'random_state', None)
        )
    else:
        raise ValueError(
            f"Invalid tensor_type '{config.tensor_type}'. "
            f"Must be one of: {', '.join(sorted(TensorflowDatasetConfig.VALID_TENSOR_TYPES))}"
        )

    if save:
        save_tensors(
            tf_datasets,
            config.tensor_name,
            max_time_steps=config.max_time_steps
        )  # type: ignore

    return tf_datasets  # type: ignore


def get_tensors(
    config: TensorflowDatasetConfig,
    dataset: EnhancedMusicDataset = None # type: ignore
) -> Dict[str, tf.data.Dataset]:
    """
    Get tensors - load from cache if exists, otherwise generate.

    Args:
        config: Configuration for tensor generation
        dataset: EnhancedMusicDataset (required if not cached)

    Returns:
        Dict with 'train', 'validation', 'test' TF datasets
    """
    if tensors_exist(config.tensor_name):
        return load_tensors(config.tensor_name)

    if dataset is None:
        raise ValueError(f"Tensors '{config.tensor_name}' not found and no dataset provided")

    return generate_tensorflow_dataset(config, dataset, save=True)


def main():
    """Command-line interface for generating TensorFlow datasets."""
    parser = argparse.ArgumentParser(
        description='Generate TensorFlow datasets from EnhancedMusicDataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Generate tensors from config and dataset
    python generate_tensorflow_dataset.py config.json dataset.h5
    
    # Generate without saving
    python generate_tensorflow_dataset.py config.json dataset.h5 --no-save
            """
    )
    
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to TensorflowDatasetConfig JSON file'
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to EnhancedMusicDataset H5 file'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save tensors to disk (default: save)'
    )

    parser.add_argument(
        '--max-time-steps',
        type=int,
        default=None,
        help='Maximum sequence length (overrides config value). Use smaller values for less memory, e.g., 512, 1024'
    )

    args = parser.parse_args()
    
    # Validate paths exist
    config_path = Path(args.config_path)
    dataset_path = Path(args.dataset_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load config
    print(f"Loading config from: {config_path}")
    config = TensorflowDatasetConfig.load(str(config_path))

    # Override max_time_steps if provided from CLI
    if args.max_time_steps is not None:
        config.max_time_steps = args.max_time_steps
        print(f"  Max time steps: {config.max_time_steps} (from CLI)")
    else:
        print(f"  Max time steps: {config.max_time_steps}")

    print(f"  Tensor type: {config.tensor_type}")
    print(f"  Representation: {config.representation_type}")
    print(f"  Splits: train={config.train_split}, val={config.val_split}, test={config.test_split}")
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = EnhancedMusicDataset.load(str(dataset_path))
    print(f"  Dataset size: {len(dataset)} items")
    print(f"  Genres: {dataset.vocabulary.num_genres}")
    print(f"  Artists: {dataset.vocabulary.num_artists}")
    
    # Generate tensors
    print(f"\nGenerating TensorFlow datasets...")
    save = not args.no_save
    tf_datasets = generate_tensorflow_dataset(config, dataset, save=save)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"{'='*60}")
    for split_name, split_dataset in tf_datasets.items():
        # Get cardinality (may be -1 for unknown, -2 for infinite)
        cardinality = split_dataset.cardinality().numpy()
        if cardinality >= 0:
            print(f"  {split_name}: {cardinality} samples")
        else:
            print(f"  {split_name}: created")
    
    if save:
        print(f"\nTensors saved as: {config.tensor_name}")
    else:
        print(f"\nTensors generated but not saved (--no-save flag)")


if __name__ == '__main__':
    main()