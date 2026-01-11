from typing import Dict
import tensorflow as tf

from data.enhanced_music_dataset import EnhancedMusicDataset
from data.dataset_tensors import save_tensors, load_tensors, tensors_exist
from data.configs.data_tensors_config import DataTensorsConfig


def generate_tensors(
    config: DataTensorsConfig,
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
    tf_datasets = dataset.to_tensorflow_dataset_with_metadata(
        representation=config.representation_type,
        splits=[config.train_split, config.val_split, config.test_split],
        random_state=getattr(config, 'random_state', None)
    )

    if save:
        save_tensors(tf_datasets, config.tensor_name) # type: ignore

    return tf_datasets # type: ignore


def get_tensors(
    config: DataTensorsConfig,
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

    return generate_tensors(config, dataset, save=True)
