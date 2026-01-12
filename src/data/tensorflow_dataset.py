from pathlib import Path
from typing import Dict
import numpy as np
import h5py
import tensorflow as tf

TENSORS_DIR = Path(__file__).parent.parent.parent / "data" / "tensors"


def save_tensors(
    tf_datasets: Dict[str, tf.data.Dataset],
    name: str
) -> Path:
    """
    Save TensorFlow datasets to H5 file.

    Args:
        tf_datasets: Dict with 'train', 'validation', 'test' TF datasets
        name: Name for the saved file (without extension)

    Returns:
        Path to saved file
    """
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = TENSORS_DIR / f"{name}.h5"

    with h5py.File(filepath, 'w') as f:
        for split_name, dataset in tf_datasets.items():
            split_group = f.create_group(split_name)

            # Materialize dataset to lists
            data_lists: Dict[str, list] = {}
            for sample in dataset:
                for key, value in sample.items(): # type: ignore
                    if key not in data_lists:
                        data_lists[key] = []
                    data_lists[key].append(value.numpy())

            # Save each key as a dataset
            for key, values in data_lists.items():
                split_group.create_dataset(key, data=np.array(values), compression='gzip')

    print(f"Saved tensors to: {filepath}")
    return filepath


def load_tensors(name: str) -> Dict[str, tf.data.Dataset]:
    """
    Load TensorFlow datasets from H5 file.

    Args:
        name: Name of the saved file (without extension)

    Returns:
        Dict with 'train', 'validation', 'test' TF datasets
    """
    filepath = TENSORS_DIR / f"{name}.h5"

    if not filepath.exists():
        raise FileNotFoundError(f"Tensors not found: {filepath}")

    tf_datasets = {}

    with h5py.File(filepath, 'r') as f:
        for split_name in f.keys():
            split_group = f[split_name]
            data = {key: split_group[key][:] for key in split_group.keys()} # type: ignore
            tf_datasets[split_name] = tf.data.Dataset.from_tensor_slices(data)

    print(f"Loaded tensors from: {filepath}")
    return tf_datasets


def tensors_exist(name: str) -> bool:
    """Check if saved tensors exist."""
    return (TENSORS_DIR / f"{name}.h5").exists()
