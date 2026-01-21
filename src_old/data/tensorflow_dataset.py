# tensorflow_dataset.py
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import h5py
import tensorflow as tf

TENSORS_DIR = Path(__file__).parent.parent.parent / "data" / "tensors"

# Default max time steps (~30 seconds at 120 BPM, resolution 24)
DEFAULT_MAX_TIME_STEPS = 1440


def _pad_or_truncate_last_dim(arr: np.ndarray, target: int) -> np.ndarray:
    """
    Ensure arr has exactly `target` elements on the last dimension:
    - truncate if larger
    - zero-pad if smaller
    """
    if arr.ndim == 0:
        raise ValueError("Expected an array with at least 1 dimension.")

    current = arr.shape[-1]

    # Truncate
    if current > target:
        slicer = (slice(None),) * (arr.ndim - 1) + (slice(0, target),)
        return arr[slicer]

    # Pad
    if current < target:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, target - current)
        return np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)

    return arr


def save_tensors(
    tf_datasets: Dict[str, tf.data.Dataset],
    name: str,
    max_time_steps: int = DEFAULT_MAX_TIME_STEPS
) -> Path:
    """
    Save TensorFlow datasets to HDF5 file.

    Args:
        tf_datasets: Dict with 'train', 'validation', 'test' datasets
        name: Name for the saved file (without extension)
        max_time_steps: Maximum sequence length (truncate/pad to this size)

    Returns:
        Path to saved file
    """
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = TENSORS_DIR / f"{name}.h5"

    print(f"Saving tensors with max_time_steps={max_time_steps}...")

    with h5py.File(filepath, "w") as f:
        # Store max_time_steps as metadata
        f.attrs['max_time_steps'] = max_time_steps

        for split_name, dataset in tf_datasets.items():
            split_group = f.create_group(split_name)

            data_lists: Dict[str, list] = {}
            for sample in dataset:
                for key, value in sample.items():  # type: ignore
                    if key not in data_lists:
                        data_lists[key] = []

                    arr = value.numpy()

                    # Only pad/truncate arrays with dimensions (skip scalars like genre/artist indices)
                    if arr.ndim > 0:
                        arr = _pad_or_truncate_last_dim(arr, max_time_steps)

                    data_lists[key].append(arr)

            for key, values in data_lists.items():
                split_group.create_dataset(
                    key,
                    data=np.stack(values, axis=0),  # stack now that shapes are consistent
                    compression="gzip",
                )

    print(f"Saved tensors to: {filepath}")
    return filepath


def load_tensors(name: str) -> Dict[str, tf.data.Dataset]:
    filepath = TENSORS_DIR / f"{name}.h5"
    if not filepath.exists():
        raise FileNotFoundError(f"Tensors not found: {filepath}")

    tf_datasets = {}
    with h5py.File(filepath, "r") as f:
        for split_name in f.keys():
            split_group = f[split_name]
            data = {key: split_group[key][:] for key in split_group.keys()}  # type: ignore
            tf_datasets[split_name] = tf.data.Dataset.from_tensor_slices(data)

    print(f"Loaded tensors from H5: {filepath}")
    return tf_datasets


def tensors_exist(name: str) -> bool:
    """Check if saved tensors exist."""
    return (TENSORS_DIR / f"{name}.h5").exists()
