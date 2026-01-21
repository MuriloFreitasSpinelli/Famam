"""
Train a rock-only genre model.

Creates a dataset of 150 rock samples, generates tensors, and trains the model.
"""
import sys
import subprocess
from pathlib import Path

# Add project paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from data import EnhancedDatasetConfig
from data.scripts.create_enchanced_dataset import create_and_save_dataset
from data.tensorflow_dataset import create_tensorflow_datasets, save_tensors
from data.configs.tensorflow_dataset_config import TensorflowDatasetConfig


def main():
    print("=" * 60)
    print("  Rock Genre Model Training Pipeline")
    print("=" * 60)

    # Paths
    dataset_path = project_root / "data" / "datasets" / "rock_dataset.h5"
    tensor_path = project_root / "data" / "tensors" / "rock_dataset.h5"
    genre_tsv = project_root / "data" / "core_datasets" / "clean_midi" / "genre.tsv"
    midi_dir = project_root / "data" / "core_datasets" / "clean_midi"

    # Step 1: Create rock-only dataset with 150 samples
    print("\n[Step 1/3] Creating rock-only dataset...")
    print("-" * 40)

    dataset_config = EnhancedDatasetConfig(
        dataset_name="rock_dataset",
        input_dirs=[str(midi_dir)],
        output_path=str(dataset_path),
        genre_tsv_path=str(genre_tsv),
        allowed_genres=["Rock"],  # Filter for Rock genre only
        max_samples=150,
        verbose=True
    )

    dataset = create_and_save_dataset(dataset_config)
    print(f"Dataset created with {len(dataset)} samples")

    # Step 2: Generate tensors with genre-only conditioning
    print("\n[Step 2/3] Generating tensors...")
    print("-" * 40)

    tensor_config = TensorflowDatasetConfig(
        dataset_path=str(dataset_path),
        output_path=str(tensor_path),
        use_genre=True,
        use_artist=False,
        use_instruments=False,
        max_time_steps=1024,  # Keep tensors manageable for GPU memory
        verbose=True
    )
    tensor_config.save(str(tensor_path.with_suffix('.config.json')))

    # Create and save tensors
    tf_datasets = create_tensorflow_datasets(
        dataset_path=str(dataset_path),
        config=tensor_config
    )

    saved_path = save_tensors(
        tf_datasets,
        name="rock_dataset",
        max_time_steps=tensor_config.max_time_steps
    )
    print(f"Tensors saved to: {saved_path}")

    print("\n[Step 3/3] Training model...")
    print("-" * 40)
    print("Run the following command to train:")
    print(f"\npython src/training/scripts/train_pipeline.py full --dataset rock_dataset --experiment rock_model --tuning-epochs 5 --epochs 15 --batch-size 16")


if __name__ == "__main__":
    main()
