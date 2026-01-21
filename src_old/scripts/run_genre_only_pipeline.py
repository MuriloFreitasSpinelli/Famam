"""
Genre-Only Training Pipeline

Creates a dataset with only music and genre conditioning (no artist or instruments),
generates TensorFlow datasets, and runs the full training pipeline.

Usage:
    python src/scripts/run_genre_only_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
src_root = script_dir.parent  # src directory
project_root = src_root.parent  # Famam root
sys.path.insert(0, str(project_root))

from src_old.data import EnhancedDatasetConfig
from src_old.data.configs.tensorflow_dataset_config import TensorflowDatasetConfig
from src_old.data.scripts.create_enchanced_dataset import create_and_save_dataset
from src_old.data.scripts.generate_tensorflow_dataset import generate_tensorflow_dataset


def main():
    print("=" * 70)
    print("GENRE-ONLY TRAINING PIPELINE")
    print("=" * 70)

    # Configuration
    dataset_name = "genre_only_dataset"
    experiment_name = "genre_only_experiment"

    # Paths
    midi_dir = project_root / "data" / "core_datasets" / "clean_midi"
    genre_tsv = midi_dir / "genre.tsv"
    output_path = project_root / "data" / "datasets" / f"{dataset_name}.h5"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Create Enhanced Dataset (Genre only, no artist)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 1/3] Creating Enhanced Music Dataset")
    print("=" * 70)
    print(f"  - Input directory: {midi_dir}")
    print(f"  - Genre TSV: {genre_tsv}")
    print(f"  - Output path: {output_path}")
    print(f"  - Extract genre: Yes")
    print(f"  - Extract artist: No (genre-only conditioning)")

    dataset_config = EnhancedDatasetConfig(
        dataset_name=dataset_name,
        input_dirs=[str(midi_dir)],
        output_path=str(output_path),
        genre_tsv_path=str(genre_tsv),
        # Genre conditioning only - no artist
        extract_genre_from_path=True,
        extract_artist_from_path=False,  # No artist conditioning
        # Filtering options
        min_tracks=1,
        max_tracks=32,
        min_notes=50,
        min_duration=10.0,  # At least 10 seconds
        max_duration=600.0,  # Max 10 minutes
        # Preprocessing
        resolution=24,
        quantize=True,
        remove_empty_tracks=True,
        # Processing
        max_samples=150,  # Limit to 150 samples
        random_seed=42,
        verbose=True
    )

    # Create and save the dataset (or load if exists)
    from src_old.data import EnhancedMusicDataset
    if output_path.exists():
        print(f"Loading existing dataset from {output_path}...")
        dataset = EnhancedMusicDataset.load(str(output_path))
    else:
        dataset = create_and_save_dataset(dataset_config)
    print(f"\nDataset ready with {len(dataset)} samples")
    print(f"  Genres: {dataset.vocabulary.num_genres}")
    print(f"  Artists: {dataset.vocabulary.num_artists} (should be 0 since not extracted)")

    # =========================================================================
    # Step 2: Generate TensorFlow Dataset (music-genre type)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 2/3] Generating TensorFlow Dataset")
    print("=" * 70)

    tf_config = TensorflowDatasetConfig(
        tensor_name=dataset_name,
        tensor_type='music-genre',  # Only music and genre - no artist or instruments
        representation_type='piano-roll',
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        output_dir='./models'
    )

    print(f"  - Tensor type: {tf_config.tensor_type}")
    print(f"  - Representation: {tf_config.representation_type}")
    print(f"  - Splits: train={tf_config.train_split}, val={tf_config.val_split}, test={tf_config.test_split}")

    # Generate and save tensors (or load if exists)
    from src_old.data.tensorflow_dataset import load_tensors, tensors_exist
    if tensors_exist(dataset_name):
        print(f"Loading existing tensors for {dataset_name}...")
        tf_datasets = load_tensors(dataset_name)
    else:
        tf_datasets = generate_tensorflow_dataset(tf_config, dataset, save=True)

    print("\nTensorFlow datasets generated:")
    for split_name, split_dataset in tf_datasets.items():
        cardinality = split_dataset.cardinality().numpy()
        if cardinality >= 0:
            print(f"  {split_name}: {cardinality} samples")
        else:
            print(f"  {split_name}: created")

    # Save TF config for training pipeline
    tf_config_path = project_root / "data" / "configs" / f"{dataset_name}_tf_config.json"
    tf_config_path.parent.mkdir(parents=True, exist_ok=True)
    tf_config.save(str(tf_config_path))
    print(f"\nTF config saved to: {tf_config_path}")

    # =========================================================================
    # Step 3: Run Full Training Pipeline (using existing infrastructure)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 3/3] Running Full Training Pipeline")
    print("=" * 70)

    # Import training modules (using existing implementations)
    from src_old.training.configs.training_config import TrainingConfig
    from src_old.training.trainer import Trainer
    from src_old.training.hyperparameter_tuning import tune_from_dataset

    # -------------------------------------------------------------------------
    # Step 3a: Hyperparameter Tuning (using existing tune_from_dataset)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Hyperparameter Tuning")
    print("-" * 50)

    tuning_results_path = Path('./models') / 'tuning' / f'{experiment_name}.json'

    if tuning_results_path.exists():
        print(f"Loading existing tuning results from {tuning_results_path}...")
    else:
        tuning_config = TrainingConfig(
            model_name=f"{experiment_name}_tuning",
            epochs=10,  # Fewer epochs for tuning
            output_dir='./models',
            param_grid={
                'lstm_units': [[64], [128], [128, 64]],
                'dense_units': [[32], [64]],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.0001, 0.001, 0.01],
            },
            random_seed=42,
        )

        # Use the existing tune_from_dataset function
        search, best_params = tune_from_dataset(
            train_dataset=tf_datasets['train'],
            config=tuning_config,
            method='random_search',
            n_iter=15,
            cv=3,
            max_samples=2000,
            verbose=2,
        )

        # Save tuning results
        tuning_results_path = tuning_config.save_tuning_results(search, experiment_name)
        print(f"\nTuning results saved to: {tuning_results_path}")

    # -------------------------------------------------------------------------
    # Step 3b: Final Training with Best Parameters
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Training Final Model")
    print("-" * 50)

    # Load config from tuning results
    final_model_name = f"{experiment_name}_final"
    training_config = TrainingConfig.from_tuning_results(
        str(tuning_results_path),
        model_name=final_model_name,
        epochs=100,  # Full training epochs
        use_early_stopping=True,
        use_checkpointing=True,
        use_tensorboard=True,
        output_dir='./models',
    )

    print(training_config.summary())

    # Train using the existing Trainer class
    trainer = Trainer(training_config)
    history = trainer.train(
        tf_datasets['train'],
        tf_datasets['validation'],
        num_genres=dataset.vocabulary.num_genres,
        num_artists=dataset.vocabulary.num_artists,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = trainer.evaluate(tf_datasets['test'])

    # Save with vocabulary and tf_config using existing SavedModel infrastructure
    save_path = Path('./models') / final_model_name / 'model_bundle.h5'
    trainer.save_with_vocabulary(str(save_path), dataset.vocabulary, tf_config)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Dataset: {output_path}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Genres: {dataset.vocabulary.num_genres}")
    print(f"  Model: {save_path}")
    print(f"  Final test loss: {results.get('loss', 'N/A'):.4f}")
    print(f"  Representation: {tf_config.representation_type}")
    print(f"  Conditioning: Genre only (no artist or instruments)")

    return trainer, history, dataset


if __name__ == '__main__':
    main()
