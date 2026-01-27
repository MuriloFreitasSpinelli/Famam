"""
Script to create a ModelBundle from a trained Keras model checkpoint and dataset.

Usage:
    python scripts/create_model_bundle.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import h5py
import tensorflow as tf

from src.core.model_bundle import ModelBundle
from src.core.vocabulary import Vocabulary
from src.model_training.configs import ModelTrainingConfig
from src.model_training.model_trainer import build_lstm_model


def load_vocabulary_from_dataset(dataset_path: str) -> Vocabulary:
    """Load vocabulary from an HDF5 dataset file."""
    vocab = Vocabulary()

    with h5py.File(dataset_path, 'r') as f:
        # Load vocabulary data
        vocab.genre_to_id = json.loads(f['vocabulary'].attrs['genre_to_id'])
        vocab.artist_to_id = json.loads(f['vocabulary'].attrs['artist_to_id'])

        # Load instrument mappings
        if 'instrument_to_songs' in f['vocabulary'].attrs:
            inst_to_songs = json.loads(f['vocabulary'].attrs['instrument_to_songs'])
            for k, v in inst_to_songs.items():
                vocab.instrument_to_songs[int(k)] = set(v)

        if 'genre_to_instruments' in f['vocabulary'].attrs:
            genre_to_inst = json.loads(f['vocabulary'].attrs['genre_to_instruments'])
            for k, v in genre_to_inst.items():
                vocab.genre_to_instruments[k] = set(v)

    return vocab


def create_bundle(
    weights_path: str,
    dataset_path: str,
    output_path: str,
    model_name: str = "decent_lstm_v1",
):
    """
    Create a ModelBundle from trained weights and dataset.

    Args:
        weights_path: Path to the .keras weights checkpoint (HDF5 format)
        dataset_path: Path to the .h5 dataset file (for vocabulary)
        output_path: Path to save the bundle (without extension)
        model_name: Name for the model bundle
    """
    print(f"Loading vocabulary from: {dataset_path}")
    vocabulary = load_vocabulary_from_dataset(dataset_path)
    print(f"  - Genres: {list(vocabulary.genre_to_id.keys())}")
    print(f"  - Active instruments: {vocabulary.num_active_instruments}")

    # Create training config - these must match what was used during training
    training_config = ModelTrainingConfig(
        model_name=model_name,
        num_pitches=128,
        max_time_steps=512,
        num_instruments=129,
        lstm_units=[128, 64],
        dense_units=[64, 32],
        dropout_rate=0.2,
        recurrent_dropout=0.1,
        bidirectional=False,
        genre_embedding_dim=16,
        instrument_embedding_dim=16,
        batch_size=32,
        epochs=100,
        optimizer='adam',
        learning_rate=0.001,
        loss_function='mse',
    )

    print("Building model architecture...")
    input_shape = (training_config.num_pitches, training_config.max_time_steps)
    model = build_lstm_model(
        input_shape=input_shape,
        num_genres=vocabulary.num_genres,
        config=training_config,
        num_instruments=training_config.num_instruments,
    )

    print(f"Loading weights from: {weights_path}")
    # The checkpoint is in legacy HDF5 format
    # Load weights manually using h5py since Keras 3 has compatibility issues
    with h5py.File(weights_path, 'r') as f:
        weights_group = f['model_weights']

        for layer in model.layers:
            layer_name = layer.name
            if layer_name not in weights_group:
                continue

            layer_group = weights_group[layer_name]
            # Navigate nested structure (layer_name/layer_name/...)
            if layer_name in layer_group:
                layer_group = layer_group[layer_name]

            weight_values = []
            # Get all weights for this layer
            for weight in layer.weights:
                # Extract variable name (e.g., 'kernel:0', 'bias:0', 'embeddings:0')
                var_name = weight.name.split('/')[-1]
                found = False

                # Direct lookup
                if var_name in layer_group:
                    weight_values.append(layer_group[var_name][:])
                    found = True
                else:
                    # Check nested groups (for LSTM cells)
                    for key in layer_group.keys():
                        if isinstance(layer_group[key], h5py.Group):
                            nested = layer_group[key]
                            if var_name in nested:
                                weight_values.append(nested[var_name][:])
                                found = True
                                break
                            # Try without :0 suffix
                            var_base = var_name.replace(':0', '')
                            for nkey in nested.keys():
                                if nkey.startswith(var_base):
                                    weight_values.append(nested[nkey][:])
                                    found = True
                                    break
                            if found:
                                break

                if not found:
                    # Try without :0 suffix at top level
                    var_base = var_name.replace(':0', '')
                    for key in layer_group.keys():
                        if key.startswith(var_base):
                            weight_values.append(layer_group[key][:])
                            break

            if weight_values and len(weight_values) == len(layer.weights):
                layer.set_weights(weight_values)
                print(f"  Loaded weights for: {layer_name}")
            elif layer.weights:
                print(f"  Warning: Could not load all weights for {layer_name} "
                      f"(got {len(weight_values)}, expected {len(layer.weights)})")

    # Compile the model
    model.compile(
        optimizer='adam',
        loss=training_config.loss_function,
        metrics=training_config.metrics,
    )

    print("Creating ModelBundle...")
    bundle = ModelBundle(
        model=model,
        vocabulary=vocabulary,
        training_config=training_config,
        model_name=model_name,
    )

    print(f"Saving bundle to: {output_path}")
    bundle.save(output_path)

    print("\n" + bundle.summary())

    return bundle


if __name__ == "__main__":
    # Paths - relative to project root
    # Use the best checkpoint (lowest loss)
    weights_path = project_root / "models" / "decent_lstm_v1" / "checkpoints" / "model-049-0.0258.keras"
    dataset_path = project_root / "data" / "datasets" / "rock_dataset.h5"
    output_path = project_root / "models" / "decent_lstm_v1" / "bundle" / "decent_lstm_v1"

    print(f"Weights path exists: {weights_path.exists()}")
    print(f"Dataset path exists: {dataset_path.exists()}")

    create_bundle(
        weights_path=str(weights_path),
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        model_name="decent_lstm_v1",
    )
