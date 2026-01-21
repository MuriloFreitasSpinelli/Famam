"""
ModelBundle: Bundles a trained LSTM model with its vocabulary and config.

Saves/loads the complete model package for inference, ensuring the same
vocabulary mappings used during training are available during generation.
"""

import json
import h5py # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import tensorflow as tf # type: ignore

from .vocabulary import Vocabulary
from ..model_training.configs import ModelTrainingConfig


@dataclass
class ModelMetadata:
    """Metadata about the trained model."""
    model_name: str
    input_shape: tuple  # (num_pitches, max_time_steps)
    num_genres: int
    genre_embedding_dim: int
    training_config_dict: dict  # Serialized ModelTrainingConfig


class ModelBundle:
    """
    Bundles a trained Keras LSTM model with its vocabulary and metadata.

    The model uses genre conditioning:
        - Input: pianoroll (128, max_time_steps) + genre_id (scalar)
        - Output: predicted pianoroll

    Saves to HDF5 format containing:
        - Model weights (saved as separate .keras file)
        - Vocabulary (genre_to_id mappings)
        - Model metadata (shapes, config)
    """

    def __init__(
        self,
        model: tf.keras.Model,  # type: ignore
        vocabulary: Vocabulary,
        training_config: ModelTrainingConfig,
        model_name: str = "music_generator",
    ):
        """
        Initialize ModelBundle.

        Args:
            model: Trained Keras model
            vocabulary: Vocabulary with genre mappings
            training_config: Training configuration used
            model_name: Name identifier for the model
        """
        self.model = model
        self.vocabulary = vocabulary
        self.training_config = training_config
        self.model_name = model_name

        # Build metadata
        self.metadata = ModelMetadata(
            model_name=model_name,
            input_shape=(training_config.num_pitches, training_config.max_time_steps),
            num_genres=vocabulary.num_genres,
            genre_embedding_dim=training_config.genre_embedding_dim,
            training_config_dict=asdict(training_config),
        )

    def save(self, filepath: str) -> None:
        """
        Save model, vocabulary, and metadata to HDF5 file.

        Creates two files:
            - {filepath}.h5: Vocabulary and metadata
            - {filepath}.keras: Keras model weights

        Args:
            filepath: Base path to save (without extension)
        """
        filepath = Path(filepath) # type: ignore
        filepath.parent.mkdir(parents=True, exist_ok=True) # type: ignore

        # Ensure filepath has .h5 extension
        if filepath.suffix != '.h5': # type: ignore
            h5_path = filepath.with_suffix('.h5') # type: ignore
        else:
            h5_path = filepath

        keras_path = h5_path.with_suffix('.keras') # type: ignore

        # Save Keras model
        self.model.save(keras_path)

        # Save vocabulary and metadata to HDF5
        with h5py.File(h5_path, 'w') as f:
            # === Save vocabulary ===
            vocab_group = f.create_group('vocabulary')
            vocab_group.attrs['genre_to_id'] = json.dumps(self.vocabulary.genre_to_id)
            vocab_group.attrs['artist_to_id'] = json.dumps(self.vocabulary.artist_to_id)

            # === Save metadata ===
            meta_group = f.create_group('metadata')
            meta_group.attrs['model_name'] = self.metadata.model_name
            meta_group.attrs['input_shape'] = json.dumps(self.metadata.input_shape)
            meta_group.attrs['num_genres'] = self.metadata.num_genres
            meta_group.attrs['genre_embedding_dim'] = self.metadata.genre_embedding_dim
            meta_group.attrs['training_config'] = json.dumps(self.metadata.training_config_dict)

            # Store keras model filename reference
            f.attrs['keras_model_path'] = keras_path.name

        print(f"Saved model bundle:")
        print(f"  - Metadata: {h5_path}")
        print(f"  - Keras model: {keras_path}")
        print(f"  - Vocabulary: {self.vocabulary.num_genres} genres")

    @classmethod
    def load(cls, filepath: str) -> 'ModelBundle':
        """
        Load model, vocabulary, and metadata from HDF5 file.

        Args:
            filepath: Path to the .h5 file

        Returns:
            ModelBundle instance
        """
        filepath = Path(filepath) # type: ignore

        # Ensure we have the .h5 path
        if filepath.suffix != '.h5': # type: ignore
            h5_path = filepath.with_suffix('.h5') # type: ignore
        else:
            h5_path = filepath

        with h5py.File(h5_path, 'r') as f:
            # === Load vocabulary ===
            vocab = Vocabulary()
            vocab.genre_to_id = json.loads(f['vocabulary'].attrs['genre_to_id'])
            vocab.artist_to_id = json.loads(f['vocabulary'].attrs['artist_to_id'])

            # === Load metadata ===
            meta = f['metadata']
            model_name = str(meta.attrs['model_name'])
            training_config_dict = json.loads(meta.attrs['training_config'])

            # === Get Keras model path ===
            keras_model_path = h5_path.parent / f.attrs['keras_model_path'] # type: ignore

        # Load Keras model
        model = tf.keras.models.load_model(keras_model_path)  # type: ignore

        # Reconstruct training config
        training_config = ModelTrainingConfig(**training_config_dict)

        bundle = cls(
            model=model,
            vocabulary=vocab,
            training_config=training_config,
            model_name=model_name,
        )

        print(f"Loaded model bundle from: {h5_path}")
        print(f"  - Vocabulary: {vocab.num_genres} genres")
        print(f"  - Input shape: {bundle.metadata.input_shape}")

        return bundle

    # === Convenience methods ===

    def get_genre_id(self, genre_name: str) -> int:
        """Get genre ID for conditioning."""
        return self.vocabulary.get_genre_id(genre_name)

    def list_genres(self) -> list:
        """List available genres."""
        return list(self.vocabulary.genre_to_id.keys())

    @property
    def input_shape(self) -> tuple:
        """Get expected pianoroll input shape."""
        return self.metadata.input_shape

    @property
    def num_genres(self) -> int:
        """Get number of genres in vocabulary."""
        return self.vocabulary.num_genres

    def predict(
        self,
        pianoroll: np.ndarray,
        genre: str,
    ) -> np.ndarray:
        """
        Generate prediction with genre conditioning.

        Args:
            pianoroll: Input pianoroll array, shape (128, time_steps)
            genre: Genre name for conditioning

        Returns:
            Model prediction (predicted pianoroll)
        """
        # Validate input shape
        expected_shape = self.metadata.input_shape
        if pianoroll.shape != expected_shape:
            raise ValueError(
                f"Expected pianoroll shape {expected_shape}, got {pianoroll.shape}"
            )

        # Get genre ID
        genre_id = self.get_genre_id(genre)
        if genre_id == -1:
            available = ', '.join(self.list_genres()[:5])
            raise ValueError(
                f"Unknown genre '{genre}'. Available: {available}..."
            )

        # Build inputs
        inputs = {
            'pianoroll_input': np.expand_dims(pianoroll, 0),
            'genre_input': np.array([[genre_id]], dtype=np.int32),
        }

        # Run prediction
        prediction = self.model.predict(inputs, verbose=0)

        # Return without batch dimension
        if isinstance(prediction, list):
            return prediction[0][0]
        return prediction[0]

    def summary(self) -> str:
        """Generate a human-readable summary of the model bundle."""
        lines = [
            "=" * 60,
            "Model Bundle Summary",
            "=" * 60,
            f"Model Name: {self.model_name}",
            f"Input Shape: {self.metadata.input_shape}",
            "",
            "Genre Conditioning:",
            f"  Num Genres: {self.num_genres}",
            f"  Embedding Dim: {self.metadata.genre_embedding_dim}",
            f"  Available: {', '.join(self.list_genres()[:5])}{'...' if self.num_genres > 5 else ''}",
            "",
            "LSTM Architecture:",
            f"  LSTM Units: {self.training_config.lstm_units}",
            f"  Dense Units: {self.training_config.dense_units}",
            f"  Bidirectional: {self.training_config.bidirectional}",
            "=" * 60,
        ]
        return "\n".join(lines)
