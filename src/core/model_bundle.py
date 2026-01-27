"""
ModelBundle: Bundles a trained model with its vocabulary and config.

Supports both LSTM and Transformer models.
Saves/loads the complete model package for inference, ensuring the same
vocabulary mappings used during training are available during generation.
"""

import json
import h5py # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, asdict

import tensorflow as tf # type: ignore

from .vocabulary import Vocabulary
from .event_vocabulary import EventVocabulary

if TYPE_CHECKING:
    from ..model_training.configs import ModelTrainingConfig
    from ..model_training.configs.transformer_config import TransformerTrainingConfig


def _load_legacy_model(model_path):
    """
    Load a model with Keras 2/3 compatibility issues.

    Handles:
    - batch_shape -> batch_input_shape conversion for InputLayer
    - DTypePolicy objects -> simple dtype strings
    """
    import h5py
    import json

    # Read model config
    with h5py.File(model_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError("No model config found in HDF5 file")

        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        config = json.loads(model_config)

    def fix_config(obj):
        """Recursively fix compatibility issues in config."""
        if isinstance(obj, dict):
            # Fix InputLayer: batch_shape -> batch_input_shape (Keras 2 naming)
            if obj.get('class_name') == 'InputLayer' and 'config' in obj:
                layer_config = obj['config']
                if 'batch_shape' in layer_config:
                    layer_config['batch_input_shape'] = layer_config.pop('batch_shape')

            # Fix DTypePolicy objects -> simple dtype string
            if 'dtype' in obj and isinstance(obj['dtype'], dict):
                dtype_config = obj['dtype']
                if dtype_config.get('class_name') == 'DTypePolicy':
                    obj['dtype'] = dtype_config.get('config', {}).get('name', 'float32')

            # Recurse into all values
            for key, value in obj.items():
                fix_config(value)

        elif isinstance(obj, list):
            for item in obj:
                fix_config(item)

        return obj

    config = fix_config(config)

    # Reconstruct model from patched config
    model = tf.keras.models.model_from_json(json.dumps(config))

    # Load weights
    model.load_weights(model_path)

    return model


@dataclass
class ModelMetadata:
    """Metadata about the trained model."""
    model_name: str
    input_shape: tuple  # (num_pitches, max_time_steps)
    num_genres: int
    genre_embedding_dim: int
    num_instruments: int  # Always 129 (0-127 + drums)
    instrument_embedding_dim: int
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
        training_config: "ModelTrainingConfig",
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
            num_instruments=vocabulary.num_instruments,
            instrument_embedding_dim=getattr(training_config, 'instrument_embedding_dim', 16),
            training_config_dict=asdict(training_config),
        )

    def save(self, filepath: str) -> None:
        """
        Save model, vocabulary, and metadata.

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

        # Use .keras suffix for the model weights (most compatible format)
        keras_path = h5_path.with_suffix('.keras') # type: ignore

        # Save Keras model in native .keras format (compatible with Keras 2 and 3)
        self.model.save(keras_path)

        # Save vocabulary and metadata to HDF5
        with h5py.File(h5_path, 'w') as f:
            # Mark model type for factory function
            f.attrs['model_type'] = 'lstm'

            # === Save vocabulary ===
            vocab_group = f.create_group('vocabulary')
            vocab_group.attrs['genre_to_id'] = json.dumps(self.vocabulary.genre_to_id)
            vocab_group.attrs['artist_to_id'] = json.dumps(self.vocabulary.artist_to_id)
            vocab_group.attrs['instrument_to_songs'] = json.dumps(
                {str(k): list(v) for k, v in self.vocabulary.instrument_to_songs.items()}
            )
            vocab_group.attrs['genre_to_instruments'] = json.dumps(
                {k: list(v) for k, v in self.vocabulary.genre_to_instruments.items()}
            )

            # === Save metadata ===
            meta_group = f.create_group('metadata')
            meta_group.attrs['model_name'] = self.metadata.model_name
            meta_group.attrs['input_shape'] = json.dumps(self.metadata.input_shape)
            meta_group.attrs['num_genres'] = self.metadata.num_genres
            meta_group.attrs['genre_embedding_dim'] = self.metadata.genre_embedding_dim
            meta_group.attrs['num_instruments'] = self.metadata.num_instruments
            meta_group.attrs['instrument_embedding_dim'] = self.metadata.instrument_embedding_dim
            meta_group.attrs['training_config'] = json.dumps(self.metadata.training_config_dict)

            # Store keras model filename reference
            f.attrs['keras_model_path'] = keras_path.name

        print(f"Saved model bundle:")
        print(f"  - Metadata: {h5_path}")
        print(f"  - Keras model: {keras_path}")
        print(f"  - Vocabulary: {self.vocabulary.num_genres} genres, {self.vocabulary.num_active_instruments} instruments")

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

            # Load instrument mappings (if present, for backward compatibility)
            if 'instrument_to_songs' in f['vocabulary'].attrs:
                inst_to_songs = json.loads(f['vocabulary'].attrs['instrument_to_songs'])
                for k, v in inst_to_songs.items():
                    vocab.instrument_to_songs[int(k)] = set(v)

            if 'genre_to_instruments' in f['vocabulary'].attrs:
                genre_to_inst = json.loads(f['vocabulary'].attrs['genre_to_instruments'])
                for k, v in genre_to_inst.items():
                    vocab.genre_to_instruments[k] = set(v)

            # === Load metadata ===
            meta = f['metadata']
            model_name = str(meta.attrs['model_name'])
            training_config_dict = json.loads(meta.attrs['training_config'])

            # === Get Keras model path ===
            stored_model_path = str(f.attrs['keras_model_path'])

        # Try multiple model file locations for backwards compatibility
        # Priority: 1) .keras (most compatible), 2) stored path, 3) _model.h5
        model_candidates = [
            h5_path.with_suffix('.keras'),                  # .keras format (most compatible)
            h5_path.parent / stored_model_path,             # Stored path from bundle
            h5_path.parent / (h5_path.stem + '_model.h5'),  # HDF5 format
        ]

        keras_model_path = None
        for candidate in model_candidates:
            if candidate.exists():
                keras_model_path = candidate
                break

        if keras_model_path is None:
            raise FileNotFoundError(
                f"Could not find model file. Tried: {[str(c) for c in model_candidates]}"
            )

        # Reconstruct training config first (needed for recompiling)
        training_config = ModelTrainingConfig(**training_config_dict)

        # Load Keras model without compiling to avoid deserialization issues
        print(f"Loading model from: {keras_model_path}")
        try:
            model = tf.keras.models.load_model(keras_model_path, compile=False)  # type: ignore
        except Exception as e:
            error_str = str(e)
            if 'batch_shape' in error_str or '__keras_tensor__' in error_str:
                # Keras 2/3 version mismatch
                raise RuntimeError(
                    f"Model was saved with a different Keras version than currently installed.\n"
                    f"Current: Keras {tf.keras.__version__}\n"
                    f"The model appears to be saved with Keras 3 format.\n"
                    f"Solutions:\n"
                    f"  1. Upgrade TensorFlow: pip install tensorflow>=2.16\n"
                    f"  2. Re-train the model with your current environment\n"
                    f"Original error: {e}"
                ) from e
            else:
                raise

        # Recompile the model with the original settings
        model.compile(
            optimizer=training_config.optimizer,
            loss=training_config.loss_function,
            metrics=training_config.metrics,
        )

        bundle = cls(
            model=model,
            vocabulary=vocab,
            training_config=training_config,
            model_name=model_name,
        )

        print(f"Loaded model bundle from: {h5_path}")
        print(f"  - Vocabulary: {vocab.num_genres} genres, {vocab.num_active_instruments} instruments")
        print(f"  - Input shape: {bundle.metadata.input_shape}")

        return bundle

    # === Convenience methods ===

    def get_genre_id(self, genre_name: str) -> int:
        """Get genre ID for conditioning."""
        return self.vocabulary.get_genre_id(genre_name)

    def list_genres(self) -> list:
        """List available genres."""
        return list(self.vocabulary.genre_to_id.keys())

    def get_instrument_id(self, instrument_name: str) -> int:
        """Get instrument ID for conditioning."""
        return self.vocabulary.get_instrument_id(instrument_name)

    def list_instruments(self) -> list:
        """List all available instruments (General MIDI)."""
        from .vocabulary import GENERAL_MIDI_INSTRUMENTS
        return list(GENERAL_MIDI_INSTRUMENTS.values())

    def list_active_instruments(self) -> list:
        """List instruments that are actually used in training data."""
        from .vocabulary import GENERAL_MIDI_INSTRUMENTS
        return [
            GENERAL_MIDI_INSTRUMENTS[i]
            for i in range(129)
            if len(self.vocabulary.instrument_to_songs.get(i, set())) > 0
        ]

    def get_instruments_for_genre(self, genre: str) -> list:
        """Get instruments commonly used in a specific genre."""
        from .vocabulary import GENERAL_MIDI_INSTRUMENTS
        instrument_ids = self.vocabulary.get_instruments_for_genre(genre)
        return [GENERAL_MIDI_INSTRUMENTS[i] for i in instrument_ids]

    def get_top_instruments_for_genre(
        self,
        genre: str,
        top_n: int = 3,
        exclude_drums: bool = True
    ) -> list:
        """
        Get top N most frequently used instruments for a genre.

        Args:
            genre: Genre name to get instruments for
            top_n: Number of instruments to return
            exclude_drums: If True, exclude drums from the selection

        Returns:
            List of instrument names sorted by frequency (most frequent first)
        """
        from .vocabulary import GENERAL_MIDI_INSTRUMENTS
        instrument_ids = self.vocabulary.get_top_instruments_for_genre(
            genre, top_n, exclude_drums
        )
        return [GENERAL_MIDI_INSTRUMENTS[i] for i in instrument_ids]

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
        instrument: str,
        genre: str,
        drum_pianoroll: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate prediction with genre and instrument conditioning.

        Args:
            pianoroll: Input pianoroll array, shape (128, time_steps)
            instrument: Instrument name for conditioning
            genre: Genre name for conditioning
            drum_pianoroll: Optional drum track for alignment (128, time_steps)

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

        # Get Instrument ID
        instrument_id = self.get_instrument_id(instrument)
        if instrument_id == -1:
            available = ', '.join(self.list_instruments()[:5])
            raise ValueError(
                f"Unknown instrument '{instrument}'. Available: {available}..."
            )

        # Build inputs
        inputs = {
            'pianoroll_input': np.expand_dims(pianoroll, 0),
            'genre_input': np.array([[genre_id]], dtype=np.int32),
            'instrument_input': np.array([[instrument_id]], dtype=np.int32),
        }

        # Add drum pianoroll if provided and model expects it
        if drum_pianoroll is not None:
            if drum_pianoroll.shape != expected_shape:
                raise ValueError(
                    f"Expected drum_pianoroll shape {expected_shape}, got {drum_pianoroll.shape}"
                )
            inputs['drum_input'] = np.expand_dims(drum_pianoroll, 0)
        else:
            # Use zeros if no drum track provided
            inputs['drum_input'] = np.zeros((1,) + expected_shape, dtype=np.float32)

        # Run prediction
        prediction = self.model.predict(inputs, verbose=0)

        # Return without batch dimension
        if isinstance(prediction, list):
            return prediction[0][0]
        return prediction[0]

    def summary(self) -> str:
        """Generate a human-readable summary of the model bundle."""
        active_instruments = self.list_active_instruments()
        lines = [
            "=" * 60,
            "Model Bundle Summary (LSTM)",
            "=" * 60,
            f"Model Name: {self.model_name}",
            f"Model Type: lstm",
            f"Input Shape: {self.metadata.input_shape}",
            "",
            "Genre Conditioning:",
            f"  Num Genres: {self.num_genres}",
            f"  Embedding Dim: {self.metadata.genre_embedding_dim}",
            f"  Available: {', '.join(self.list_genres()[:5])}{'...' if self.num_genres > 5 else ''}",
            "",
            "Instrument Conditioning:",
            f"  Num Instruments: {self.metadata.num_instruments}",
            f"  Active Instruments: {len(active_instruments)}",
            f"  Embedding Dim: {self.metadata.instrument_embedding_dim}",
            f"  Examples: {', '.join(active_instruments[:5])}{'...' if len(active_instruments) > 5 else ''}",
            "",
            "LSTM Architecture:",
            f"  LSTM Units: {self.training_config.lstm_units}",
            f"  Dense Units: {self.training_config.dense_units}",
            f"  Bidirectional: {self.training_config.bidirectional}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
# Transformer Model Bundle
# ============================================================================

@dataclass
class TransformerModelMetadata:
    """Metadata about a trained Transformer model."""
    model_name: str
    model_type: str  # Always "transformer"
    vocab_size: int
    max_seq_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    num_genres: int
    num_instruments: int
    training_config_dict: dict


class TransformerModelBundle:
    """
    Bundles a trained Transformer model with its vocabularies and metadata.

    The transformer uses event-based representation for autoregressive generation:
        - Input: [BOS] [GENRE] [INSTRUMENT] [events...]
        - Output: Next token prediction over vocabulary

    Saves to HDF5 format containing:
        - Model weights (saved as separate .keras file)
        - Event vocabulary configuration
        - Genre/instrument vocabulary mappings
        - Model metadata
    """

    def __init__(
        self,
        model: tf.keras.Model,  # type: ignore
        event_vocabulary: EventVocabulary,
        vocabulary: Vocabulary,
        training_config: "TransformerTrainingConfig",
        model_name: str = "music_transformer",
    ):
        """
        Initialize TransformerModelBundle.

        Args:
            model: Trained MusicTransformer model
            event_vocabulary: EventVocabulary with token mappings
            vocabulary: Vocabulary with genre/instrument name mappings
            training_config: Training configuration used
            model_name: Name identifier for the model
        """
        self.model = model
        self.event_vocabulary = event_vocabulary
        self.vocabulary = vocabulary
        self.training_config = training_config
        self.model_name = model_name

        # Build metadata
        self.metadata = TransformerModelMetadata(
            model_name=model_name,
            model_type="transformer",
            vocab_size=event_vocabulary.vocab_size,
            max_seq_length=training_config.max_seq_length,
            num_layers=training_config.num_layers,
            d_model=training_config.d_model,
            num_heads=training_config.num_heads,
            d_ff=training_config.d_ff,
            num_genres=vocabulary.num_genres,
            num_instruments=vocabulary.num_instruments,
            training_config_dict=asdict(training_config),
        )

    def save(self, filepath: str) -> None:
        """
        Save model, vocabularies, and metadata.

        Creates:
            - {filepath}.h5: Vocabularies and metadata
            - {filepath}_model/: SavedModel directory for transformer weights

        Args:
            filepath: Base path to save (without extension)
        """
        filepath = Path(filepath)  # type: ignore
        filepath.parent.mkdir(parents=True, exist_ok=True)  # type: ignore

        # Ensure filepath has .h5 extension
        if filepath.suffix != '.h5':  # type: ignore
            h5_path = filepath.with_suffix('.h5')  # type: ignore
        else:
            h5_path = filepath

        # Use SavedModel format for subclassed models (directory)
        model_dir = h5_path.with_suffix('')  # Remove .h5
        model_dir = Path(str(model_dir) + '_model')  # type: ignore

        # Save Keras model in SavedModel format
        self.model.save(model_dir, save_format='tf')

        # Save vocabularies and metadata to HDF5
        with h5py.File(h5_path, 'w') as f:
            # Mark model type
            f.attrs['model_type'] = 'transformer'
            f.attrs['keras_model_path'] = model_dir.name

            # === Save event vocabulary ===
            event_vocab_group = f.create_group('event_vocabulary')
            event_vocab_group.attrs['num_genres'] = self.event_vocabulary.num_genres
            event_vocab_group.attrs['num_instruments'] = self.event_vocabulary.num_instruments
            event_vocab_group.attrs['vocab_size'] = self.event_vocabulary.vocab_size

            # === Save name vocabulary (genre/instrument mappings) ===
            vocab_group = f.create_group('vocabulary')
            vocab_group.attrs['genre_to_id'] = json.dumps(self.vocabulary.genre_to_id)
            vocab_group.attrs['artist_to_id'] = json.dumps(self.vocabulary.artist_to_id)
            vocab_group.attrs['instrument_to_songs'] = json.dumps(
                {str(k): list(v) for k, v in self.vocabulary.instrument_to_songs.items()}
            )
            vocab_group.attrs['genre_to_instruments'] = json.dumps(
                {k: list(v) for k, v in self.vocabulary.genre_to_instruments.items()}
            )

            # === Save metadata ===
            meta_group = f.create_group('metadata')
            meta_group.attrs['model_name'] = self.metadata.model_name
            meta_group.attrs['model_type'] = self.metadata.model_type
            meta_group.attrs['vocab_size'] = self.metadata.vocab_size
            meta_group.attrs['max_seq_length'] = self.metadata.max_seq_length
            meta_group.attrs['num_layers'] = self.metadata.num_layers
            meta_group.attrs['d_model'] = self.metadata.d_model
            meta_group.attrs['num_heads'] = self.metadata.num_heads
            meta_group.attrs['d_ff'] = self.metadata.d_ff
            meta_group.attrs['num_genres'] = self.metadata.num_genres
            meta_group.attrs['num_instruments'] = self.metadata.num_instruments
            meta_group.attrs['training_config'] = json.dumps(self.metadata.training_config_dict)

        print(f"Saved Transformer model bundle:")
        print(f"  - Metadata: {h5_path}")
        print(f"  - Model: {model_dir}")
        print(f"  - Vocabulary: {self.vocabulary.num_genres} genres, vocab_size={self.event_vocabulary.vocab_size}")

    @classmethod
    def load(cls, filepath: str) -> 'TransformerModelBundle':
        """
        Load transformer model, vocabularies, and metadata from HDF5 file.

        Args:
            filepath: Path to the .h5 file

        Returns:
            TransformerModelBundle instance
        """
        from ..model_training.configs.transformer_config import TransformerTrainingConfig
        from ..model_training.transformer_model import (
            MusicTransformer,
            TransformerBlock,
            TokenAndPositionEmbedding,
        )

        filepath = Path(filepath)  # type: ignore

        # Ensure we have the .h5 path
        if filepath.suffix != '.h5':  # type: ignore
            h5_path = filepath.with_suffix('.h5')  # type: ignore
        else:
            h5_path = filepath

        with h5py.File(h5_path, 'r') as f:
            # Verify model type
            model_type = f.attrs.get('model_type', 'lstm')
            if isinstance(model_type, bytes):
                model_type = model_type.decode('utf-8')
            if model_type != 'transformer':
                raise ValueError(
                    f"Expected transformer model, but file contains '{model_type}' model. "
                    f"Use ModelBundle.load() for LSTM models."
                )

            # === Load event vocabulary ===
            event_vocab = EventVocabulary(
                num_genres=int(f['event_vocabulary'].attrs['num_genres']),
                num_instruments=int(f['event_vocabulary'].attrs['num_instruments']),
            )

            # === Load name vocabulary ===
            vocab = Vocabulary()
            vocab.genre_to_id = json.loads(f['vocabulary'].attrs['genre_to_id'])
            vocab.artist_to_id = json.loads(f['vocabulary'].attrs['artist_to_id'])

            if 'instrument_to_songs' in f['vocabulary'].attrs:
                inst_to_songs = json.loads(f['vocabulary'].attrs['instrument_to_songs'])
                for k, v in inst_to_songs.items():
                    vocab.instrument_to_songs[int(k)] = set(v)

            if 'genre_to_instruments' in f['vocabulary'].attrs:
                genre_to_inst = json.loads(f['vocabulary'].attrs['genre_to_instruments'])
                for k, v in genre_to_inst.items():
                    vocab.genre_to_instruments[k] = set(v)

            # === Load metadata ===
            meta = f['metadata']
            model_name = str(meta.attrs['model_name'])
            training_config_dict = json.loads(meta.attrs['training_config'])

            # === Get Keras model path ===
            stored_model_path = str(f.attrs['keras_model_path'])

        # Find model - check for SavedModel directory or .keras file
        base_path = h5_path.with_suffix('')
        model_candidates = [
            Path(str(base_path) + '_model'),  # SavedModel directory
            h5_path.parent / stored_model_path,  # Stored path
            h5_path.with_suffix('.keras'),  # Legacy .keras file
        ]

        keras_model_path = None
        for candidate in model_candidates:
            if candidate.exists() or candidate.is_dir():
                keras_model_path = candidate
                break

        if keras_model_path is None:
            raise FileNotFoundError(
                f"Could not find model. Tried: {[str(c) for c in model_candidates]}"
            )

        # Reconstruct training config
        training_config = TransformerTrainingConfig(**training_config_dict)

        # Load Keras model with custom objects
        print(f"Loading Transformer model from: {keras_model_path}")
        custom_objects = {
            'MusicTransformer': MusicTransformer,
            'TransformerBlock': TransformerBlock,
            'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
        }
        model = tf.keras.models.load_model(
            keras_model_path,
            custom_objects=custom_objects,
            compile=False,
        )

        bundle = cls(
            model=model,
            event_vocabulary=event_vocab,
            vocabulary=vocab,
            training_config=training_config,
            model_name=model_name,
        )

        print(f"Loaded Transformer model bundle from: {h5_path}")
        print(f"  - Vocabulary: {vocab.num_genres} genres, vocab_size={event_vocab.vocab_size}")
        print(f"  - Architecture: {training_config.num_layers} layers, d_model={training_config.d_model}")

        return bundle

    # === Convenience methods ===

    def get_genre_id(self, genre_name: str) -> int:
        """Get genre ID for conditioning."""
        return self.vocabulary.get_genre_id(genre_name)

    def list_genres(self) -> list:
        """List available genres."""
        return list(self.vocabulary.genre_to_id.keys())

    def get_instrument_id(self, instrument_name: str) -> int:
        """Get instrument ID for conditioning."""
        return self.vocabulary.get_instrument_id(instrument_name)

    def list_instruments(self) -> list:
        """List all available instruments (General MIDI)."""
        from .vocabulary import GENERAL_MIDI_INSTRUMENTS
        return list(GENERAL_MIDI_INSTRUMENTS.values())

    def create_start_sequence(self, genre: str, instrument: str) -> np.ndarray:
        """
        Create starting sequence for autoregressive generation.

        Args:
            genre: Genre name
            instrument: Instrument name

        Returns:
            Array of tokens: [BOS, GENRE, INSTRUMENT]
        """
        genre_id = self.get_genre_id(genre)
        if genre_id == -1:
            raise ValueError(f"Unknown genre '{genre}'. Available: {self.list_genres()[:5]}")

        instrument_id = self.get_instrument_id(instrument)
        if instrument_id == -1:
            raise ValueError(f"Unknown instrument '{instrument}'")

        return self.event_vocabulary.create_start_sequence(genre_id, instrument_id)

    @property
    def vocab_size(self) -> int:
        """Get total vocabulary size."""
        return self.event_vocabulary.vocab_size

    @property
    def max_seq_length(self) -> int:
        """Get maximum sequence length."""
        return self.training_config.max_seq_length

    @property
    def num_genres(self) -> int:
        """Get number of genres."""
        return self.vocabulary.num_genres

    def summary(self) -> str:
        """Generate a human-readable summary of the transformer bundle."""
        lines = [
            "=" * 60,
            "Model Bundle Summary (Transformer)",
            "=" * 60,
            f"Model Name: {self.model_name}",
            f"Model Type: transformer",
            "",
            "Vocabulary:",
            f"  Total Vocab Size: {self.vocab_size}",
            f"  Max Sequence Length: {self.max_seq_length}",
            "",
            "Genre Conditioning:",
            f"  Num Genres: {self.num_genres}",
            f"  Available: {', '.join(self.list_genres()[:5])}{'...' if self.num_genres > 5 else ''}",
            "",
            "Transformer Architecture:",
            f"  Layers: {self.training_config.num_layers}",
            f"  Model Dimension: {self.training_config.d_model}",
            f"  Attention Heads: {self.training_config.num_heads}",
            f"  Feed-Forward Dim: {self.training_config.d_ff}",
            f"  Dropout: {self.training_config.dropout_rate}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
# Factory Function for Loading Any Model Type
# ============================================================================

def load_model_bundle(filepath: str) -> Union[ModelBundle, TransformerModelBundle]:
    """
    Load a model bundle, automatically detecting the model type.

    This is the recommended way to load model bundles when you don't know
    whether the model is LSTM or Transformer.

    Args:
        filepath: Path to the .h5 metadata file

    Returns:
        ModelBundle for LSTM models, TransformerModelBundle for Transformer models

    Example:
        bundle = load_model_bundle("models/my_model.h5")
        if isinstance(bundle, TransformerModelBundle):
            # Use autoregressive generation
            ...
        else:
            # Use LSTM prediction
            ...
    """
    filepath = Path(filepath)  # type: ignore

    if filepath.suffix != '.h5':  # type: ignore
        h5_path = filepath.with_suffix('.h5')  # type: ignore
    else:
        h5_path = filepath

    # Check model type in the file
    with h5py.File(h5_path, 'r') as f:
        model_type = f.attrs.get('model_type', 'lstm')
        if isinstance(model_type, bytes):
            model_type = model_type.decode('utf-8')

    if model_type == 'transformer':
        return TransformerModelBundle.load(str(filepath))
    else:
        return ModelBundle.load(str(filepath))
