"""
ModelBundle: Bundles a trained model with its encoder and config for inference.

Author: Murilo de Freitas Spinelli
"""

import json
import h5py
from pathlib import Path
from typing import Optional, Union, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..config import TrainingConfig
from ..data import Vocabulary
from .base_model import BaseMusicModel

if TYPE_CHECKING:
    from ..data.encoders.base_encoder import BaseEncoder


@dataclass
class ModelMetadata:
    """Metadata about a trained model."""
    model_name: str
    model_type: str  # "transformer" or "lstm"
    vocab_size: int
    max_seq_length: int
    d_model: int

    # Architecture-specific
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    d_ff: Optional[int] = None
    lstm_units: Optional[int] = None

    # Encoder info
    encoder_type: str = "event"  # "event" or "remi"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class ModelBundle:
    """
    Bundles a trained model with its encoder and configuration.

    Works with both Transformer and LSTM models that use event-based encoding.
    Saves/loads complete packages for inference.
    """

    def __init__(
        self,
        model: BaseMusicModel,
        encoder: "BaseEncoder",
        config: TrainingConfig,
        model_name: Optional[str] = None,
        vocabulary: Optional[Vocabulary] = None,
    ):
        """
        Initialize ModelBundle.

        Args:
            model: Trained model (TransformerModel or LSTMModel)
            encoder: Encoder instance (EventEncoder or REMIEncoder)
            config: TrainingConfig used for training
            model_name: Name identifier for the model
            vocabulary: Vocabulary from the dataset (genres, instruments, etc.)
        """
        self.model = model
        self.encoder = encoder
        self.config = config
        self.model_name = model_name or config.model_name
        self.vocabulary = vocabulary

        # Build metadata
        self.metadata = ModelMetadata(
            model_name=self.model_name,
            model_type=config.model_type,
            vocab_size=encoder.vocab_size,
            max_seq_length=config.max_seq_length,
            d_model=config.d_model,
            num_layers=config.num_layers if config.model_type == "transformer" else None,
            num_heads=config.num_heads if config.model_type == "transformer" else None,
            d_ff=config.d_ff if config.model_type == "transformer" else None,
            lstm_units=config.lstm_units if config.model_type == "lstm" else None,
            encoder_type=encoder.__class__.__name__.lower().replace("encoder", ""),
            pad_token_id=encoder.pad_token_id,
            bos_token_id=encoder.bos_token_id,
            eos_token_id=encoder.eos_token_id,
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model, encoder, and config.

        Creates:
            - {filepath}.h5: Encoder config and metadata
            - {filepath}_model/: SavedModel directory for weights

        Args:
            filepath: Base path to save (without extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Ensure filepath has .h5 extension
        if filepath.suffix != '.h5':
            h5_path = filepath.with_suffix('.h5')
        else:
            h5_path = filepath

        # Model path - use .keras extension for Keras 3
        model_path = h5_path.with_suffix('')
        model_path = Path(str(model_path) + '_model.keras')

        # Save Keras model (Keras 3 format)
        self.model.save(model_path)

        # Save encoder, config, and metadata to HDF5
        with h5py.File(h5_path, 'w') as f:
            f.attrs['keras_model_path'] = model_path.name

            meta_group = f.create_group('metadata')
            meta_group.attrs['model_name'] = self.metadata.model_name
            meta_group.attrs['model_type'] = self.metadata.model_type
            meta_group.attrs['vocab_size'] = self.metadata.vocab_size
            meta_group.attrs['max_seq_length'] = self.metadata.max_seq_length
            meta_group.attrs['d_model'] = self.metadata.d_model
            meta_group.attrs['encoder_type'] = self.metadata.encoder_type
            meta_group.attrs['pad_token_id'] = self.metadata.pad_token_id
            meta_group.attrs['bos_token_id'] = self.metadata.bos_token_id
            meta_group.attrs['eos_token_id'] = self.metadata.eos_token_id

            if self.metadata.num_layers is not None:
                meta_group.attrs['num_layers'] = self.metadata.num_layers
            if self.metadata.num_heads is not None:
                meta_group.attrs['num_heads'] = self.metadata.num_heads
            if self.metadata.d_ff is not None:
                meta_group.attrs['d_ff'] = self.metadata.d_ff
            if self.metadata.lstm_units is not None:
                meta_group.attrs['lstm_units'] = self.metadata.lstm_units

            config_group = f.create_group('config')
            config_group.attrs['config_json'] = json.dumps(asdict(self.config))

            encoder_group = f.create_group('encoder')
            encoder_state = self.encoder.get_state()
            encoder_group.attrs['state_json'] = json.dumps(encoder_state)

            # Save vocabulary if present
            if self.vocabulary is not None:
                vocab_group = f.create_group('vocabulary')
                vocab_group.attrs['vocab_json'] = json.dumps(self.vocabulary.to_dict())

        print(f"Saved model bundle:")
        print(f"  - Metadata: {h5_path}")
        print(f"  - Model: {model_path}")
        print(f"  - Encoder: {self.metadata.encoder_type}, vocab_size={self.metadata.vocab_size}")
        if self.vocabulary is not None:
            print(f"  - Vocabulary: {self.vocabulary.num_genres} genres, {self.vocabulary.num_active_instruments} instruments")

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        custom_objects: Optional[Dict[str, Any]] = None,
    ) -> "ModelBundle":
        """
        Load model bundle from files.

        Args:
            filepath: Path to the .h5 metadata file
            custom_objects: Custom Keras objects for model loading

        Returns:
            ModelBundle instance
        """
        from . import (
            TransformerModel,
            TransformerBlock,
            RelativeMultiHeadAttention,
            RelativePositionalEmbedding,
            LSTMModel, LSTMWithAttention
        )
        from ..data.encoders import EventEncoder, REMIEncoder, MultiTrackEncoder

        filepath = Path(filepath)

        if filepath.suffix != '.h5':
            h5_path = filepath.with_suffix('.h5')
        else:
            h5_path = filepath

        with h5py.File(h5_path, 'r') as f:
            meta = f['metadata']
            model_type = _decode_attr(meta.attrs['model_type'])
            encoder_type = _decode_attr(meta.attrs['encoder_type'])

            metadata = ModelMetadata(
                model_name=_decode_attr(meta.attrs['model_name']),
                model_type=model_type,
                vocab_size=int(meta.attrs['vocab_size']),
                max_seq_length=int(meta.attrs['max_seq_length']),
                d_model=int(meta.attrs['d_model']),
                num_layers=int(meta.attrs['num_layers']) if 'num_layers' in meta.attrs else None,
                num_heads=int(meta.attrs['num_heads']) if 'num_heads' in meta.attrs else None,
                d_ff=int(meta.attrs['d_ff']) if 'd_ff' in meta.attrs else None,
                lstm_units=int(meta.attrs['lstm_units']) if 'lstm_units' in meta.attrs else None,
                encoder_type=encoder_type,
                pad_token_id=int(meta.attrs['pad_token_id']),
                bos_token_id=int(meta.attrs['bos_token_id']),
                eos_token_id=int(meta.attrs['eos_token_id']),
            )

            config_json = _decode_attr(f['config'].attrs['config_json'])
            config_dict = json.loads(config_json)
            config = TrainingConfig(**config_dict)

            encoder_state_json = _decode_attr(f['encoder'].attrs['state_json'])
            encoder_state = json.loads(encoder_state_json)

            # Load vocabulary if present
            vocabulary = None
            if 'vocabulary' in f:
                vocab_json = _decode_attr(f['vocabulary'].attrs['vocab_json'])
                vocabulary = Vocabulary.from_dict(json.loads(vocab_json))

            stored_model_path = _decode_attr(f.attrs['keras_model_path'])

        # Find model file/directory (supports both old _model dir and new .keras format)
        base_path = h5_path.with_suffix('')
        model_candidates = [
            Path(str(base_path) + '_model.keras'),  # New Keras 3 format
            Path(str(base_path) + '_model'),         # Old SavedModel format
            h5_path.parent / stored_model_path,      # Stored path from metadata
        ]

        model_path = None
        for candidate in model_candidates:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            raise FileNotFoundError(
                f"Could not find model. Tried: {[str(c) for c in model_candidates]}"
            )

        # Create encoder from state
        if encoder_type == 'event':
            encoder = EventEncoder.from_state(encoder_state)
        elif encoder_type == 'remi':
            encoder = REMIEncoder.from_state(encoder_state)
        elif encoder_type == 'multitrack':
            encoder = MultiTrackEncoder.from_state(encoder_state)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Build custom objects for Keras
        default_custom_objects = {
            'TransformerModel': TransformerModel,
            'TransformerBlock': TransformerBlock,
            'RelativeMultiHeadAttention': RelativeMultiHeadAttention,
            'RelativePositionalEmbedding': RelativePositionalEmbedding,
            'LSTMModel': LSTMModel,
            'LSTMWithAttention': LSTMWithAttention,
        }
        if custom_objects:
            default_custom_objects.update(custom_objects)

        # Load Keras model
        print(f"Loading {model_type} model from: {model_path}")
        model = keras.models.load_model(
            model_path,
            custom_objects=default_custom_objects,
            compile=False,
        )

        bundle = cls(
            model=model,
            encoder=encoder,
            config=config,
            model_name=metadata.model_name,
            vocabulary=vocabulary,
        )
        bundle.metadata = metadata  # Use loaded metadata

        print(f"Loaded model bundle from: {h5_path}")
        print(f"  - Type: {model_type}")
        print(f"  - Encoder: {encoder_type}, vocab_size={metadata.vocab_size}")
        if vocabulary is not None:
            print(f"  - Vocabulary: {vocabulary.num_genres} genres, {vocabulary.num_active_instruments} instruments")

        return bundle

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size

    @property
    def max_seq_length(self) -> int:
        return self.config.max_seq_length

    @property
    def model_type(self) -> str:
        return self.config.model_type

    def create_start_tokens(
        self,
        genre_id: Optional[int] = None,
        instrument_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create conditioning tokens for generation.

        Args:
            genre_id: Genre ID for conditioning
            instrument_id: Instrument ID for conditioning

        Returns:
            Array of start tokens
        """
        return self.encoder.create_conditioning_tokens(
            genre_id=genre_id,
            instrument_id=instrument_id,
        )

    def generate(
        self,
        genre_id: Optional[int] = None,
        instrument_id: Optional[int] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        ignore_eos: bool = False,
    ) -> np.ndarray:
        """
        Generate a token sequence.

        Args:
            genre_id: Genre ID for conditioning
            instrument_id: Instrument ID for conditioning
            max_length: Maximum sequence length
            min_length: Minimum length before allowing EOS to stop
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            ignore_eos: If True, don't stop at EOS token

        Returns:
            Array of generated token IDs
        """
        from ..data.encoders.multitrack_encoder import MultiTrackEncoder

        max_length = max_length or self.max_seq_length
        start_tokens = self.create_start_tokens(genre_id, instrument_id)

        # Multi-track models expect an initial bar token after conditioning
        if isinstance(self.encoder, MultiTrackEncoder):
            start_tokens = np.append(start_tokens, self.encoder.bar_token(0))

        # Add batch dimension if needed (model expects 2D: batch, length)
        if len(start_tokens.shape) == 1:
            start_tokens = start_tokens[np.newaxis, :]  # Shape: (1, length)

        # If ignore_eos, don't pass eos_token_id
        eos_id = None if ignore_eos else self.encoder.eos_token_id

        generated = self.model.generate(
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_id,
            min_length=min_length,
        )

        # Convert to numpy if tensor
        if hasattr(generated, 'numpy'):
            generated = generated.numpy()

        # Remove batch dimension for return
        if len(generated.shape) > 1 and generated.shape[0] == 1:
            generated = generated[0]

        return generated

    def decode_tokens(self, tokens: np.ndarray) -> list:
        """
        Decode tokens to events.

        Args:
            tokens: Array of token IDs

        Returns:
            List of (event_type, value) tuples
        """
        return self.encoder.decode_tokens(tokens, skip_special=True)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            f"Model Bundle Summary ({self.metadata.model_type.upper()})",
            "=" * 60,
            f"Model Name: {self.model_name}",
            f"Model Type: {self.metadata.model_type}",
            "",
            "Vocabulary:",
            f"  Vocab Size: {self.metadata.vocab_size}",
            f"  Max Sequence Length: {self.metadata.max_seq_length}",
            f"  Encoder Type: {self.metadata.encoder_type}",
            "",
        ]

        if self.metadata.model_type == "transformer":
            lines.extend([
                "Transformer Architecture:",
                f"  Layers: {self.metadata.num_layers}",
                f"  Model Dimension: {self.metadata.d_model}",
                f"  Attention Heads: {self.metadata.num_heads}",
                f"  Feed-Forward Dim: {self.metadata.d_ff}",
            ])
        else:
            lines.extend([
                "LSTM Architecture:",
                f"  Model Dimension: {self.metadata.d_model}",
                f"  LSTM Units: {self.metadata.lstm_units}",
            ])

        if self.vocabulary is not None:
            lines.extend([
                "",
                "Dataset Vocabulary:",
                f"  Genres: {self.vocabulary.num_genres} ({', '.join(self.vocabulary.genres[:5])}{'...' if self.vocabulary.num_genres > 5 else ''})",
                f"  Active Instruments: {self.vocabulary.num_active_instruments}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


def _decode_attr(value) -> str:
    """Decode HDF5 attribute to string."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def load_model_bundle(
    filepath: Union[str, Path],
    custom_objects: Optional[Dict[str, Any]] = None,
) -> ModelBundle:
    """
    Load a model bundle from file.

    Args:
        filepath: Path to the .h5 metadata file
        custom_objects: Custom Keras objects for model loading

    Returns:
        ModelBundle instance
    """
    return ModelBundle.load(filepath, custom_objects)
