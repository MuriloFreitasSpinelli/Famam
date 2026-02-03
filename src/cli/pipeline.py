"""
MusicPipeline - Unified interface for the music generation pipeline.

Handles the complete workflow:
    1. Dataset building from MIDI files
    2. Model training (Transformer or LSTM)
    3. Music generation and MIDI export

Author: Murilo de Freitas Spinelli
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import muspy

from ..data import (
    Vocabulary,
    MusicDataset,
    BaseEncoder,
    EventEncoder,
    REMIEncoder,
    MultiTrackEncoder,
    build_dataset,
    build_and_save_dataset,
    create_encoder_from_config,
)

from ..config import ( 
    MusicDatasetConfig,
    TrainingConfig,
)


from ..training import Trainer

from ..models import (
    ModelBundle,
    load_model_bundle,
    TransformerModel,
    build_transformer_from_config,
    LSTMModel,
    build_lstm_from_config
)

from ..generation import (
    MusicGenerator,
    GenerationConfig,
    MultiTrackGenerator,
    MultiTrackConfig,
)


@dataclass
class PipelineConfig:
    """
    Configuration for the complete music pipeline.

    Combines dataset, training, and generation settings.
    """

    # === Paths ===
    data_dir: str = "data/midi"
    output_dir: str = "output"
    model_dir: str = "models"

    # === Dataset ===
    encoder_type: str = "event"  # "event", "remi", or "multitrack"
    resolution: int = 24
    max_seq_length: int = 1024
    train_split: float = 0.8
    val_split: float = 0.1
    multitrack: bool = False  # Use multi-track encoding (all instruments together)

    # === Model ===
    model_type: str = "transformer"  # "transformer" or "lstm"
    model_name: str = "music_model"

    # Transformer settings
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    # LSTM settings
    lstm_units: int = 512

    # === Training ===
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    warmup_steps: int = 4000

    # === Generation ===
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    def to_dataset_config(self) -> MusicDatasetConfig:
        """Convert to MusicDatasetConfig."""
        return MusicDatasetConfig(
            midi_dir=self.data_dir,
            output_dir=self.output_dir,
            resolution=self.resolution,
            max_seq_length=self.max_seq_length,
            encoder_type=self.encoder_type,
            train_split=self.train_split,
            val_split=self.val_split,
        )

    def to_training_config(self, vocab_size: int) -> TrainingConfig:
        """Convert to TrainingConfig."""
        return TrainingConfig(
            model_name=self.model_name,
            model_type=self.model_type,
            vocab_size=vocab_size,
            max_seq_length=self.max_seq_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout_rate=self.dropout,
            lstm_units=self.lstm_units,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            output_dir=self.model_dir,
        )

    def to_generation_config(self) -> GenerationConfig:
        """Convert to GenerationConfig."""
        return GenerationConfig(
            max_length=self.max_seq_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            resolution=self.resolution,
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Save config to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PipelineConfig":
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MusicPipeline:
    """
    High-level API for the complete music generation pipeline.

    Provides methods for:
        - Building datasets from MIDI files
        - Training models
        - Generating music
        - Saving and loading complete pipelines
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()

        # Components (initialized lazily or via methods)
        self.encoder: Optional[BaseEncoder] = None
        self.vocabulary: Optional[Vocabulary] = None
        self.dataset: Optional[MusicDataset] = None
        self.model: Optional[Union[TransformerModel, LSTMModel]] = None
        self.trainer: Optional[Trainer] = None
        self.generator: Optional[MusicGenerator] = None
        self.bundle: Optional[ModelBundle] = None

        # Training history
        self.history: Optional[Dict[str, List[float]]] = None

    # =========================================================================
    # Dataset Building
    # =========================================================================

    def create_encoder(
        self,
        num_genres: int = 0,
        num_instruments: int = 129,
    ) -> BaseEncoder:
        """
        Create an encoder based on config.

        Args:
            num_genres: Number of genre conditioning tokens
            num_instruments: Number of instrument tokens

        Returns:
            Encoder instance
        """
        if self.config.encoder_type == "multitrack" or self.config.multitrack:
            self.encoder = MultiTrackEncoder(
                num_genres=num_genres,
                resolution=self.config.resolution,
            )
        elif self.config.encoder_type == "remi":
            self.encoder = REMIEncoder(
                num_genres=num_genres,
                num_instruments=num_instruments,
                resolution=self.config.resolution,
            )
        else:
            self.encoder = EventEncoder(
                num_genres=num_genres,
                num_instruments=num_instruments,
            )

        encoder_name = "multitrack" if self.config.multitrack else self.config.encoder_type
        print(f"Created {encoder_name} encoder: vocab_size={self.encoder.vocab_size}")
        return self.encoder

    def build_dataset(
        self,
        midi_dir: Optional[str] = None,
        output_path: Optional[str] = None,
        genre_filter: Optional[List[str]] = None,
    ) -> MusicDataset:
        """
        Build dataset from MIDI files.

        Args:
            midi_dir: Directory containing MIDI files (uses config if None)
            output_path: Path to save dataset (optional)
            genre_filter: Only include these genres (optional)

        Returns:
            MusicDataset instance
        """
        midi_dir = midi_dir or self.config.data_dir
        dataset_config = self.config.to_dataset_config()
        dataset_config.midi_dir = midi_dir

        if genre_filter:
            dataset_config.genre_filter = genre_filter

        print(f"Building dataset from: {midi_dir}")

        if output_path:
            self.dataset, self.encoder = build_and_save_dataset(
                config=dataset_config,
                output_path=output_path,
            )
        else:
            self.dataset, self.encoder = build_dataset(config=dataset_config)

        self.vocabulary = self.dataset.vocabulary
        print(f"Dataset: {len(self.dataset)} entries")
        print(f"Vocabulary: {self.vocabulary.num_genres} genres, {self.vocabulary.num_active_instruments} instruments")

        return self.dataset

    def load_dataset(self, filepath: Union[str, Path]) -> MusicDataset:
        """
        Load a saved dataset.

        Args:
            filepath: Path to the .h5 dataset file

        Returns:
            MusicDataset instance
        """
        self.dataset = MusicDataset.load(filepath)
        self.vocabulary = self.dataset.vocabulary

        # Create encoder from dataset info
        num_genres = self.vocabulary.num_genres
        self.create_encoder(num_genres=num_genres)

        print(f"Loaded dataset from: {filepath}")
        print(f"  {len(self.dataset)} entries")

        return self.dataset

    def get_tf_datasets(
        self,
        batch_size: Optional[int] = None,
    ) -> Dict[str, tf.data.Dataset]:
        """
        Get TensorFlow datasets for training.

        Args:
            batch_size: Batch size (uses config if None)

        Returns:
            Dict with 'train', 'validation', and optionally 'test' datasets
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call build_dataset() or load_dataset() first.")
        if self.encoder is None:
            raise ValueError("No encoder created. Call create_encoder() first.")

        batch_size = batch_size or self.config.batch_size

        # Use appropriate dataset method based on encoder type
        if self.config.multitrack or self.config.encoder_type == "multitrack":
            datasets = self.dataset.to_multitrack_dataset(
                encoder=self.encoder,
                splits=(self.config.train_split, self.config.val_split, 1 - self.config.train_split - self.config.val_split),
                random_state=42,
            )
        else:
            datasets = self.dataset.to_tensorflow_dataset(
                encoder=self.encoder,
                splits=(self.config.train_split, self.config.val_split, 1 - self.config.train_split - self.config.val_split),
                random_state=42,
            )

        return datasets

    # =========================================================================
    # Model Training
    # =========================================================================

    def build_model(self) -> Union[TransformerModel, LSTMModel]:
        """
        Build the model based on config.

        Returns:
            Model instance (TransformerModel or LSTMModel)
        """
        if self.encoder is None:
            raise ValueError("No encoder created. Call create_encoder() or build_dataset() first.")

        training_config = self.config.to_training_config(self.encoder.vocab_size)

        if self.config.model_type == "transformer":
            self.model = build_transformer_from_config(training_config, self.encoder.vocab_size)
            print(f"Built Transformer: {self.config.num_layers} layers, d_model={self.config.d_model}")
        else:
            self.model = build_lstm_from_config(training_config, self.encoder.vocab_size)
            print(f"Built LSTM: {self.config.lstm_units} units")

        return self.model

    def train(
        self,
        train_dataset: Optional[tf.data.Dataset] = None,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset (auto-created if None)
            val_dataset: Validation dataset (auto-created if None)
            epochs: Number of epochs (uses config if None)

        Returns:
            Training history dict
        """
        if self.encoder is None:
            raise ValueError("No encoder. Call build_dataset() first.")

        # Get datasets if not provided
        if train_dataset is None or val_dataset is None:
            datasets = self.get_tf_datasets()
            train_dataset = train_dataset or datasets['train']
            val_dataset = val_dataset or datasets['validation']

        # Build model if needed
        if self.model is None:
            self.build_model()

        # Create training config
        training_config = self.config.to_training_config(self.encoder.vocab_size)
        if epochs:
            training_config.epochs = epochs

        # Create trainer
        self.trainer = Trainer(training_config, self.encoder)
        self.trainer.model = self.model

        # Train
        print(f"\nTraining {self.config.model_type} for {training_config.epochs} epochs...")
        self.model, self.history = self.trainer.train(train_dataset, val_dataset)

        return self.history

    def save_model(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Save trained model as a bundle.

        Args:
            filepath: Path to save (auto-generated if None)

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        if self.encoder is None:
            raise ValueError("No encoder. Build or load a dataset first.")

        filepath = filepath or Path(self.config.model_dir) / self.config.model_name
        filepath = Path(filepath)

        training_config = self.config.to_training_config(self.encoder.vocab_size)
        self.bundle = ModelBundle(
            model=self.model,
            encoder=self.encoder,
            config=training_config,
            model_name=self.config.model_name,
        )
        self.bundle.save(filepath)

        return str(filepath)

    def load_model(self, filepath: Union[str, Path]) -> ModelBundle:
        """
        Load a trained model bundle.

        Args:
            filepath: Path to the model bundle

        Returns:
            ModelBundle instance
        """
        self.bundle = load_model_bundle(filepath)
        self.model = self.bundle.model
        self.encoder = self.bundle.encoder

        # Create generator
        self.generator = MusicGenerator.from_bundle(
            self.bundle,
            self.config.to_generation_config(),
        )

        print(f"Loaded model: {self.bundle.model_name}")
        print(self.bundle.summary())

        return self.bundle

    # =========================================================================
    # Music Generation
    # =========================================================================

    def _ensure_generator(self) -> MusicGenerator:
        """Ensure generator is available."""
        if self.generator is None:
            if self.model is None or self.encoder is None:
                raise ValueError("No model loaded. Call train() or load_model() first.")

            self.generator = MusicGenerator(
                model=self.model,
                encoder=self.encoder,
                config=self.config.to_generation_config(),
            )

        return self.generator

    def generate(
        self,
        genre_id: Optional[int] = None,
        instrument_id: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate a token sequence.

        Args:
            genre_id: Genre conditioning ID
            instrument_id: Instrument conditioning ID
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Array of generated token IDs
        """
        generator = self._ensure_generator()

        return generator.generate_tokens(
            genre_id=genre_id,
            instrument_id=instrument_id,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def generate_midi(
        self,
        output_path: Union[str, Path],
        genre_id: Optional[int] = None,
        instrument_ids: Optional[List[int]] = None,
        include_drums: bool = True,
        temperature: Optional[float] = None,
    ) -> muspy.Music:
        """
        Generate music and save to MIDI file.

        Args:
            output_path: Path to save MIDI file
            genre_id: Genre conditioning ID
            instrument_ids: List of instrument IDs to generate
            include_drums: Whether to include a drum track
            temperature: Sampling temperature

        Returns:
            Generated muspy.Music object
        """
        generator = self._ensure_generator()

        return generator.generate_midi(
            output_path=output_path,
            genre_id=genre_id,
            instrument_ids=instrument_ids,
            include_drums=include_drums,
            temperature=temperature,
        )

    def generate_track(
        self,
        genre_id: Optional[int] = None,
        instrument_id: Optional[int] = None,
        program: int = 0,
        is_drum: bool = False,
        **kwargs,
    ) -> muspy.Track:
        """
        Generate a single instrument track.

        Args:
            genre_id: Genre conditioning ID
            instrument_id: Instrument conditioning ID
            program: MIDI program number
            is_drum: Whether this is a drum track
            **kwargs: Additional generation parameters

        Returns:
            muspy.Track object
        """
        generator = self._ensure_generator()

        return generator.generate_track(
            genre_id=genre_id,
            instrument_id=instrument_id,
            program=program,
            is_drum=is_drum,
            **kwargs,
        )

    # =========================================================================
    # Multi-Track Generation (all instruments know each other)
    # =========================================================================

    def _ensure_multitrack_generator(self) -> MultiTrackGenerator:
        """Ensure multi-track generator is available."""
        if not hasattr(self, '_multitrack_generator') or self._multitrack_generator is None:
            if self.model is None or self.encoder is None:
                raise ValueError("No model loaded. Call train() or load_model() first.")

            if not isinstance(self.encoder, MultiTrackEncoder):
                raise ValueError(
                    "Multi-track generation requires MultiTrackEncoder. "
                    "Set config.multitrack=True and retrain."
                )

            self._multitrack_generator = MultiTrackGenerator(
                model=self.model,
                encoder=self.encoder,
                config=MultiTrackConfig(
                    max_length=self.config.max_seq_length,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    resolution=self.config.resolution,
                ),
            )

        return self._multitrack_generator

    def generate_multitrack(
        self,
        output_path: Union[str, Path],
        genre_id: int = 0,
        num_bars: int = 8,
        temperature: Optional[float] = None,
    ) -> muspy.Music:
        """
        Generate music with all instruments aware of each other.

        This uses the multi-track encoder where all instruments are
        generated together in an interleaved sequence.

        Args:
            output_path: Path to save MIDI file
            genre_id: Genre conditioning ID
            num_bars: Number of bars to generate
            temperature: Sampling temperature

        Returns:
            Generated muspy.Music with multiple tracks
        """
        generator = self._ensure_multitrack_generator()

        music = generator.generate_midi(
            output_path=output_path,
            genre_id=genre_id,
            num_bars=num_bars,
            temperature=temperature,
        )

        generator.print_generation_stats(music)
        return music

    def is_multitrack_mode(self) -> bool:
        """Check if pipeline is in multi-track mode."""
        return self.config.multitrack or self.config.encoder_type == "multitrack"

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_genres(self) -> List[str]:
        """List available genres in the vocabulary."""
        if self.vocabulary is None:
            return []
        return list(self.vocabulary.genre_to_id.keys())

    def get_genre_id(self, genre_name: str) -> int:
        """Get genre ID by name."""
        if self.vocabulary is None:
            raise ValueError("No vocabulary loaded.")
        return self.vocabulary.get_genre_id(genre_name)

    def list_instruments(self) -> List[str]:
        """List all available instruments."""
        if self.vocabulary is None:
            return []
        return self.vocabulary.list_instruments()

    def get_instrument_id(self, instrument_name: str) -> int:
        """Get instrument ID by name."""
        if self.vocabulary is None:
            raise ValueError("No vocabulary loaded.")
        return self.vocabulary.get_instrument_id(instrument_name)

    def get_instrument_stats(self, top_n: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Get instrument usage statistics from the dataset.

        Args:
            top_n: If provided, only return the top N instruments

        Returns:
            List of (instrument_name, song_count) tuples sorted by count (descending)
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        return self.dataset.get_instrument_stats(top_n)

    def print_instrument_stats(self, top_n: int = 20) -> None:
        """
        Print a formatted table of instrument usage statistics.

        Args:
            top_n: Number of top instruments to display
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        self.dataset.print_instrument_stats(top_n)

    def summary(self) -> str:
        """Get pipeline summary."""
        lines = [
            "=" * 60,
            "Music Pipeline Summary",
            "=" * 60,
            f"Encoder: {self.config.encoder_type}",
            f"Model: {self.config.model_type}",
            "",
        ]

        if self.encoder:
            lines.append(f"Vocabulary size: {self.encoder.vocab_size}")

        if self.dataset:
            lines.append(f"Dataset entries: {len(self.dataset)}")

        if self.vocabulary:
            lines.append(f"Genres: {self.vocabulary.num_genres}")
            lines.append(f"Active instruments: {self.vocabulary.num_active_instruments}")

        if self.model:
            lines.append(f"Model loaded: Yes")

        if self.history:
            final_loss = self.history.get('loss', [])[-1] if self.history.get('loss') else 'N/A'
            final_val_loss = self.history.get('val_loss', [])[-1] if self.history.get('val_loss') else 'N/A'
            lines.append(f"Final loss: {final_loss}")
            lines.append(f"Final val_loss: {final_val_loss}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_config(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Save pipeline configuration."""
        filepath = filepath or Path(self.config.output_dir) / "pipeline_config.json"
        self.config.save(filepath)
        return str(filepath)

    @classmethod
    def from_config(cls, filepath: Union[str, Path]) -> "MusicPipeline":
        """Load pipeline from config file."""
        config = PipelineConfig.load(filepath)
        return cls(config)
