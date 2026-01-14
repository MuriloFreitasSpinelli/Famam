"""
SavedModel wrapper that bundles a trained model with its vocabulary.

This ensures the same vocabulary mappings used during training are available
during generation/inference.
"""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import sys

from data.configs.tensorflow_dataset_config import TensorflowDatasetConfig
from src.data.enhanced_music_dataset import EnhancedMusicDataset
from training.configs.training_config import TrainingConfig

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf

from src.data.dataset_vocabulary import DatasetVocabulary
from src.training.trainer import DatasetInfo, ConditioningType


@dataclass
class ModelMetadata:
    """Metadata about the trained model."""
    model_name: str
    conditioning_type: str  # ConditioningType name
    music_shape: tuple
    has_genre: bool
    has_instruments: bool
    has_artist: bool
    max_tracks: int
    num_genres: int
    num_artists: int
    num_instruments: int
    training_config: TrainingConfig
    tensorflow_dataconfig: Optional[TensorflowDatasetConfig] = None
    enhanced_dataset: Optional[EnhancedMusicDataset] = None

class SavedModel:
    """
    Bundles a trained Keras model with its vocabulary and metadata.

    Saves to HDF5 format containing:
    - Model weights
    - Vocabulary (genre_to_id, artist_to_id, instrument_to_id)
    - Model metadata (shapes, conditioning type, etc.)
    - Training config
    """

    def __init__(
        self,
        model: tf.keras.Model, # type: ignore
        vocabulary: DatasetVocabulary,
        dataset_info: DatasetInfo,
        training_config: TrainingConfig,
        tensorflow_dataconfig: Optional[TensorflowDatasetConfig] = None,
        enhanced_dataset: Optional[EnhancedMusicDataset] = None,
        model_name: str = "music_generator",
    ):
        self.model = model
        self.vocabulary = vocabulary
        self.dataset_info = dataset_info
        self.training_config = training_config
        self.model_name = model_name

        # Build metadata
        self.metadata = ModelMetadata(
            model_name=model_name,
            conditioning_type=dataset_info.conditioning_type.name,
            music_shape=dataset_info.music_shape,
            has_genre=dataset_info.has_genre,
            has_instruments=dataset_info.has_instruments,
            has_artist=dataset_info.has_artist,
            max_tracks=dataset_info.max_tracks,
            num_genres=vocabulary.num_genres,
            num_artists=vocabulary.num_artists,
            num_instruments=vocabulary.num_instruments,
            training_config=training_config,
            tensorflow_dataconfig=tensorflow_dataconfig,
            enhanced_dataset=enhanced_dataset
        )

    def save(self, filepath: str) -> None:
        """
        Save model, vocabulary, and metadata to HDF5 file.

        Args:
            filepath: Path to save the .h5 file
        """
        filepath = Path(filepath) # type: ignore
        filepath.parent.mkdir(parents=True, exist_ok=True) # type: ignore

        # First save the Keras model to a temp file, then embed in HDF5
        temp_model_path = filepath.parent / f".temp_{filepath.stem}.keras" # type: ignore
        self.model.save(temp_model_path)

        with h5py.File(filepath, 'w') as f:
            # === Save vocabulary ===
            vocab_group = f.create_group('vocabulary')
            vocab_group.attrs['genre_to_id'] = json.dumps(self.vocabulary.genre_to_id)
            vocab_group.attrs['artist_to_id'] = json.dumps(self.vocabulary.artist_to_id)
            vocab_group.attrs['instrument_to_id'] = json.dumps(self.vocabulary.instrument_to_id)

            # === Save metadata ===
            meta_group = f.create_group('metadata')
            meta_group.attrs['model_name'] = self.metadata.model_name
            meta_group.attrs['conditioning_type'] = self.metadata.conditioning_type
            meta_group.attrs['music_shape'] = json.dumps(self.metadata.music_shape)
            meta_group.attrs['has_genre'] = self.metadata.has_genre
            meta_group.attrs['has_instruments'] = self.metadata.has_instruments
            meta_group.attrs['has_artist'] = self.metadata.has_artist
            meta_group.attrs['max_tracks'] = self.metadata.max_tracks
            meta_group.attrs['num_genres'] = self.metadata.num_genres
            meta_group.attrs['num_artists'] = self.metadata.num_artists
            meta_group.attrs['num_instruments'] = self.metadata.num_instruments
            meta_group.attrs['training_config'] = json.dumps(self.metadata.training_config)
            if self.metadata.tensorflow_dataconfig is not None:
                meta_group.attrs['tensorflow_dataconfig'] = json.dumps(asdict(self.metadata.tensorflow_dataconfig))

            # === Save model path reference ===
            # Store the keras model file path (saved separately)
            keras_model_path = filepath.with_suffix('.keras') # type: ignore
            f.attrs['keras_model_path'] = str(keras_model_path.name)

        # Move temp model to final location
        keras_model_path = filepath.with_suffix('.keras') # type: ignore
        if keras_model_path.exists():
            keras_model_path.unlink()
        temp_model_path.rename(keras_model_path)

        print(f"Saved model bundle to: {filepath}")
        print(f"  - Vocabulary: {self.vocabulary.num_genres} genres, {self.vocabulary.num_artists} artists")
        print(f"  - Conditioning: {self.metadata.conditioning_type}")
        print(f"  - Keras model: {keras_model_path}")

    @classmethod
    def load(cls, filepath: str) -> 'SavedModel':
        """
        Load model, vocabulary, and metadata from HDF5 file.

        Args:
            filepath: Path to the .h5 file

        Returns:
            SavedModel instance
        """
        filepath = Path(filepath) # type: ignore

        with h5py.File(filepath, 'r') as f:
            # === Load vocabulary ===
            vocab = DatasetVocabulary()
            vocab.genre_to_id = json.loads(f['vocabulary'].attrs['genre_to_id']) # type: ignore
            vocab.artist_to_id = json.loads(f['vocabulary'].attrs['artist_to_id']) # type: ignore
            vocab.instrument_to_id = json.loads(f['vocabulary'].attrs['instrument_to_id']) # type: ignore

            # === Load metadata ===
            meta = f['metadata']
            model_name = meta.attrs['model_name']
            conditioning_type = ConditioningType[meta.attrs['conditioning_type']] # type: ignore
            music_shape = tuple(json.loads(meta.attrs['music_shape'])) # type: ignore
            training_config = json.loads(meta.attrs['training_config']) # type: ignore

            # Load tensorflow_dataconfig if present
            if 'tensorflow_dataconfig' in meta.attrs:
                tf_config_dict = json.loads(meta.attrs['tensorflow_dataconfig']) # type: ignore
                tensorflow_dataconfig = TensorflowDatasetConfig(**tf_config_dict)
            else:
                tensorflow_dataconfig = None

            dataset_info = DatasetInfo(
                music_shape=music_shape,
                conditioning_type=conditioning_type,
                has_genre=bool(meta.attrs['has_genre']),
                has_instruments=bool(meta.attrs['has_instruments']),
                has_artist=bool(meta.attrs['has_artist']),
                max_tracks=int(meta.attrs['max_tracks']), # type: ignore
            )

            # === Load Keras model ===
            keras_model_path = filepath.parent / f.attrs['keras_model_path'] # type: ignore

        model = tf.keras.models.load_model(keras_model_path) # type: ignore

        saved_model = cls(
            model=model,
            vocabulary=vocab,
            dataset_info=dataset_info,
            training_config=training_config,
            tensorflow_dataconfig=tensorflow_dataconfig,
            enhanced_dataset=None,  # Not stored in bundle, load separately if needed
            model_name=model_name, # type: ignore
        )

        print(f"Loaded model bundle from: {filepath}")
        print(f"  - Vocabulary: {vocab.num_genres} genres, {vocab.num_artists} artists")
        print(f"  - Conditioning: {conditioning_type.name}")
        if tensorflow_dataconfig:
            print(f"  - Representation: {tensorflow_dataconfig.representation_type}")

        return saved_model

    # === Convenience methods for generation ===

    def get_genre_id(self, genre_name: str) -> int:
        """Get genre ID for generation."""
        return self.vocabulary.get_genre_id(genre_name)

    def get_artist_id(self, artist_name: str) -> int:
        """Get artist ID for generation."""
        return self.vocabulary.get_artist_id(artist_name)

    def get_instrument_id(self, instrument_name: str) -> int:
        """Get instrument ID for generation."""
        return self.vocabulary.get_instrument_id(instrument_name)

    def list_genres(self) -> list:
        """List available genres."""
        return list(self.vocabulary.genre_to_id.keys())

    def list_artists(self) -> list:
        """List available artists."""
        return list(self.vocabulary.artist_to_id.keys())

    def predict(
        self,
        music_input: np.ndarray,
        genre: Optional[str] = None,
        instruments: Optional[list] = None,
        artist: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate prediction with conditioning.

        Args:
            music_input: Input music array
            genre: Genre name (optional, uses vocabulary)
            instruments: List of instrument names or program numbers
            artist: Artist name (optional, uses vocabulary)

        Returns:
            Model prediction
        """
        # Build input dict
        inputs = {'music_input': np.expand_dims(music_input, 0)}

        if self.dataset_info.has_genre and genre is not None:
            genre_id = self.get_genre_id(genre)
            inputs['genre_input'] = np.array([[genre_id]], dtype=np.int32)

        if self.dataset_info.has_artist and artist is not None:
            artist_id = self.get_artist_id(artist)
            inputs['artist_input'] = np.array([[artist_id]], dtype=np.int32)

        if self.dataset_info.has_instruments and instruments is not None:
            # Convert instrument names to IDs
            instrument_ids = []
            for inst in instruments:
                if isinstance(inst, str):
                    instrument_ids.append(self.get_instrument_id(inst))
                else:
                    instrument_ids.append(int(inst))

            # Pad to max_tracks
            while len(instrument_ids) < self.dataset_info.max_tracks:
                instrument_ids.append(0)
            instrument_ids = instrument_ids[:self.dataset_info.max_tracks]

            inputs['instrument_input'] = np.array([instrument_ids], dtype=np.int32)

        return self.model.predict(inputs, verbose=0)[0]
