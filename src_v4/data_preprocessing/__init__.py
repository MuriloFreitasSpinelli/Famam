"""
Data preprocessing module for music generation.

Provides encoders and dataset utilities for converting MIDI/music data
into formats suitable for model training.
"""

from .encoders import (
    BaseEncoder,
    EncodedSequence,
    EventEncoder,
    EventVocabulary,
    REMIEncoder,
    REMIVocabulary,
    MultiTrackEncoder,
)
from .config import MusicDatasetConfig
from .vocabulary import Vocabulary, GENERAL_MIDI_INSTRUMENTS, DRUM_PROGRAM_ID
from .music_dataset import MusicDataset, MusicEntry
from .scripts.dataset_builder import (
    build_dataset,
    build_and_save_dataset,
    create_encoder_from_config,
    load_genre_map,
    find_midi_files,
)
from .scripts.preprocessing import (
    preprocess_music,
    adjust_resolution,
    quantize_music,
    remove_empty_tracks,
    segment_music,
    transpose_music,
    vary_tempo,
)

__all__ = [
    # Encoders
    'BaseEncoder',
    'EncodedSequence',
    'EventEncoder',
    'EventVocabulary',
    'REMIEncoder',
    'REMIVocabulary',
    'MultiTrackEncoder',
    # Config
    'MusicDatasetConfig',
    # Vocabulary
    'Vocabulary',
    'GENERAL_MIDI_INSTRUMENTS',
    'DRUM_PROGRAM_ID',
    # Dataset
    'MusicDataset',
    'MusicEntry',
    # Builder
    'build_dataset',
    'build_and_save_dataset',
    'create_encoder_from_config',
    'load_genre_map',
    'find_midi_files',
    # Preprocessing
    'preprocess_music',
    'adjust_resolution',
    'quantize_music',
    'remove_empty_tracks',
    'segment_music',
    'transpose_music',
    'vary_tempo',
]