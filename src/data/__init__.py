"""
Data module for music generation.

Provides encoders and dataset utilities for converting MIDI/music data
into formats suitable for model training. Supports genre conditioning,
if genre mappings are provided.
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
from .vocabulary import Vocabulary, GENERAL_MIDI_INSTRUMENTS, DRUM_PROGRAM_ID
from .music_dataset import MusicDataset, MusicEntry
from .preprocessing import (
    build_dataset,
    build_and_save_dataset,
    create_encoder_from_config,
    load_genre_map,
    find_midi_files,
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
