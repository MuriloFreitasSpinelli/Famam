"""
Data preprocessing pipeline module.

Provides dataset building and preprocessing utilities.
"""

from .dataset_builder import (
    build_dataset,
    build_and_save_dataset,
    create_encoder_from_config,
    load_genre_map,
    find_midi_files,
    get_genre_for_file,
    get_duration_seconds,
    passes_filters,
)
from .preprocessing import (
    preprocess_music,
    adjust_resolution,
    quantize_music,
    remove_empty_tracks,
    segment_music,
    transpose_music,
    vary_tempo,
    generate_transpositions,
    generate_tempo_variations,
)

__all__ = [
    # Dataset builder
    'build_dataset',
    'build_and_save_dataset',
    'create_encoder_from_config',
    'load_genre_map',
    'find_midi_files',
    'get_genre_for_file',
    'get_duration_seconds',
    'passes_filters',
    # Preprocessing
    'preprocess_music',
    'adjust_resolution',
    'quantize_music',
    'remove_empty_tracks',
    'segment_music',
    'transpose_music',
    'vary_tempo',
    'generate_transpositions',
    'generate_tempo_variations',
]
