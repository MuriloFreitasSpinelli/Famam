"""Configuration classes for the music generation pipeline."""

from .training_config import (
    TrainingConfig,
    get_transformer_small,
    get_transformer_medium,
    get_transformer_large,
    get_lstm_small,
    get_lstm_medium,
)
from .music_dataset_config import MusicDatasetConfig

__all__ = [
    'TrainingConfig',
    'get_transformer_small',
    'get_transformer_medium',
    'get_transformer_large',
    'get_lstm_small',
    'get_lstm_medium',
    'MusicDatasetConfig',
]
