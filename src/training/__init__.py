"""
Training module for music generation models.

Provides training infrastructure for Transformer and LSTM models.
"""

from .trainer import (
    Trainer,
    train_model,
    TransformerLRSchedule,
    MaskedSparseCategoricalCrossentropy,
    MaskedAccuracy,
)

__all__ = [
    'Trainer',
    'train_model',
    'TransformerLRSchedule',
    'MaskedSparseCategoricalCrossentropy',
    'MaskedAccuracy',
]
