"""
Model training pipeline module.

Provides training infrastructure for music generation models.
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
