"""
Model training module for music generation.

Provides unified training infrastructure for Transformer and LSTM models
working with event-based token sequences.
"""

from .config import (
    TrainingConfig,
    get_transformer_small,
    get_transformer_medium,
    get_transformer_large,
    get_lstm_small,
    get_lstm_medium,
)


from .architectures import (
    LSTMModel,
    LSTMWithAttention,
    build_lstm_from_config,
    BaseMusicModel, 
    get_causal_attention_mask,
    TransformerModel,
    TransformerBlock,
    RelativeMultiHeadAttention,
    RelativePositionalEmbedding,
    build_transformer_from_config,
)

from .pipeline.trainer import (
    Trainer,
    train_model,
    TransformerLRSchedule,
    MaskedSparseCategoricalCrossentropy,
    MaskedAccuracy,
)
from .model_bundle import (
    ModelBundle,
    ModelMetadata,
    load_model_bundle,
)

__all__ = [
    # Config
    'TrainingConfig',
    'get_transformer_small',
    'get_transformer_medium',
    'get_transformer_large',
    'get_lstm_small',
    'get_lstm_medium',
    # Base
    'BaseMusicModel',
    'get_causal_attention_mask',
    # Transformer
    'TransformerModel',
    'TransformerBlock',
    'RelativeMultiHeadAttention',
    'RelativePositionalEmbedding',
    'build_transformer_from_config',
    # LSTM
    'LSTMModel',
    'LSTMWithAttention',
    'build_lstm_from_config',
    # Trainer
    'Trainer',
    'train_model',
    'TransformerLRSchedule',
    'MaskedSparseCategoricalCrossentropy',
    'MaskedAccuracy',
    # Model Bundle
    'ModelBundle',
    'ModelMetadata',
    'load_model_bundle',
]