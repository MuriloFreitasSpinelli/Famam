"""
Models module.

Provides neural network architectures and model bundling for music generation.
"""

from .base_model import BaseMusicModel, get_causal_attention_mask
from .transformer_model import (
    TransformerModel,
    TransformerBlock,
    RelativeMultiHeadAttention,
    RelativePositionalEmbedding,
    build_transformer_from_config,
)
from .lstm_model import (
    LSTMModel,
    LSTMWithAttention,
    build_lstm_from_config,
)
from .model_bundle import (
    ModelBundle,
    ModelMetadata,
    load_model_bundle,
)

__all__ = [
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
    # Bundle
    'ModelBundle',
    'ModelMetadata',
    'load_model_bundle',
]
