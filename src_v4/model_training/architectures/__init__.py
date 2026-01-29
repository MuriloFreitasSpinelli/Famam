"""
Model architectures module.

Provides neural network architectures for music generation.
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
]
