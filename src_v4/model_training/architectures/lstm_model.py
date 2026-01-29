"""
LSTM model for autoregressive music generation.

Works with event-based token sequences, predicting the next token
in an autoregressive fashion (same as Transformer).
"""

from typing import Optional, Dict, Any, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base_model import BaseMusicModel
from ..config import TrainingConfig


class LSTMModel(BaseMusicModel):
    """
    LSTM model for autoregressive music generation.

    Uses stacked LSTM layers to process token sequences and predict
    the next token. Can be bidirectional for encoding tasks, but
    uses unidirectional (forward-only) for generation.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        d_model: int = 512,
        lstm_units: Tuple[int, ...] = (512, 512),
        dropout_rate: float = 0.1,
        recurrent_dropout: float = 0.0,
        bidirectional: bool = False,
        **kwargs
    ):
        """
        Initialize LSTM model.

        Args:
            vocab_size: Size of token vocabulary
            max_seq_length: Maximum sequence length
            d_model: Embedding dimension
            lstm_units: Tuple of units for each LSTM layer
            dropout_rate: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        self.lstm_units = lstm_units
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional

        super().__init__(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            d_model=d_model,
            dropout_rate=dropout_rate,
            **kwargs
        )

    def build_layers(self) -> None:
        """Build LSTM layers."""
        self.embedding_dropout = layers.Dropout(self.dropout_rate)

        self.lstm_layers = []
        self.dropout_layers = []

        for i, units in enumerate(self.lstm_units):
            # All LSTM layers return sequences for seq-to-seq
            lstm = layers.LSTM(
                units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                name=f'lstm_{i}',
            )

            if self.bidirectional:
                lstm = layers.Bidirectional(lstm, name=f'bilstm_{i}')

            self.lstm_layers.append(lstm)

            # Add dropout between layers (except after last)
            if i < len(self.lstm_units) - 1:
                self.dropout_layers.append(
                    layers.Dropout(self.dropout_rate, name=f'dropout_{i}')
                )

        # Project LSTM output to embedding dimension if needed
        lstm_output_dim = self.lstm_units[-1]
        if self.bidirectional:
            lstm_output_dim *= 2

        if lstm_output_dim != self.d_model:
            self.projection = layers.Dense(self.d_model, name='projection')
        else:
            self.projection = None

        self.final_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs: Dict with 'input_ids' and optional 'attention_mask'
            training: Training mode flag

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            mask = inputs.get('attention_mask', None)
        else:
            input_ids = inputs
            mask = None

        # Embed tokens
        x = self.embed_tokens(input_ids, training=training)
        x = self.embedding_dropout(x, training=training)

        # Convert attention mask to LSTM mask format if provided
        lstm_mask = None
        if mask is not None:
            lstm_mask = tf.cast(mask, tf.bool)

        # Pass through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x = lstm(x, mask=lstm_mask, training=training)

            # Apply dropout between layers
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x, training=training)

        # Project if needed
        if self.projection is not None:
            x = self.projection(x)

        # Final norm and output projection
        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'd_model': self.d_model,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout': self.recurrent_dropout,
            'bidirectional': self.bidirectional,
        }

    def summary_str(self) -> str:
        lines = [
            super().summary_str(),
            f"  LSTM Units: {self.lstm_units}",
            f"  Bidirectional: {self.bidirectional}",
            f"  Recurrent Dropout: {self.recurrent_dropout}",
        ]
        return "\n".join(lines)


class LSTMWithAttention(BaseMusicModel):
    """
    LSTM with self-attention for music generation.

    Combines LSTM's sequential processing with attention mechanism
    for better long-range dependencies.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        d_model: int = 512,
        lstm_units: Tuple[int, ...] = (512,),
        num_attention_heads: int = 8,
        dropout_rate: float = 0.1,
        recurrent_dropout: float = 0.0,
        **kwargs
    ):
        self.lstm_units = lstm_units
        self.num_attention_heads = num_attention_heads
        self.recurrent_dropout = recurrent_dropout

        super().__init__(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            d_model=d_model,
            dropout_rate=dropout_rate,
            **kwargs
        )

    def build_layers(self) -> None:
        """Build LSTM + Attention layers."""
        self.embedding_dropout = layers.Dropout(self.dropout_rate)

        # LSTM layers
        self.lstm_layers = []
        for i, units in enumerate(self.lstm_units):
            lstm = layers.LSTM(
                units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                name=f'lstm_{i}',
            )
            self.lstm_layers.append(lstm)

        # Project to d_model if needed
        if self.lstm_units[-1] != self.d_model:
            self.lstm_projection = layers.Dense(self.d_model, name='lstm_projection')
        else:
            self.lstm_projection = None

        # Self-attention layer
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.d_model // self.num_attention_heads,
            dropout=self.dropout_rate,
            name='self_attention',
        )

        self.attention_norm = layers.LayerNormalization(epsilon=1e-6)
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            mask = inputs.get('attention_mask', None)
        else:
            input_ids = inputs
            mask = None

        seq_len = tf.shape(input_ids)[1]

        # Embed
        x = self.embed_tokens(input_ids, training=training)
        x = self.embedding_dropout(x, training=training)

        # LSTM
        lstm_mask = tf.cast(mask, tf.bool) if mask is not None else None
        for lstm in self.lstm_layers:
            x = lstm(x, mask=lstm_mask, training=training)

        if self.lstm_projection is not None:
            x = self.lstm_projection(x)

        # Self-attention with causal mask
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)

        x_norm = self.attention_norm(x)
        attn_output = self.attention(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            attention_mask=causal_mask,
            training=training,
        )
        x = x + attn_output

        # Output
        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'd_model': self.d_model,
            'lstm_units': self.lstm_units,
            'num_attention_heads': self.num_attention_heads,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout': self.recurrent_dropout,
        }


def build_lstm_from_config(
    config: TrainingConfig,
    vocab_size: int,
) -> LSTMModel:
    """
    Build LSTM model from configuration.

    Args:
        config: TrainingConfig instance
        vocab_size: Size of token vocabulary

    Returns:
        LSTMModel instance
    """
    return LSTMModel(
        vocab_size=vocab_size,
        max_seq_length=config.max_seq_length,
        d_model=config.d_model,
        lstm_units=config.lstm_units,
        dropout_rate=config.dropout_rate,
        recurrent_dropout=config.recurrent_dropout,
        bidirectional=config.bidirectional,
    )
