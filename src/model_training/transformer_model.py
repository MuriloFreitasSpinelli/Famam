"""
GPT-style causal Transformer decoder for autoregressive music generation.

Architecture:
- Token embedding + learned positional embedding
- N transformer blocks (self-attention + FFN with pre-norm)
- Causal mask (can only attend to previous positions)
- Output: logits over vocab_size
"""

from typing import Optional, Tuple
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore

from .configs.transformer_config import TransformerTrainingConfig


def get_causal_attention_mask(seq_length: int) -> tf.Tensor:
    """
    Create a causal attention mask for autoregressive decoding.

    The mask prevents attending to future positions.

    Args:
        seq_length: Sequence length

    Returns:
        Boolean mask of shape (seq_length, seq_length) where True = masked
    """
    # Create lower triangular matrix (1s in lower triangle, 0s in upper)
    mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
    return tf.cast(mask, tf.bool)


class TransformerBlock(layers.Layer):
    """
    Single Transformer block with pre-norm architecture.

    Components:
    - LayerNorm -> Multi-Head Self-Attention -> Residual
    - LayerNorm -> Feed-Forward Network -> Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Pre-norm layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )

        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate),
        ])

    def call(self, x, attention_mask=None, training=False):
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_output = self.mha(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            attention_mask=attention_mask,
            training=training,
        )
        x = x + attn_output

        # FFN with pre-norm
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm, training=training)
        x = x + ffn_output

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    """
    Combined token and position embedding layer.

    Adds learned positional embeddings to token embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        d_model: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            name='token_embedding',
        )
        self.pos_emb = layers.Embedding(
            input_dim=max_seq_length,
            output_dim=d_model,
            name='position_embedding',
        )
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        """
        Embed tokens with position information.

        Args:
            x: Token IDs of shape (batch, seq_len)
            training: Whether in training mode

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)

        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(positions)

        # Scale token embeddings by sqrt(d_model) as in original Transformer
        embeddings = token_embeddings * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = embeddings + position_embeddings

        return self.dropout(embeddings, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config


class MusicTransformer(keras.Model):
    """
    GPT-style Transformer for autoregressive music generation.

    Takes token IDs as input and outputs logits over vocabulary.
    Uses causal masking to ensure autoregressive property.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Embedding layer
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            d_model=d_model,
            dropout_rate=dropout_rate,
        )

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                name=f'transformer_block_{i}',
            )
            for i in range(num_layers)
        ]

        # Final layer norm
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)

        # Output projection (tied with embedding weights is optional)
        self.output_projection = layers.Dense(vocab_size, name='output_projection')

    def call(self, inputs, training=False, return_hidden_states=False):
        """
        Forward pass through the transformer.

        Args:
            inputs: Dict with 'input_ids' and optional 'attention_mask'
                   or just token IDs tensor
            training: Whether in training mode
            return_hidden_states: If True, return hidden states from all layers

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
            Optionally also hidden states from each layer
        """
        # Handle dict or tensor input
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            padding_mask = inputs.get('attention_mask', None)
        else:
            input_ids = inputs
            padding_mask = None

        # Get sequence length for causal mask
        seq_len = tf.shape(input_ids)[1]

        # Create causal attention mask
        causal_mask = get_causal_attention_mask(seq_len)

        # Combine with padding mask if provided
        if padding_mask is not None:
            # padding_mask: (batch, seq_len), 1 = attend, 0 = mask
            # Convert to (batch, 1, 1, seq_len) for broadcasting
            padding_mask = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], tf.bool)
            # Invert: True = mask out
            padding_mask = ~padding_mask
            # Combine: mask if causal mask OR padding mask
            causal_mask = causal_mask | padding_mask

        # Embed tokens
        x = self.embedding(input_ids, training=training)

        # Track hidden states if requested
        hidden_states = [x] if return_hidden_states else None

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=causal_mask, training=training)
            if return_hidden_states:
                hidden_states.append(x)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.output_projection(x)

        if return_hidden_states:
            return logits, hidden_states
        return logits

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_transformer_model(
    vocab_size: int,
    max_seq_length: int,
    num_layers: int = 6,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """
    Build a GPT-style causal Transformer model.

    Args:
        vocab_size: Size of token vocabulary
        max_seq_length: Maximum sequence length
        num_layers: Number of transformer blocks
        d_model: Model dimension / embedding size
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras Model (uncompiled, call .compile() after)
    """
    return MusicTransformer(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
    )


def build_transformer_from_config(
    config: TransformerTrainingConfig,
    vocab_size: int,
) -> keras.Model:
    """
    Build transformer model from configuration.

    Args:
        config: TransformerTrainingConfig instance
        vocab_size: Size of token vocabulary (from EventVocabulary.vocab_size)

    Returns:
        Keras Model instance
    """
    return build_transformer_model(
        vocab_size=vocab_size,
        max_seq_length=config.max_seq_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout_rate=config.dropout_rate,
    )
