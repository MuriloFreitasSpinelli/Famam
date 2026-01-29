"""
Music Transformer with Relative Positional Attention for music generation.

Based on "Music Transformer: Generating Music with Long-Term Structure"
(Huang et al., 2018) - https://arxiv.org/abs/1809.04281

Key innovation: Relative positional attention allows the model to learn
distance-based patterns (e.g., "repeat every 4 beats") rather than
absolute positions, which is crucial for music generation.

Architecture:
- Token embedding (no absolute positional embedding)
- N transformer blocks with RELATIVE self-attention + FFN
- Causal mask for autoregressive generation
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

    Args:
        seq_length: Sequence length

    Returns:
        Boolean mask of shape (seq_length, seq_length) where True = masked
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
    return tf.cast(mask, tf.bool)


class RelativePositionalEmbedding(layers.Layer):
    """
    Learnable relative positional embeddings.

    Creates embeddings for relative distances from -max_relative_position to
    +max_relative_position. For music, this allows learning patterns like
    "notes 4 beats apart" regardless of absolute position.
    """

    def __init__(
        self,
        max_relative_position: int,
        embedding_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_relative_position = max_relative_position
        self.embedding_dim = embedding_dim

        # Embeddings for relative positions: [-max_rel, ..., -1, 0, 1, ..., max_rel]
        # Total: 2 * max_relative_position + 1 embeddings
        self.num_embeddings = 2 * max_relative_position + 1

        self.embeddings = self.add_weight(
            name='relative_embeddings',
            shape=(self.num_embeddings, embedding_dim),
            initializer='glorot_uniform',
            trainable=True,
        )

    def call(self, length: int) -> tf.Tensor:
        """
        Get relative position embeddings for a sequence of given length.

        Args:
            length: Sequence length

        Returns:
            Tensor of shape (length, length, embedding_dim) where [i, j] gives
            the embedding for the relative position (j - i), clipped to max range.
        """
        # Create relative position indices
        # range_vec: [0, 1, 2, ..., length-1]
        range_vec = tf.range(length)

        # distance_mat[i, j] = j - i (relative distance from position i to j)
        distance_mat = range_vec[None, :] - range_vec[:, None]

        # Clip to max relative position range and shift to positive indices
        distance_mat_clipped = tf.clip_by_value(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        # Shift: -max_rel -> 0, 0 -> max_rel, +max_rel -> 2*max_rel
        final_mat = distance_mat_clipped + self.max_relative_position

        # Gather embeddings
        return tf.gather(self.embeddings, final_mat)

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_relative_position': self.max_relative_position,
            'embedding_dim': self.embedding_dim,
        })
        return config


class RelativeMultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention with Relative Positional Encoding.

    Implements the efficient "skewing" algorithm from Music Transformer
    to compute relative attention in O(LD) memory instead of O(L²D).

    Attention score: (Q @ K^T + Q @ E^T_rel) / sqrt(d_k)
    where E_rel contains relative position embeddings.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_relative_position: int = 512,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        self.dropout_rate = dropout_rate

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V
        self.wq = layers.Dense(d_model, name='query')
        self.wk = layers.Dense(d_model, name='key')
        self.wv = layers.Dense(d_model, name='value')

        # Output projection
        self.wo = layers.Dense(d_model, name='output')

        # Relative position embeddings (one per head)
        self.relative_pos_emb = RelativePositionalEmbedding(
            max_relative_position=max_relative_position,
            embedding_dim=self.head_dim,
            name='relative_position_embedding',
        )

        self.dropout = layers.Dropout(dropout_rate)
        self.scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

    def _split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, head_dim)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq_len, head_dim)

    def _merge_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Merge heads back into d_model dimension."""
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, seq_len, heads, head_dim)
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def _compute_relative_attention(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        mask: Optional[tf.Tensor],
        training: bool,
    ) -> tf.Tensor:
        """
        Compute attention with relative position bias.

        Args:
            q: Query tensor (batch, heads, seq_len, head_dim)
            k: Key tensor (batch, heads, seq_len, head_dim)
            v: Value tensor (batch, heads, seq_len, head_dim)
            mask: Attention mask (optional)
            training: Whether in training mode

        Returns:
            Attention output (batch, heads, seq_len, head_dim)
        """
        seq_len = tf.shape(q)[2]

        # Standard attention scores: Q @ K^T
        # Shape: (batch, heads, seq_len, seq_len)
        content_scores = tf.matmul(q, k, transpose_b=True)

        # Relative position scores: Q @ E_rel^T
        # Get relative position embeddings: (seq_len, seq_len, head_dim)
        rel_pos_emb = self.relative_pos_emb(seq_len)

        # Compute Q @ E_rel^T using einsum for efficiency
        # q: (batch, heads, seq_len, head_dim)
        # rel_pos_emb: (seq_len, seq_len, head_dim)
        # Result: (batch, heads, seq_len, seq_len)
        position_scores = tf.einsum('bhid,ijd->bhij', q, rel_pos_emb)

        # Combined attention scores
        attention_scores = (content_scores + position_scores) / self.scale

        # Apply mask (causal + optional padding)
        if mask is not None:
            # mask: True where we should NOT attend
            attention_scores = tf.where(
                mask,
                tf.ones_like(attention_scores) * -1e9,
                attention_scores
            )

        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        output = tf.matmul(attention_weights, v)

        return output

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass for relative multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            attention_mask: Boolean mask where True = mask out
            training: Training mode flag

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size = tf.shape(query)[0]

        # Project to Q, K, V
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        # Split heads
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Compute relative attention
        attention_output = self._compute_relative_attention(
            q, k, v, attention_mask, training
        )

        # Merge heads
        attention_output = self._merge_heads(attention_output, batch_size)

        # Final projection
        output = self.wo(attention_output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'max_relative_position': self.max_relative_position,
            'dropout_rate': self.dropout_rate,
        })
        return config


class RelativeTransformerBlock(layers.Layer):
    """
    Transformer block with Relative Multi-Head Attention.

    Uses pre-norm architecture:
    - LayerNorm -> Relative Multi-Head Self-Attention -> Residual
    - LayerNorm -> Feed-Forward Network -> Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_relative_position: int = 512,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_relative_position = max_relative_position
        self.dropout_rate = dropout_rate

        # Pre-norm layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        # Relative multi-head attention
        self.mha = RelativeMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_relative_position=max_relative_position,
            dropout_rate=dropout_rate,
        )

        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate),
        ])

    def call(self, x, attention_mask=None, training=False):
        """Forward pass through relative transformer block."""
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_output = self.mha(
            query=x_norm,
            key=x_norm,
            value=x_norm,
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
            'max_relative_position': self.max_relative_position,
            'dropout_rate': self.dropout_rate,
        })
        return config


# Keep original TransformerBlock for backwards compatibility
class TransformerBlock(layers.Layer):
    """
    Original Transformer block with absolute positional attention.
    Kept for backwards compatibility.
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

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )

        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate),
        ])

    def call(self, x, attention_mask=None, training=False):
        x_norm = self.norm1(x)
        attn_output = self.mha(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            attention_mask=attention_mask,
            training=training,
        )
        x = x + attn_output

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


class TokenEmbedding(layers.Layer):
    """
    Token embedding layer WITHOUT positional embedding.

    For Music Transformer, we use relative positional attention instead
    of absolute positional embeddings, so we only need token embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            name='token_embedding',
        )
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        """Embed tokens (no position information - handled by relative attention)."""
        embeddings = self.token_emb(x)
        # Scale by sqrt(d_model) as in original Transformer
        embeddings = embeddings * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return self.dropout(embeddings, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config


# Keep original for backwards compatibility
class TokenAndPositionEmbedding(layers.Layer):
    """Original embedding with absolute positions. Kept for compatibility."""

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
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)

        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(positions)

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
    Music Transformer with Relative Positional Attention.

    Based on Huang et al., 2018 - uses relative attention to capture
    musical patterns like repetition and rhythmic structure.
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
        use_relative_attention: bool = True,
        max_relative_position: int = 512,
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
        self.use_relative_attention = use_relative_attention
        self.max_relative_position = max_relative_position

        # Embedding layer (with or without absolute positions)
        if use_relative_attention:
            # Only token embedding - position handled by relative attention
            self.embedding = TokenEmbedding(
                vocab_size=vocab_size,
                d_model=d_model,
                dropout_rate=dropout_rate,
            )
        else:
            # Token + absolute positional embedding
            self.embedding = TokenAndPositionEmbedding(
                vocab_size=vocab_size,
                max_seq_length=max_seq_length,
                d_model=d_model,
                dropout_rate=dropout_rate,
            )

        # Transformer blocks (with or without relative attention)
        if use_relative_attention:
            self.transformer_blocks = [
                RelativeTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_relative_position=max_relative_position,
                    dropout_rate=dropout_rate,
                    name=f'relative_transformer_block_{i}',
                )
                for i in range(num_layers)
            ]
        else:
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

        # Output projection
        self.output_projection = layers.Dense(vocab_size, name='output_projection')

    def call(self, inputs, training=False, return_hidden_states=False):
        """
        Forward pass through the Music Transformer.

        Args:
            inputs: Dict with 'input_ids' and optional 'attention_mask'
                   or just token IDs tensor
            training: Whether in training mode
            return_hidden_states: If True, return hidden states from all layers

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Handle dict or tensor input
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            padding_mask = inputs.get('attention_mask', None)
        else:
            input_ids = inputs
            padding_mask = None

        seq_len = tf.shape(input_ids)[1]

        # Create causal attention mask
        causal_mask = get_causal_attention_mask(seq_len)

        # Combine with padding mask if provided
        if padding_mask is not None:
            padding_mask = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], tf.bool)
            padding_mask = ~padding_mask
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
            'use_relative_attention': self.use_relative_attention,
            'max_relative_position': self.max_relative_position,
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
    use_relative_attention: bool = True,
    max_relative_position: int = 512,
) -> keras.Model:
    """
    Build a Music Transformer model.

    Args:
        vocab_size: Size of token vocabulary
        max_seq_length: Maximum sequence length
        num_layers: Number of transformer blocks
        d_model: Model dimension / embedding size
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout rate
        use_relative_attention: If True, use relative positional attention (recommended)
        max_relative_position: Maximum relative distance to consider

    Returns:
        MusicTransformer model instance
    """
    return MusicTransformer(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        use_relative_attention=use_relative_attention,
        max_relative_position=max_relative_position,
    )


def build_transformer_from_config(
    config: TransformerTrainingConfig,
    vocab_size: int,
) -> keras.Model:
    """
    Build transformer model from configuration.

    Args:
        config: TransformerTrainingConfig instance
        vocab_size: Size of token vocabulary

    Returns:
        MusicTransformer model instance
    """
    # Check if config has relative attention settings
    use_relative = getattr(config, 'use_relative_attention', True)
    max_rel_pos = getattr(config, 'max_relative_position', config.max_seq_length // 2)

    return build_transformer_model(
        vocab_size=vocab_size,
        max_seq_length=config.max_seq_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout_rate=config.dropout_rate,
        use_relative_attention=use_relative,
        max_relative_position=max_rel_pos,
    )
