"""
Music Transformer with Relative Positional Attention.

Based on "Music Transformer: Generating Music with Long-Term Structure"
(Huang et al., 2018) - https://arxiv.org/abs/1809.04281

Uses relative positional attention to learn distance-based patterns
(e.g., "repeat every 4 beats") rather than absolute positions.

Author: Murilo de Freitas Spinelli
"""

from typing import Optional, Dict, Any, List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base_model import BaseMusicModel, get_causal_attention_mask
from ..config import TrainingConfig


class RelativePositionalEmbedding(layers.Layer):
    """
    Learnable relative positional embeddings.

    Creates embeddings for relative distances from -max_relative_position
    to +max_relative_position.
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
        self.num_embeddings = 2 * max_relative_position + 1

        self.embeddings = self.add_weight(
            name='relative_embeddings',
            shape=(self.num_embeddings, embedding_dim),
            initializer='glorot_uniform',
            trainable=True,
        )

    def call(self, length: int) -> tf.Tensor:
        """
        Get relative position embeddings for a sequence.

        Args:
            length: Sequence length

        Returns:
            Tensor of shape (length, length, embedding_dim)
        """
        range_vec = tf.range(length)
        distance_mat = range_vec[None, :] - range_vec[:, None]

        distance_mat_clipped = tf.clip_by_value(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position

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

    Attention score: (Q @ K^T + Q @ E_rel^T) / sqrt(d_k)
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
        self.head_dim = d_model // num_heads

        self.wq = layers.Dense(d_model, name='query')
        self.wk = layers.Dense(d_model, name='key')
        self.wv = layers.Dense(d_model, name='value')
        self.wo = layers.Dense(d_model, name='output')

        self.relative_pos_emb = RelativePositionalEmbedding(
            max_relative_position=max_relative_position,
            embedding_dim=self.head_dim,
        )

        self.dropout = layers.Dropout(dropout_rate)
        self.scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

    def _split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _merge_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
        cache: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tf.Tensor:
        batch_size = tf.shape(query)[0]

        q = self._split_heads(self.wq(query), batch_size)
        k = self._split_heads(self.wk(key), batch_size)
        v = self._split_heads(self.wv(value), batch_size)

        if cache is not None:
            k = tf.concat([cache[0], k], axis=2)
            v = tf.concat([cache[1], v], axis=2)

        # Content-based attention
        content_scores = tf.matmul(q, k, transpose_b=True)

        # Relative position scores
        if cache is not None:
            # Cached: query is at the last position, keys span full history
            seq_len_k = tf.shape(k)[2]
            query_pos = seq_len_k - 1
            key_positions = tf.range(seq_len_k)
            distances = key_positions - query_pos
            distances_clipped = tf.clip_by_value(
                distances,
                -self.max_relative_position,
                self.max_relative_position,
            )
            indices = distances_clipped + self.max_relative_position
            rel_pos_emb = tf.gather(self.relative_pos_emb.embeddings, indices)
            rel_pos_emb = rel_pos_emb[tf.newaxis, :, :]  # (1, key_len, head_dim)
        else:
            seq_len = tf.shape(query)[1]
            rel_pos_emb = self.relative_pos_emb(seq_len)

        position_scores = tf.einsum('bhid,ijd->bhij', q, rel_pos_emb)

        attention_scores = (content_scores + position_scores) / self.scale

        if attention_mask is not None:
            attention_scores = tf.where(
                attention_mask,
                tf.ones_like(attention_scores) * -1e9,
                attention_scores
            )

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        output = tf.matmul(attention_weights, v)
        output = self._merge_heads(output, batch_size)
        output = self.wo(output)

        if use_cache:
            return output, (k, v)
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


class TransformerBlock(layers.Layer):
    """
    Transformer block with optional relative attention.

    Uses pre-norm architecture:
        LayerNorm -> Self-Attention -> Residual
        LayerNorm -> FFN -> Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        use_relative_attention: bool = True,
        max_relative_position: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.use_relative_attention = use_relative_attention
        self.max_relative_position = max_relative_position

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        if use_relative_attention:
            self.mha = RelativeMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                max_relative_position=max_relative_position,
                dropout_rate=dropout_rate,
            )
        else:
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

    def call(self, x, attention_mask=None, training=False, cache=None, use_cache=False):
        # Self-attention
        x_norm = self.norm1(x)
        if self.use_relative_attention:
            attn_result = self.mha(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                attention_mask=attention_mask,
                training=training,
                cache=cache,
                use_cache=use_cache,
            )
            if use_cache:
                attn_output, new_cache = attn_result
            else:
                attn_output = attn_result
                new_cache = None
        else:
            attn_output = self.mha(
                query=x_norm,
                value=x_norm,
                key=x_norm,
                attention_mask=attention_mask,
                training=training,
            )
            new_cache = None
        x = x + attn_output

        # FFN
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm, training=training)
        x = x + ffn_output

        if use_cache:
            return x, new_cache
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'use_relative_attention': self.use_relative_attention,
            'max_relative_position': self.max_relative_position,
        })
        return config


class TransformerModel(BaseMusicModel):
    """
    Music Transformer for autoregressive music generation.

    Uses relative positional attention to capture musical patterns
    like repetition and rhythmic structure.
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_relative_attention = use_relative_attention
        self.max_relative_position = max_relative_position

        super().__init__(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            d_model=d_model,
            dropout_rate=dropout_rate,
            **kwargs
        )

    def build_layers(self) -> None:
        """Build transformer layers."""
        self.embedding_dropout = layers.Dropout(self.dropout_rate)

        self.transformer_blocks = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                use_relative_attention=self.use_relative_attention,
                max_relative_position=self.max_relative_position,
                name=f'transformer_block_{i}',
            )
            for i in range(self.num_layers)
        ]

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
            padding_mask = inputs.get('attention_mask', None)
        else:
            input_ids = inputs
            padding_mask = None

        seq_len = tf.shape(input_ids)[1]

        # Causal mask
        causal_mask = get_causal_attention_mask(seq_len)

        # Combine with padding mask
        if padding_mask is not None:
            padding_mask = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], tf.bool)
            padding_mask = ~padding_mask
            causal_mask = causal_mask | padding_mask

        # Embed tokens
        x = self.embed_tokens(input_ids, training=training)
        x = self.embedding_dropout(x, training=training)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=causal_mask, training=training)

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits

    def generate_step(
        self,
        input_ids: tf.Tensor,
        past_caches: Optional[List] = None,
    ) -> Tuple[tf.Tensor, List]:
        """
        Single-step forward pass with KV-cache for fast generation.

        Args:
            input_ids: Token IDs - full prompt on first call, single token after.
            past_caches: List of (K, V) tuples per layer, or None for first call.

        Returns:
            (logits, new_caches) tuple.
        """
        # Causal mask only needed for the initial multi-token prompt pass
        if past_caches is None:
            seq_len = tf.shape(input_ids)[1]
            causal_mask = get_causal_attention_mask(seq_len)
        else:
            causal_mask = None

        x = self.embed_tokens(input_ids, training=False)

        new_caches = []
        for i, block in enumerate(self.transformer_blocks):
            cache = past_caches[i] if past_caches is not None else None
            x, new_cache = block(
                x,
                attention_mask=causal_mask,
                training=False,
                cache=cache,
                use_cache=True,
            )
            new_caches.append(new_cache)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits, new_caches

    def get_config(self) -> Dict[str, Any]:
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

    def summary_str(self) -> str:
        lines = [
            super().summary_str(),
            f"  Layers: {self.num_layers}",
            f"  Attention Heads: {self.num_heads}",
            f"  FFN Dimension: {self.d_ff}",
            f"  Relative Attention: {self.use_relative_attention}",
            f"  Max Relative Position: {self.max_relative_position}",
        ]
        return "\n".join(lines)


def build_transformer_from_config(
    config: TrainingConfig,
    vocab_size: int,
) -> TransformerModel:
    """
    Build transformer model from configuration.

    Args:
        config: TrainingConfig instance
        vocab_size: Size of token vocabulary

    Returns:
        TransformerModel instance
    """
    return TransformerModel(
        vocab_size=vocab_size,
        max_seq_length=config.max_seq_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout_rate=config.dropout_rate,
        use_relative_attention=config.use_relative_attention,
        max_relative_position=config.max_relative_position,
    )
