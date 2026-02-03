"""
Abstract base class for music generation models.

Defines the common interface for Transformer and LSTM models
that work with event-based token sequences.

Author: Murilo de Freitas Spinelli
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple

import tensorflow as tf
from tensorflow import keras

from ..config import TrainingConfig


class BaseMusicModel(ABC, keras.Model):
    """
    Abstract base class for autoregressive music generation models.

    All models work with token sequences and predict the next token
    in an autoregressive fashion.

    Subclasses must implement:
        - build_layers(): Construct model layers
        - call(): Forward pass
        - get_config(): Return model configuration
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        d_model: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize base model.

        Args:
            vocab_size: Size of token vocabulary
            max_seq_length: Maximum sequence length
            d_model: Model/embedding dimension
            dropout_rate: Dropout rate
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # Token embedding (shared by all model types)
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            name='token_embedding',
        )

        # Output projection to vocabulary
        self.output_projection = keras.layers.Dense(
            vocab_size,
            name='output_projection',
        )

        # Build model-specific layers
        self.build_layers()

    @abstractmethod
    def build_layers(self) -> None:
        """
        Build model-specific layers.

        Subclasses should create their architecture components here.
        Called automatically during __init__.
        """
        pass

    @abstractmethod
    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass through the model.

        Args:
            inputs: Dict with 'input_ids' and optionally 'attention_mask'
            training: Whether in training mode

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        pass

    def embed_tokens(self, input_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Embed input tokens.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            training: Whether in training mode

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        embeddings = self.token_embedding(input_ids)
        # Scale by sqrt(d_model) as in original Transformer
        embeddings = embeddings * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return embeddings

    def generate(
        self,
        start_tokens: tf.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            start_tokens: Initial token sequence (batch, start_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling threshold
            eos_token_id: Stop generation when this token is produced
            min_length: Minimum length before allowing EOS to stop generation

        Returns:
            Generated token sequence (batch, generated_len)
        """
        batch_size = tf.shape(start_tokens)[0]
        generated = start_tokens
        min_length = min_length or 0

        for step in range(max_length - tf.shape(start_tokens)[1]):
            # Get predictions for next token
            inputs = {'input_ids': generated}
            logits = self(inputs, training=False)

            # Get logits for last position only
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = tf.math.top_k(next_logits, k=top_k)
                # Set all non-top-k logits to -inf
                indices_to_remove = tf.less(
                    next_logits,
                    tf.reduce_min(top_k_logits, axis=-1, keepdims=True)
                )
                next_logits = tf.where(
                    indices_to_remove,
                    tf.ones_like(next_logits) * -1e9,
                    next_logits
                )

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits = tf.sort(next_logits, direction='DESCENDING', axis=-1)
                sorted_probs = tf.nn.softmax(sorted_logits, axis=-1)
                cumulative_probs = tf.cumsum(sorted_probs, axis=-1)

                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove = tf.concat([
                    tf.zeros_like(sorted_indices_to_remove[:, :1]),
                    sorted_indices_to_remove[:, :-1]
                ], axis=-1)

                # Scatter back to original ordering
                indices = tf.argsort(tf.argsort(next_logits, direction='DESCENDING'))
                indices_to_remove = tf.gather(sorted_indices_to_remove, indices, batch_dims=1)
                next_logits = tf.where(
                    indices_to_remove,
                    tf.ones_like(next_logits) * -1e9,
                    next_logits
                )

            # Sample from distribution
            next_token = tf.random.categorical(next_logits, num_samples=1)
            next_token = tf.cast(next_token, tf.int32)

            # Append to sequence
            generated = tf.concat([generated, next_token], axis=1)

            # Check for EOS (only after min_length)
            current_length = tf.shape(generated)[1]
            if eos_token_id is not None and current_length >= min_length:
                if tf.reduce_all(next_token == eos_token_id):
                    break

        return generated

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseMusicModel':
        """Create model from configuration dict."""
        return cls(**config)

    def summary_str(self) -> str:
        """Get a string summary of the model architecture."""
        lines = [
            f"Model: {self.__class__.__name__}",
            f"  Vocab Size: {self.vocab_size}",
            f"  Max Sequence Length: {self.max_seq_length}",
            f"  Model Dimension: {self.d_model}",
            f"  Dropout Rate: {self.dropout_rate}",
        ]
        return "\n".join(lines)


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
