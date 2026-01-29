"""
Abstract base class for music encoders.

Encoders are responsible for converting between music representations
(e.g., muspy.Track) and token sequences suitable for model training.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EncodedSequence:
    """
    Container for an encoded music sequence.

    Attributes:
        token_ids: Array of token IDs
        attention_mask: Array indicating valid tokens (1) vs padding (0)
        metadata: Optional dictionary for encoder-specific metadata
    """
    token_ids: np.ndarray
    attention_mask: np.ndarray
    metadata: Optional[dict] = None


class BaseEncoder(ABC):
    """
    Abstract base class for music encoders.

    Subclasses must implement:
        - encode_track(): Convert a track to token sequence
        - decode_tokens(): Convert tokens back to events/music
        - vocab_size: Total vocabulary size
        - special_tokens: Dictionary of special token IDs
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size including all token types."""
        pass

    @property
    @abstractmethod
    def special_tokens(self) -> dict:
        """
        Dictionary of special token IDs.

        Expected keys: 'pad', 'bos', 'eos'
        Additional keys may be added by subclasses.
        """
        pass

    @property
    def pad_token_id(self) -> int:
        """PAD token ID."""
        return self.special_tokens['pad']

    @property
    def bos_token_id(self) -> int:
        """BOS (begin of sequence) token ID."""
        return self.special_tokens['bos']

    @property
    def eos_token_id(self) -> int:
        """EOS (end of sequence) token ID."""
        return self.special_tokens['eos']

    @abstractmethod
    def encode_track(
        self,
        track: Any,
        genre_id: int,
        instrument_id: int,
        max_length: int,
        **kwargs,
    ) -> EncodedSequence:
        """
        Encode a music track to a token sequence.

        Args:
            track: Music track object (e.g., muspy.Track)
            genre_id: Genre conditioning ID
            instrument_id: Instrument conditioning ID
            max_length: Maximum sequence length (including special tokens)
            **kwargs: Encoder-specific options

        Returns:
            EncodedSequence with token_ids and attention_mask
        """
        pass

    @abstractmethod
    def decode_tokens(
        self,
        tokens: np.ndarray,
        skip_special: bool = True,
    ) -> List[Tuple[str, int]]:
        """
        Decode a token sequence back to events.

        Args:
            tokens: Array of token IDs
            skip_special: If True, skip special/conditioning tokens

        Returns:
            List of (event_type, value) tuples
        """
        pass

    @abstractmethod
    def create_conditioning_tokens(
        self,
        genre_id: int,
        instrument_id: int,
    ) -> np.ndarray:
        """
        Create the conditioning token sequence for generation.

        Args:
            genre_id: Genre conditioning ID
            instrument_id: Instrument conditioning ID

        Returns:
            Array of conditioning tokens (typically [BOS, genre, instrument])
        """
        pass

    def create_labels(
        self,
        token_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Create labels for next-token prediction training.

        Default implementation: shift input by 1, pad at end.

        Args:
            token_ids: Input token IDs

        Returns:
            Labels array (same shape as input)
        """
        labels = np.concatenate([token_ids[1:], [self.pad_token_id]])
        return labels.astype(np.int32)

    def pad_sequence(
        self,
        tokens: List[int],
        max_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad or truncate a token sequence to fixed length.

        Args:
            tokens: List of token IDs
            max_length: Target length

        Returns:
            Tuple of (padded_tokens, attention_mask) arrays
        """
        if len(tokens) > max_length:
            # Truncate but preserve EOS at end
            tokens = tokens[:max_length - 1] + [self.eos_token_id]

        attention_mask = [1] * len(tokens)

        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
            attention_mask.append(0)

        return np.array(tokens, dtype=np.int32), np.array(attention_mask, dtype=np.int32)

    def is_special_token(self, token: int) -> bool:
        """Check if token is a special token (PAD, BOS, EOS)."""
        return token in (self.pad_token_id, self.bos_token_id, self.eos_token_id)

    @abstractmethod
    def get_state(self) -> dict:
        """
        Get encoder state for serialization.

        Returns:
            Dictionary containing all parameters needed to reconstruct encoder
        """
        pass

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> "BaseEncoder":
        """
        Reconstruct encoder from serialized state.

        Args:
            state: Dictionary from get_state()

        Returns:
            New encoder instance
        """
        pass
