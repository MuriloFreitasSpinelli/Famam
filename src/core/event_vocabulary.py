"""
Event vocabulary for autoregressive Transformer music generation.

Extends muspy's event representation with special tokens for conditioning
(genre, instrument) and sequence control (PAD, BOS, EOS).

Token ranges:
    0-127:    Note-on (pitch)
    128-255:  Note-off (pitch)
    256-355:  Time-shift (1-100 ticks)
    356-387:  Velocity (32 bins, optional)
    388:      PAD
    389:      BOS (begin sequence)
    390:      EOS (end sequence)
    391+:     Genre tokens (one per genre)
    391+G:    Instrument tokens (one per instrument)
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np  # type: ignore


@dataclass
class EventVocabulary:
    """
    Token vocabulary for event-based music representation.

    Combines muspy's event tokens with special tokens for:
    - Sequence control: PAD, BOS, EOS
    - Conditioning: Genre, Instrument
    """

    # Event token offsets (matching muspy's default)
    NOTE_ON_OFFSET: int = 0
    NOTE_OFF_OFFSET: int = 128
    TIME_SHIFT_OFFSET: int = 256
    MAX_TIME_SHIFT: int = 100  # Maximum time shift in ticks

    # Velocity bins (optional, 32 bins for 0-127 velocity range)
    VELOCITY_OFFSET: int = 356
    NUM_VELOCITY_BINS: int = 32

    # Special tokens
    PAD_TOKEN: int = 388
    BOS_TOKEN: int = 389
    EOS_TOKEN: int = 390

    # Conditioning token offsets (set during initialization)
    GENRE_OFFSET: int = 391

    # Number of genres and instruments
    num_genres: int = 0
    num_instruments: int = 129  # 0-127 + drums

    def __post_init__(self):
        """Calculate instrument offset after genre tokens."""
        self.INSTRUMENT_OFFSET = self.GENRE_OFFSET + self.num_genres

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including all token types."""
        return self.INSTRUMENT_OFFSET + self.num_instruments

    @property
    def base_event_vocab_size(self) -> int:
        """Size of base muspy event vocabulary (before special tokens)."""
        return self.PAD_TOKEN

    # === Encoding methods ===

    def encode_note_on(self, pitch: int) -> int:
        """Encode a note-on event."""
        assert 0 <= pitch <= 127, f"Pitch must be 0-127, got {pitch}"
        return self.NOTE_ON_OFFSET + pitch

    def encode_note_off(self, pitch: int) -> int:
        """Encode a note-off event."""
        assert 0 <= pitch <= 127, f"Pitch must be 0-127, got {pitch}"
        return self.NOTE_OFF_OFFSET + pitch

    def encode_time_shift(self, ticks: int) -> int:
        """
        Encode a time shift event.

        Time shifts are 1-indexed (1 = shift by 1 tick).
        Shifts larger than MAX_TIME_SHIFT are clamped.
        """
        ticks = max(1, min(ticks, self.MAX_TIME_SHIFT))
        return self.TIME_SHIFT_OFFSET + ticks - 1  # 1-indexed to 0-indexed

    def encode_velocity(self, velocity: int) -> int:
        """
        Encode a velocity value into a binned token.

        Velocity 0-127 is mapped to NUM_VELOCITY_BINS bins.
        """
        velocity = max(0, min(127, velocity))
        bin_idx = velocity * self.NUM_VELOCITY_BINS // 128
        return self.VELOCITY_OFFSET + bin_idx

    def encode_genre(self, genre_id: int) -> int:
        """Encode a genre ID as a conditioning token."""
        assert 0 <= genre_id < self.num_genres, f"Genre ID {genre_id} out of range"
        return self.GENRE_OFFSET + genre_id

    def encode_instrument(self, instrument_id: int) -> int:
        """Encode an instrument ID as a conditioning token."""
        assert 0 <= instrument_id < self.num_instruments, f"Instrument ID {instrument_id} out of range"
        return self.INSTRUMENT_OFFSET + instrument_id

    # === Decoding methods ===

    def decode_token(self, token: int) -> Tuple[str, int]:
        """
        Decode a token into its type and value.

        Returns:
            Tuple of (token_type, value) where token_type is one of:
            'note_on', 'note_off', 'time_shift', 'velocity',
            'pad', 'bos', 'eos', 'genre', 'instrument'
        """
        if token == self.PAD_TOKEN:
            return ('pad', 0)
        elif token == self.BOS_TOKEN:
            return ('bos', 0)
        elif token == self.EOS_TOKEN:
            return ('eos', 0)
        elif token >= self.INSTRUMENT_OFFSET:
            return ('instrument', token - self.INSTRUMENT_OFFSET)
        elif token >= self.GENRE_OFFSET:
            return ('genre', token - self.GENRE_OFFSET)
        elif token >= self.VELOCITY_OFFSET:
            bin_idx = token - self.VELOCITY_OFFSET
            velocity = bin_idx * 128 // self.NUM_VELOCITY_BINS + 64 // self.NUM_VELOCITY_BINS
            return ('velocity', velocity)
        elif token >= self.TIME_SHIFT_OFFSET:
            return ('time_shift', token - self.TIME_SHIFT_OFFSET + 1)  # 0-indexed to 1-indexed
        elif token >= self.NOTE_OFF_OFFSET:
            return ('note_off', token - self.NOTE_OFF_OFFSET)
        else:
            return ('note_on', token - self.NOTE_ON_OFFSET)

    def is_special_token(self, token: int) -> bool:
        """Check if token is a special token (PAD, BOS, EOS)."""
        return token in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN)

    def is_conditioning_token(self, token: int) -> bool:
        """Check if token is a conditioning token (genre or instrument)."""
        return token >= self.GENRE_OFFSET

    def is_event_token(self, token: int) -> bool:
        """Check if token is a music event token (note, time, velocity)."""
        return token < self.PAD_TOKEN

    # === Sequence encoding ===

    def encode_events_to_sequence(
        self,
        events: List[Tuple[str, int]],
        genre_id: int,
        instrument_id: int,
        max_length: int,
        encode_velocity: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a list of events to a padded token sequence.

        Args:
            events: List of (event_type, value) tuples from muspy
            genre_id: Genre ID for conditioning
            instrument_id: Instrument ID for conditioning
            max_length: Maximum sequence length (including special tokens)
            encode_velocity: Whether to include velocity tokens

        Returns:
            Tuple of (input_ids, attention_mask) arrays
        """
        tokens = [self.BOS_TOKEN, self.encode_genre(genre_id), self.encode_instrument(instrument_id)]

        for event_type, value in events:
            if event_type == 'note_on':
                tokens.append(self.encode_note_on(value))
            elif event_type == 'note_off':
                tokens.append(self.encode_note_off(value))
            elif event_type == 'time_shift':
                # Handle large time shifts by splitting into multiple tokens
                while value > self.MAX_TIME_SHIFT:
                    tokens.append(self.encode_time_shift(self.MAX_TIME_SHIFT))
                    value -= self.MAX_TIME_SHIFT
                if value > 0:
                    tokens.append(self.encode_time_shift(value))
            elif event_type == 'velocity' and encode_velocity:
                tokens.append(self.encode_velocity(value))

            # Stop if we're running out of space
            if len(tokens) >= max_length - 1:
                break

        tokens.append(self.EOS_TOKEN)

        # Pad or truncate to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.EOS_TOKEN]

        attention_mask = [1] * len(tokens)

        while len(tokens) < max_length:
            tokens.append(self.PAD_TOKEN)
            attention_mask.append(0)

        return np.array(tokens, dtype=np.int32), np.array(attention_mask, dtype=np.int32)

    def decode_sequence_to_events(
        self,
        tokens: np.ndarray,
        skip_special: bool = True,
    ) -> List[Tuple[str, int]]:
        """
        Decode a token sequence back to events.

        Args:
            tokens: Array of token IDs
            skip_special: If True, skip PAD/BOS/EOS and conditioning tokens

        Returns:
            List of (event_type, value) tuples
        """
        events = []

        for token in tokens:
            token = int(token)
            event_type, value = self.decode_token(token)

            if skip_special and event_type in ('pad', 'bos', 'eos', 'genre', 'instrument'):
                if event_type == 'eos':
                    break  # Stop at EOS
                continue

            events.append((event_type, value))

        return events

    # === Utility methods ===

    def create_start_sequence(self, genre_id: int, instrument_id: int) -> np.ndarray:
        """Create starting sequence for autoregressive generation."""
        return np.array([
            self.BOS_TOKEN,
            self.encode_genre(genre_id),
            self.encode_instrument(instrument_id),
        ], dtype=np.int32)

    def get_token_info(self) -> Dict[str, int]:
        """Get dictionary of token type boundaries for debugging."""
        return {
            'note_on_start': self.NOTE_ON_OFFSET,
            'note_on_end': self.NOTE_OFF_OFFSET - 1,
            'note_off_start': self.NOTE_OFF_OFFSET,
            'note_off_end': self.TIME_SHIFT_OFFSET - 1,
            'time_shift_start': self.TIME_SHIFT_OFFSET,
            'time_shift_end': self.VELOCITY_OFFSET - 1,
            'velocity_start': self.VELOCITY_OFFSET,
            'velocity_end': self.PAD_TOKEN - 1,
            'pad': self.PAD_TOKEN,
            'bos': self.BOS_TOKEN,
            'eos': self.EOS_TOKEN,
            'genre_start': self.GENRE_OFFSET,
            'genre_end': self.INSTRUMENT_OFFSET - 1,
            'instrument_start': self.INSTRUMENT_OFFSET,
            'instrument_end': self.vocab_size - 1,
            'vocab_size': self.vocab_size,
        }

    def __repr__(self) -> str:
        return (
            f"EventVocabulary(num_genres={self.num_genres}, "
            f"num_instruments={self.num_instruments}, "
            f"vocab_size={self.vocab_size})"
        )
