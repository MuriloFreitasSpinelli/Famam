"""
Event-based encoder for autoregressive music generation.

Works with both Transformer and LSTM models. Converts music tracks
to sequences of discrete events:
    - Note-on (pitch 0-127)
    - Note-off (pitch 0-127)
    - Time-shift (1-100 ticks)
    - Velocity (32 bins, optional)

Plus special tokens for conditioning (genre, instrument) and
sequence control (PAD, BOS, EOS).

Author: Murilo de Freitas Spinelli
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from .base_encoder import BaseEncoder, EncodedSequence


@dataclass
class EventVocabulary:
    """
    Token vocabulary for event-based music representation.

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

    # Event token offsets
    NOTE_ON_OFFSET: int = 0
    NOTE_OFF_OFFSET: int = 128
    TIME_SHIFT_OFFSET: int = 256
    MAX_TIME_SHIFT: int = 100

    # Velocity bins
    VELOCITY_OFFSET: int = 356
    NUM_VELOCITY_BINS: int = 32

    # Special tokens
    PAD_TOKEN: int = 388
    BOS_TOKEN: int = 389
    EOS_TOKEN: int = 390

    # Conditioning token offsets
    GENRE_OFFSET: int = 391

    # Configurable counts
    num_genres: int = 0
    num_instruments: int = 129  # 0-127 + drums (128)

    # Computed after init
    INSTRUMENT_OFFSET: int = field(init=False)

    def __post_init__(self):
        """Calculate instrument offset after genre tokens."""
        self.INSTRUMENT_OFFSET = self.GENRE_OFFSET + self.num_genres

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including all token types."""
        return self.INSTRUMENT_OFFSET + self.num_instruments

    def encode_note_on(self, pitch: int) -> int:
        """Encode a note-on event (pitch 0-127)."""
        pitch = max(0, min(127, pitch))
        return self.NOTE_ON_OFFSET + pitch

    def encode_note_off(self, pitch: int) -> int:
        """Encode a note-off event (pitch 0-127)."""
        pitch = max(0, min(127, pitch))
        return self.NOTE_OFF_OFFSET + pitch

    def encode_time_shift(self, ticks: int) -> int:
        """
        Encode a time shift event.

        Time shifts are 1-indexed (1 = shift by 1 tick).
        Shifts larger than MAX_TIME_SHIFT are clamped.
        """
        ticks = max(1, min(ticks, self.MAX_TIME_SHIFT))
        return self.TIME_SHIFT_OFFSET + ticks - 1

    def encode_velocity(self, velocity: int) -> int:
        """Encode a velocity value into a binned token (32 bins)."""
        velocity = max(0, min(127, velocity))
        bin_idx = velocity * self.NUM_VELOCITY_BINS // 128
        return self.VELOCITY_OFFSET + bin_idx

    def encode_genre(self, genre_id: int) -> int:
        """Encode a genre ID as a conditioning token."""
        if not (0 <= genre_id < self.num_genres):
            raise ValueError(f"Genre ID {genre_id} out of range [0, {self.num_genres})")
        return self.GENRE_OFFSET + genre_id

    def encode_instrument(self, instrument_id: int) -> int:
        """Encode an instrument ID as a conditioning token."""
        if not (0 <= instrument_id < self.num_instruments):
            raise ValueError(f"Instrument ID {instrument_id} out of range [0, {self.num_instruments})")
        return self.INSTRUMENT_OFFSET + instrument_id

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
            return ('time_shift', token - self.TIME_SHIFT_OFFSET + 1)
        elif token >= self.NOTE_OFF_OFFSET:
            return ('note_off', token - self.NOTE_OFF_OFFSET)
        else:
            return ('note_on', token - self.NOTE_ON_OFFSET)

    def is_special_token(self, token: int) -> bool:
        """Check if token is PAD, BOS, or EOS."""
        return token in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN)

    def is_conditioning_token(self, token: int) -> bool:
        """Check if token is a genre or instrument token."""
        return token >= self.GENRE_OFFSET

    def is_event_token(self, token: int) -> bool:
        """Check if token is a music event (note, time, velocity)."""
        return token < self.PAD_TOKEN

    def get_token_info(self) -> Dict[str, int]:
        """Get dictionary of token type boundaries for debugging."""
        return {
            'note_on_range': (self.NOTE_ON_OFFSET, self.NOTE_OFF_OFFSET - 1),
            'note_off_range': (self.NOTE_OFF_OFFSET, self.TIME_SHIFT_OFFSET - 1),
            'time_shift_range': (self.TIME_SHIFT_OFFSET, self.VELOCITY_OFFSET - 1),
            'velocity_range': (self.VELOCITY_OFFSET, self.PAD_TOKEN - 1),
            'pad': self.PAD_TOKEN,
            'bos': self.BOS_TOKEN,
            'eos': self.EOS_TOKEN,
            'genre_range': (self.GENRE_OFFSET, self.INSTRUMENT_OFFSET - 1),
            'instrument_range': (self.INSTRUMENT_OFFSET, self.vocab_size - 1),
            'vocab_size': self.vocab_size,
        }


class EventEncoder(BaseEncoder):
    """
    Event-based music encoder using note-on/off and time-shift tokens.

    Converts muspy.Track objects to token sequences for Transformer training.
    """

    def __init__(
        self,
        num_genres: int = 0,
        num_instruments: int = 129,
        encode_velocity: bool = False,
    ):
        """
        Initialize the event encoder.

        Args:
            num_genres: Number of genre conditioning tokens
            num_instruments: Number of instrument tokens (default 129 = 128 MIDI + drums)
            encode_velocity: Whether to include velocity tokens in encoding
        """
        self.vocabulary = EventVocabulary(
            num_genres=num_genres,
            num_instruments=num_instruments,
        )
        self.encode_velocity = encode_velocity

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return self.vocabulary.vocab_size

    @property
    def special_tokens(self) -> dict:
        """Dictionary of special token IDs."""
        return {
            'pad': self.vocabulary.PAD_TOKEN,
            'bos': self.vocabulary.BOS_TOKEN,
            'eos': self.vocabulary.EOS_TOKEN,
        }

    @property
    def max_time_shift(self) -> int:
        """Maximum time shift value per token."""
        return self.vocabulary.MAX_TIME_SHIFT

    def track_to_events(
        self,
        track: Any,
        encode_velocity: Optional[bool] = None,
    ) -> List[Tuple[str, int]]:
        """
        Convert a muspy.Track to event representation.

        Converts notes to a sequence of note-on, note-off, and time-shift events.
        Events are sorted by time, with note-offs before note-ons at the same time.

        Args:
            track: muspy.Track object
            encode_velocity: Override instance setting for velocity encoding

        Returns:
            List of (event_type, value) tuples
        """
        if encode_velocity is None:
            encode_velocity = self.encode_velocity

        if not track.notes:
            return []

        # Build list of all events: (time, event_type, pitch, velocity)
        # event_type: 0 = note_on, 1 = note_off
        all_events = []
        for note in track.notes:
            all_events.append((note.time, 0, note.pitch, note.velocity))
            all_events.append((note.time + note.duration, 1, note.pitch, 0))

        if not all_events:
            return []

        # Sort by time, then note-offs before note-ons at same time
        all_events.sort(key=lambda x: (x[0], x[1]))

        events = []
        current_time = 0

        for time, event_type, pitch, velocity in all_events:
            pitch = max(0, min(127, pitch))

            # Add time shift if needed
            time_diff = time - current_time
            if time_diff > 0:
                events.append(('time_shift', time_diff))
                current_time = time

            if event_type == 0:  # note_on
                if encode_velocity and velocity > 0:
                    events.append(('velocity', velocity))
                events.append(('note_on', pitch))
            else:  # note_off
                events.append(('note_off', pitch))

        return events

    def events_to_tokens(
        self,
        events: List[Tuple[str, int]],
        genre_id: int,
        instrument_id: int,
    ) -> List[int]:
        """
        Convert events to token IDs.

        Handles large time shifts by splitting into multiple MAX_TIME_SHIFT tokens.

        Args:
            events: List of (event_type, value) tuples
            genre_id: Genre conditioning ID
            instrument_id: Instrument conditioning ID

        Returns:
            List of token IDs (without padding)
        """
        vocab = self.vocabulary

        tokens = [
            vocab.BOS_TOKEN,
            vocab.encode_genre(genre_id),
            vocab.encode_instrument(instrument_id),
        ]

        for event_type, value in events:
            if event_type == 'note_on':
                tokens.append(vocab.encode_note_on(value))
            elif event_type == 'note_off':
                tokens.append(vocab.encode_note_off(value))
            elif event_type == 'time_shift':
                # Split large time shifts into multiple tokens
                while value > vocab.MAX_TIME_SHIFT:
                    tokens.append(vocab.encode_time_shift(vocab.MAX_TIME_SHIFT))
                    value -= vocab.MAX_TIME_SHIFT
                if value > 0:
                    tokens.append(vocab.encode_time_shift(value))
            elif event_type == 'velocity':
                tokens.append(vocab.encode_velocity(value))

        tokens.append(vocab.EOS_TOKEN)
        return tokens

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
            track: muspy.Track object
            genre_id: Genre conditioning ID
            instrument_id: Instrument conditioning ID
            max_length: Maximum sequence length (including special tokens)
            **kwargs: Optional 'encode_velocity' to override instance setting

        Returns:
            EncodedSequence with token_ids and attention_mask
        """
        encode_velocity = kwargs.get('encode_velocity', self.encode_velocity)

        # Convert track to events
        events = self.track_to_events(track, encode_velocity=encode_velocity)

        # Convert events to tokens
        tokens = self.events_to_tokens(events, genre_id, instrument_id)

        # Pad/truncate to max_length
        token_ids, attention_mask = self.pad_sequence(tokens, max_length)

        return EncodedSequence(
            token_ids=token_ids,
            attention_mask=attention_mask,
            metadata={
                'genre_id': genre_id,
                'instrument_id': instrument_id,
                'original_length': len(tokens),
            }
        )

    def decode_tokens(
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
        vocab = self.vocabulary

        for token in tokens:
            token = int(token)
            event_type, value = vocab.decode_token(token)

            if skip_special:
                if event_type == 'eos':
                    break
                if event_type in ('pad', 'bos', 'genre', 'instrument'):
                    continue

            events.append((event_type, value))

        return events

    def create_conditioning_tokens(
        self,
        genre_id: int,
        instrument_id: int,
    ) -> np.ndarray:
        """
        Create conditioning token sequence for generation.

        Args:
            genre_id: Genre conditioning ID
            instrument_id: Instrument conditioning ID

        Returns:
            Array of [BOS, genre_token, instrument_token]
        """
        vocab = self.vocabulary
        return np.array([
            vocab.BOS_TOKEN,
            vocab.encode_genre(genre_id),
            vocab.encode_instrument(instrument_id),
        ], dtype=np.int32)

    def decode_to_notes(
        self,
        tokens: np.ndarray,
        ticks_per_beat: int = 24,
    ) -> List[Dict[str, int]]:
        """
        Decode tokens to a list of note dictionaries.

        Reconstructs note timing from time-shift events.

        Args:
            tokens: Array of token IDs
            ticks_per_beat: Resolution for timing (not used in reconstruction,
                           but could be used for tempo-based adjustments)

        Returns:
            List of dicts with keys: pitch, time, duration, velocity
        """
        events = self.decode_tokens(tokens, skip_special=True)

        notes = []
        current_time = 0
        current_velocity = 64  # Default velocity
        active_notes: Dict[int, Tuple[int, int]] = {}  # pitch -> (start_time, velocity)

        for event_type, value in events:
            if event_type == 'time_shift':
                current_time += value
            elif event_type == 'velocity':
                current_velocity = value
            elif event_type == 'note_on':
                active_notes[value] = (current_time, current_velocity)
            elif event_type == 'note_off':
                if value in active_notes:
                    start_time, velocity = active_notes.pop(value)
                    duration = current_time - start_time
                    if duration > 0:
                        notes.append({
                            'pitch': value,
                            'time': start_time,
                            'duration': duration,
                            'velocity': velocity,
                        })

        # Close any remaining active notes
        for pitch, (start_time, velocity) in active_notes.items():
            duration = current_time - start_time
            if duration > 0:
                notes.append({
                    'pitch': pitch,
                    'time': start_time,
                    'duration': duration,
                    'velocity': velocity,
                })

        return sorted(notes, key=lambda n: (n['time'], n['pitch']))

    def get_state(self) -> dict:
        """
        Get encoder state for serialization.

        Returns:
            Dictionary containing all parameters needed to reconstruct encoder
        """
        return {
            'encoder_type': 'event',
            'num_genres': self.vocabulary.num_genres,
            'num_instruments': self.vocabulary.num_instruments,
            'encode_velocity': self.encode_velocity,
        }

    @classmethod
    def from_state(cls, state: dict) -> "EventEncoder":
        """
        Reconstruct encoder from serialized state.

        Args:
            state: Dictionary from get_state()

        Returns:
            New EventEncoder instance
        """
        return cls(
            num_genres=state['num_genres'],
            num_instruments=state['num_instruments'],
            encode_velocity=state.get('encode_velocity', False),
        )

    def __repr__(self) -> str:
        return (
            f"EventEncoder(num_genres={self.vocabulary.num_genres}, "
            f"num_instruments={self.vocabulary.num_instruments}, "
            f"vocab_size={self.vocab_size}, "
            f"encode_velocity={self.encode_velocity})"
        )
