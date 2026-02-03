"""
REMI (Revamped MIDI-derived Events) encoder for music generation.

Based on "Pop Music Transformer" (Huang & Yang, 2020).

REMI adds musical structure awareness through:
    - Bar tokens: Mark measure boundaries
    - Position tokens: Beat subdivisions within bars
    - Duration tokens: Explicit note length (no note-off needed)
    - Velocity tokens: Dynamics

This results in shorter sequences and better musical structure learning
compared to basic note-on/note-off event encoding.

Typical sequence:
    BOS, Genre, Instrument, Bar, Position(0), Velocity(80), Pitch(60), Duration(8),
    Pitch(64), Duration(8), Position(8), Velocity(70), Pitch(67), Duration(4), Bar, ...

Author: Murilo de Freitas Spinelli
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from .base_encoder import BaseEncoder, EncodedSequence


@dataclass
class REMIVocabulary:
    """
    Token vocabulary for REMI music representation.

    Token ranges:
        0:            Bar token (marks start of measure)
        1-32:         Position (32 subdivisions per bar, e.g., 32nd notes)
        33-160:       Pitch (128 MIDI pitches)
        161-224:      Duration (64 values, in position units)
        225-256:      Velocity (32 bins)
        257:          PAD
        258:          BOS (begin sequence)
        259:          EOS (end sequence)
        260+:         Genre tokens
        260+G:        Instrument tokens
    """

    # Structure tokens
    BAR_TOKEN: int = 0
    POSITION_OFFSET: int = 1
    NUM_POSITIONS: int = 32  # Subdivisions per bar (32nd note resolution)

    # Note tokens
    PITCH_OFFSET: int = 33  # 1 + 32 positions
    NUM_PITCHES: int = 128

    DURATION_OFFSET: int = 161  # 33 + 128 pitches
    MAX_DURATION: int = 64  # Max duration in position units

    VELOCITY_OFFSET: int = 225  # 161 + 64 durations
    NUM_VELOCITY_BINS: int = 32

    # Special tokens
    PAD_TOKEN: int = 257  # 225 + 32 velocity bins
    BOS_TOKEN: int = 258
    EOS_TOKEN: int = 259

    # Conditioning tokens
    GENRE_OFFSET: int = 260

    # Configurable counts
    num_genres: int = 0
    num_instruments: int = 129  # 0-127 + drums (128)

    # Timing configuration
    ticks_per_bar: int = 96  # Assumes 4/4 time with resolution=24 (24 * 4 = 96)

    # Computed after init
    INSTRUMENT_OFFSET: int = field(init=False)

    def __post_init__(self):
        """Calculate instrument offset after genre tokens."""
        self.INSTRUMENT_OFFSET = self.GENRE_OFFSET + self.num_genres
        # Ticks per position for quantization
        self._ticks_per_position = self.ticks_per_bar // self.NUM_POSITIONS

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including all token types."""
        return self.INSTRUMENT_OFFSET + self.num_instruments

    @property
    def ticks_per_position(self) -> int:
        """Number of ticks per position unit."""
        return self._ticks_per_position

    def encode_bar(self) -> int:
        """Encode a bar marker."""
        return self.BAR_TOKEN

    def encode_position(self, position: int) -> int:
        """Encode a position within bar (0 to NUM_POSITIONS-1)."""
        position = max(0, min(self.NUM_POSITIONS - 1, position))
        return self.POSITION_OFFSET + position

    def encode_pitch(self, pitch: int) -> int:
        """Encode a pitch value (0-127)."""
        pitch = max(0, min(127, pitch))
        return self.PITCH_OFFSET + pitch

    def encode_duration(self, duration: int) -> int:
        """
        Encode a duration in position units (1 to MAX_DURATION).

        Duration is clamped to [1, MAX_DURATION].
        """
        duration = max(1, min(self.MAX_DURATION, duration))
        return self.DURATION_OFFSET + duration - 1  # 1-indexed to 0-indexed

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
            'bar', 'position', 'pitch', 'duration', 'velocity',
            'pad', 'bos', 'eos', 'genre', 'instrument'
        """
        if token == self.BAR_TOKEN:
            return ('bar', 0)
        elif token == self.PAD_TOKEN:
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
        elif token >= self.DURATION_OFFSET:
            return ('duration', token - self.DURATION_OFFSET + 1)  # 0-indexed to 1-indexed
        elif token >= self.PITCH_OFFSET:
            return ('pitch', token - self.PITCH_OFFSET)
        elif token >= self.POSITION_OFFSET:
            return ('position', token - self.POSITION_OFFSET)
        else:
            return ('unknown', token)

    def is_special_token(self, token: int) -> bool:
        """Check if token is PAD, BOS, or EOS."""
        return token in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN)

    def is_conditioning_token(self, token: int) -> bool:
        """Check if token is a genre or instrument token."""
        return token >= self.GENRE_OFFSET

    def is_structure_token(self, token: int) -> bool:
        """Check if token is a bar or position token."""
        return token == self.BAR_TOKEN or (self.POSITION_OFFSET <= token < self.PITCH_OFFSET)

    def is_note_token(self, token: int) -> bool:
        """Check if token is pitch, duration, or velocity."""
        return self.PITCH_OFFSET <= token < self.PAD_TOKEN

    def get_token_info(self) -> Dict[str, Any]:
        """Get dictionary of token type boundaries for debugging."""
        return {
            'bar': self.BAR_TOKEN,
            'position_range': (self.POSITION_OFFSET, self.POSITION_OFFSET + self.NUM_POSITIONS - 1),
            'pitch_range': (self.PITCH_OFFSET, self.PITCH_OFFSET + self.NUM_PITCHES - 1),
            'duration_range': (self.DURATION_OFFSET, self.DURATION_OFFSET + self.MAX_DURATION - 1),
            'velocity_range': (self.VELOCITY_OFFSET, self.VELOCITY_OFFSET + self.NUM_VELOCITY_BINS - 1),
            'pad': self.PAD_TOKEN,
            'bos': self.BOS_TOKEN,
            'eos': self.EOS_TOKEN,
            'genre_range': (self.GENRE_OFFSET, self.INSTRUMENT_OFFSET - 1),
            'instrument_range': (self.INSTRUMENT_OFFSET, self.vocab_size - 1),
            'vocab_size': self.vocab_size,
            'ticks_per_bar': self.ticks_per_bar,
            'ticks_per_position': self.ticks_per_position,
        }


class REMIEncoder(BaseEncoder):
    """
    REMI-based music encoder with bar/position structure.

    Converts muspy.Track objects to REMI token sequences:
        - Bar tokens mark measure boundaries
        - Position tokens indicate beat position within bar
        - Each note is: [Velocity, Pitch, Duration] at current position
        - No note-off tokens needed (duration is explicit)

    This results in better polyphonic learning and shorter sequences.
    """

    def __init__(
        self,
        num_genres: int = 0,
        num_instruments: int = 129,
        resolution: int = 24,
        positions_per_bar: int = 32,
        time_signature: Tuple[int, int] = (4, 4),
    ):
        """
        Initialize the REMI encoder.

        Args:
            num_genres: Number of genre conditioning tokens
            num_instruments: Number of instrument tokens (default 129 = 128 MIDI + drums)
            resolution: Ticks per beat (quarter note)
            positions_per_bar: Number of position subdivisions per bar
            time_signature: (numerator, denominator) tuple, e.g., (4, 4)
        """
        self.resolution = resolution
        self.positions_per_bar = positions_per_bar
        self.time_signature = time_signature

        # Calculate ticks per bar based on time signature
        # For 4/4: 4 beats * resolution = ticks_per_bar
        beats_per_bar = time_signature[0] * (4 / time_signature[1])
        ticks_per_bar = int(beats_per_bar * resolution)

        self.vocabulary = REMIVocabulary(
            num_genres=num_genres,
            num_instruments=num_instruments,
            NUM_POSITIONS=positions_per_bar,
            ticks_per_bar=ticks_per_bar,
        )

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
            'bar': self.vocabulary.BAR_TOKEN,
        }

    @property
    def ticks_per_position(self) -> int:
        """Number of ticks per position unit."""
        return self.vocabulary.ticks_per_position

    @property
    def ticks_per_bar(self) -> int:
        """Number of ticks per bar."""
        return self.vocabulary.ticks_per_bar

    def _quantize_time(self, ticks: int) -> Tuple[int, int]:
        """
        Quantize a tick time to bar and position.

        Args:
            ticks: Time in ticks

        Returns:
            Tuple of (bar_number, position_within_bar)
        """
        bar = ticks // self.ticks_per_bar
        position_in_ticks = ticks % self.ticks_per_bar
        position = position_in_ticks // self.ticks_per_position
        return bar, min(position, self.positions_per_bar - 1)

    def _quantize_duration(self, duration_ticks: int) -> int:
        """
        Quantize a duration in ticks to position units.

        Args:
            duration_ticks: Duration in ticks

        Returns:
            Duration in position units (1 to MAX_DURATION)
        """
        duration_positions = max(1, duration_ticks // self.ticks_per_position)
        return min(duration_positions, self.vocabulary.MAX_DURATION)

    def track_to_remi_events(
        self,
        track: Any,
    ) -> List[Tuple[str, int]]:
        """
        Convert a muspy.Track to REMI event representation.

        Groups notes by bar and position, outputting:
            Bar -> Position -> [Velocity, Pitch, Duration]* -> Position -> ...

        Args:
            track: muspy.Track object

        Returns:
            List of (event_type, value) tuples
        """
        if not track.notes:
            return []

        # Group notes by (bar, position)
        notes_by_position: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}

        for note in track.notes:
            bar, position = self._quantize_time(note.time)
            duration = self._quantize_duration(note.duration)
            pitch = max(0, min(127, note.pitch))
            velocity = max(0, min(127, note.velocity))

            key = (bar, position)
            if key not in notes_by_position:
                notes_by_position[key] = []
            notes_by_position[key].append((pitch, duration, velocity))

        if not notes_by_position:
            return []

        # Sort by bar, then position
        sorted_positions = sorted(notes_by_position.keys())

        events = []
        current_bar = -1

        for bar, position in sorted_positions:
            # Add bar token if new bar
            if bar > current_bar:
                # Add bar tokens for any skipped bars too
                for b in range(current_bar + 1, bar + 1):
                    events.append(('bar', 0))
                current_bar = bar

            # Add position token
            events.append(('position', position))

            # Add all notes at this position
            # Sort by pitch for consistency
            notes = sorted(notes_by_position[(bar, position)], key=lambda x: x[0])
            for pitch, duration, velocity in notes:
                events.append(('velocity', velocity))
                events.append(('pitch', pitch))
                events.append(('duration', duration))

        return events

    def events_to_tokens(
        self,
        events: List[Tuple[str, int]],
        genre_id: int,
        instrument_id: int,
    ) -> List[int]:
        """
        Convert REMI events to token IDs.

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
            if event_type == 'bar':
                tokens.append(vocab.encode_bar())
            elif event_type == 'position':
                tokens.append(vocab.encode_position(value))
            elif event_type == 'pitch':
                tokens.append(vocab.encode_pitch(value))
            elif event_type == 'duration':
                tokens.append(vocab.encode_duration(value))
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
        Encode a music track to a REMI token sequence.

        Args:
            track: muspy.Track object
            genre_id: Genre conditioning ID
            instrument_id: Instrument conditioning ID
            max_length: Maximum sequence length (including special tokens)

        Returns:
            EncodedSequence with token_ids and attention_mask
        """
        # Convert track to REMI events
        events = self.track_to_remi_events(track)

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
                'encoding': 'remi',
            }
        )

    def decode_tokens(
        self,
        tokens: np.ndarray,
        skip_special: bool = True,
    ) -> List[Tuple[str, int]]:
        """
        Decode a token sequence back to REMI events.

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
    ) -> List[Dict[str, int]]:
        """
        Decode tokens to a list of note dictionaries.

        Reconstructs note timing from bar and position tokens.

        Args:
            tokens: Array of token IDs

        Returns:
            List of dicts with keys: pitch, time, duration, velocity
        """
        events = self.decode_tokens(tokens, skip_special=True)

        notes = []
        current_bar = 0
        current_position = 0
        current_velocity = 64  # Default velocity
        pending_pitch: Optional[int] = None

        for event_type, value in events:
            if event_type == 'bar':
                current_bar += 1
                current_position = 0
            elif event_type == 'position':
                current_position = value
            elif event_type == 'velocity':
                current_velocity = value
            elif event_type == 'pitch':
                pending_pitch = value
            elif event_type == 'duration':
                if pending_pitch is not None:
                    # Calculate time in ticks
                    time_ticks = (current_bar * self.ticks_per_bar +
                                  current_position * self.ticks_per_position)
                    duration_ticks = value * self.ticks_per_position

                    notes.append({
                        'pitch': pending_pitch,
                        'time': time_ticks,
                        'duration': duration_ticks,
                        'velocity': current_velocity,
                    })
                    pending_pitch = None

        return sorted(notes, key=lambda n: (n['time'], n['pitch']))

    def get_state(self) -> dict:
        """
        Get encoder state for serialization.

        Returns:
            Dictionary containing all parameters needed to reconstruct encoder
        """
        return {
            'encoder_type': 'remi',
            'num_genres': self.vocabulary.num_genres,
            'num_instruments': self.vocabulary.num_instruments,
            'resolution': self.resolution,
            'positions_per_bar': self.positions_per_bar,
            'time_signature': list(self.time_signature),
        }

    @classmethod
    def from_state(cls, state: dict) -> "REMIEncoder":
        """
        Reconstruct encoder from serialized state.

        Args:
            state: Dictionary from get_state()

        Returns:
            New REMIEncoder instance
        """
        return cls(
            num_genres=state['num_genres'],
            num_instruments=state['num_instruments'],
            resolution=state.get('resolution', 24),
            positions_per_bar=state.get('positions_per_bar', 32),
            time_signature=tuple(state.get('time_signature', [4, 4])),
        )

    def __repr__(self) -> str:
        return (
            f"REMIEncoder(num_genres={self.vocabulary.num_genres}, "
            f"num_instruments={self.vocabulary.num_instruments}, "
            f"vocab_size={self.vocab_size}, "
            f"resolution={self.resolution}, "
            f"positions_per_bar={self.positions_per_bar})"
        )
