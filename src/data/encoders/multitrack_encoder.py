"""
Multi-Track Encoder - Interleaved encoding for all tracks.

Encodes all tracks from a song into a single interleaved sequence,
sorted by time position. This allows the model to learn relationships
between all instruments simultaneously.

Token sequence structure:
    [BOS, genre, bar_0,
     pos_0, inst_drums, pitch_36, dur_2, vel_100,
     pos_0, inst_bass, pitch_40, dur_4, vel_80,
     pos_2, inst_drums, pitch_38, dur_2, vel_100,
     pos_2, inst_guitar, pitch_64, dur_8, vel_70,
     bar_1, ...]

Author: Murilo de Freitas Spinelli
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
import numpy as np
import muspy

from .base_encoder import BaseEncoder, EncodedSequence


@dataclass
class MultiTrackEvent:
    """A single event with timing and instrument info."""
    time: int           # Absolute time in ticks
    bar: int            # Bar number
    position: int       # Position within bar
    instrument_id: int  # Instrument (0-127 melodic, 128 drums)
    pitch: int          # MIDI pitch
    duration: int       # Duration in ticks
    velocity: int       # Velocity (0-127)


class MultiTrackEncoder(BaseEncoder):
    """
    Encoder that interleaves all tracks by time position.

    The model sees all instruments at each time step, learning
    their relationships (e.g., bass follows kick, guitar accents snare).

    Vocabulary structure:
        0: PAD
        1: BOS (beginning of sequence)
        2: EOS (end of sequence)
        3: SEP (separator, unused but reserved)
        4-67: Bar tokens (bar_0 to bar_63)
        68-131: Position tokens (pos_0 to pos_63, within bar)
        132-259: Instrument tokens (inst_0 to inst_127, plus inst_128 for drums)
        260-387: Pitch tokens (pitch_0 to pitch_127)
        388-419: Duration tokens (dur_1 to dur_32, in subdivisions)
        420-451: Velocity tokens (vel_0 to vel_31, quantized)
        452+: Genre tokens (genre_0, genre_1, ...)
    """

    # Vocabulary ranges
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    SEP_TOKEN = 3

    BAR_OFFSET = 4
    NUM_BARS = 64

    POSITION_OFFSET = BAR_OFFSET + NUM_BARS  # 68
    NUM_POSITIONS = 64  # Positions per bar (16th note resolution at 16 per bar)

    INSTRUMENT_OFFSET = POSITION_OFFSET + NUM_POSITIONS  # 132
    NUM_INSTRUMENTS = 129  # 0-127 melodic + 128 drums

    PITCH_OFFSET = INSTRUMENT_OFFSET + NUM_INSTRUMENTS  # 261
    NUM_PITCHES = 128

    DURATION_OFFSET = PITCH_OFFSET + NUM_PITCHES  # 389
    NUM_DURATIONS = 32  # Duration values 1-32

    VELOCITY_OFFSET = DURATION_OFFSET + NUM_DURATIONS  # 421
    NUM_VELOCITIES = 32  # Quantized velocity bins

    GENRE_OFFSET = VELOCITY_OFFSET + NUM_VELOCITIES  # 453

    def __init__(
        self,
        num_genres: int = 10,
        resolution: int = 24,
        max_bars: int = 64,
        positions_per_bar: int = 16,
    ):
        """
        Initialize multi-track encoder.

        Args:
            num_genres: Number of genre conditioning tokens
            resolution: Ticks per beat (quarter note)
            max_bars: Maximum number of bars to encode
            positions_per_bar: Time resolution within each bar
        """
        self.num_genres = num_genres
        self.resolution = resolution
        self.max_bars = max_bars
        self.positions_per_bar = positions_per_bar

        # Ticks per bar (assuming 4/4 time)
        self.ticks_per_bar = resolution * 4
        # Ticks per position
        self.ticks_per_position = self.ticks_per_bar // positions_per_bar

        # Calculate vocab size
        self._vocab_size = self.GENRE_OFFSET + num_genres

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_tokens(self) -> dict:
        """Dictionary of special token IDs."""
        return {
            'pad': self.PAD_TOKEN,
            'bos': self.BOS_TOKEN,
            'eos': self.EOS_TOKEN,
            'sep': self.SEP_TOKEN,
        }

    @property
    def pad_token_id(self) -> int:
        return self.PAD_TOKEN

    @property
    def bos_token_id(self) -> int:
        return self.BOS_TOKEN

    @property
    def eos_token_id(self) -> int:
        return self.EOS_TOKEN

    def bar_token(self, bar: int) -> int:
        """Create bar token."""
        return self.BAR_OFFSET + min(bar, self.NUM_BARS - 1)

    def position_token(self, position: int) -> int:
        """Create position token."""
        return self.POSITION_OFFSET + min(position, self.NUM_POSITIONS - 1)

    def instrument_token(self, instrument_id: int) -> int:
        """Create instrument token."""
        return self.INSTRUMENT_OFFSET + min(instrument_id, self.NUM_INSTRUMENTS - 1)

    def pitch_token(self, pitch: int) -> int:
        """Create pitch token."""
        return self.PITCH_OFFSET + min(max(pitch, 0), self.NUM_PITCHES - 1)

    def duration_token(self, duration_ticks: int) -> int:
        """Create duration token (quantized to subdivisions)."""
        # Convert ticks to subdivisions
        subdivisions = max(1, duration_ticks // self.ticks_per_position)
        subdivisions = min(subdivisions, self.NUM_DURATIONS)
        return self.DURATION_OFFSET + subdivisions - 1

    def velocity_token(self, velocity: int) -> int:
        """Create velocity token (quantized to bins)."""
        bin_idx = min(velocity // 4, self.NUM_VELOCITIES - 1)
        return self.VELOCITY_OFFSET + bin_idx

    def genre_token(self, genre_id: int) -> int:
        """Create genre token."""
        return self.GENRE_OFFSET + min(genre_id, self.num_genres - 1)

    def decode_token(self, token: int) -> Tuple[str, int]:
        """Decode a single token to (event_type, value)."""
        if token == self.PAD_TOKEN:
            return ('pad', 0)
        elif token == self.BOS_TOKEN:
            return ('bos', 0)
        elif token == self.EOS_TOKEN:
            return ('eos', 0)
        elif token == self.SEP_TOKEN:
            return ('sep', 0)
        elif self.BAR_OFFSET <= token < self.POSITION_OFFSET:
            return ('bar', token - self.BAR_OFFSET)
        elif self.POSITION_OFFSET <= token < self.INSTRUMENT_OFFSET:
            return ('position', token - self.POSITION_OFFSET)
        elif self.INSTRUMENT_OFFSET <= token < self.PITCH_OFFSET:
            return ('instrument', token - self.INSTRUMENT_OFFSET)
        elif self.PITCH_OFFSET <= token < self.DURATION_OFFSET:
            return ('pitch', token - self.PITCH_OFFSET)
        elif self.DURATION_OFFSET <= token < self.VELOCITY_OFFSET:
            return ('duration', token - self.DURATION_OFFSET + 1)
        elif self.VELOCITY_OFFSET <= token < self.GENRE_OFFSET:
            return ('velocity', (token - self.VELOCITY_OFFSET) * 4)
        elif token >= self.GENRE_OFFSET:
            return ('genre', token - self.GENRE_OFFSET)
        else:
            return ('unknown', token)

    def encode_music(
        self,
        music: muspy.Music,
        genre_id: int = 0,
        max_length: int = 2048,
    ) -> EncodedSequence:
        """
        Encode a complete multi-track music piece.

        Args:
            music: muspy.Music object with multiple tracks
            genre_id: Genre conditioning ID
            max_length: Maximum sequence length

        Returns:
            EncodedSequence with interleaved tokens
        """
        # Collect all events from all tracks
        all_events: List[MultiTrackEvent] = []

        for track in music.tracks:
            instrument_id = 128 if track.is_drum else track.program

            for note in track.notes:
                bar = note.time // self.ticks_per_bar
                position_in_bar = (note.time % self.ticks_per_bar) // self.ticks_per_position

                if bar < self.max_bars:
                    all_events.append(MultiTrackEvent(
                        time=note.time,
                        bar=bar,
                        position=position_in_bar,
                        instrument_id=instrument_id,
                        pitch=note.pitch,
                        duration=note.duration,
                        velocity=note.velocity,
                    ))

        # Sort by time, then by instrument (drums first for consistency)
        all_events.sort(key=lambda e: (e.time, 0 if e.instrument_id == 128 else e.instrument_id + 1))

        # Build token sequence
        tokens = [self.BOS_TOKEN, self.genre_token(genre_id)]

        current_bar = -1
        current_position = -1

        for event in all_events:
            # Add bar token if new bar
            if event.bar != current_bar:
                tokens.append(self.bar_token(event.bar))
                current_bar = event.bar
                current_position = -1

            # Add position token if new position
            if event.position != current_position:
                tokens.append(self.position_token(event.position))
                current_position = event.position

            # Add event tokens: instrument, pitch, duration, velocity
            tokens.extend([
                self.instrument_token(event.instrument_id),
                self.pitch_token(event.pitch),
                self.duration_token(event.duration),
                self.velocity_token(event.velocity),
            ])

            # Check length
            if len(tokens) >= max_length - 1:
                break

        tokens.append(self.EOS_TOKEN)

        # Pad or truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        attention_mask = [1] * len(tokens)

        while len(tokens) < max_length:
            tokens.append(self.PAD_TOKEN)
            attention_mask.append(0)

        return EncodedSequence(
            token_ids=np.array(tokens, dtype=np.int32),
            attention_mask=np.array(attention_mask, dtype=np.int32),
        )

    def encode_track(
        self,
        track: muspy.Track,
        genre_id: int = 0,
        instrument_id: int = 0,
        max_length: int = 2048,
    ) -> EncodedSequence:
        """
        Encode a single track (for compatibility with base class).

        For true multi-track encoding, use encode_music() instead.
        """
        # Create a temporary Music object with just this track
        music = muspy.Music(
            resolution=self.resolution,
            tracks=[track],
        )
        return self.encode_music(music, genre_id, max_length)

    def decode_tokens(
        self,
        tokens: np.ndarray,
        skip_special: bool = True,
    ) -> List[Tuple[str, int]]:
        """Decode tokens to list of (event_type, value) tuples."""
        events = []
        for token in tokens:
            event_type, value = self.decode_token(int(token))
            if skip_special and event_type in ('pad', 'bos', 'eos', 'sep'):
                continue
            events.append((event_type, value))
        return events

    def decode_to_music(
        self,
        tokens: np.ndarray,
        resolution: Optional[int] = None,
        tempo: float = 120.0,
    ) -> muspy.Music:
        """
        Decode token sequence back to muspy.Music with multiple tracks.

        Args:
            tokens: Token sequence
            resolution: Ticks per beat (uses encoder default if None)
            tempo: Tempo in BPM

        Returns:
            muspy.Music object with separate tracks for each instrument
        """
        resolution = resolution or self.resolution
        ticks_per_bar = resolution * 4
        ticks_per_position = ticks_per_bar // self.positions_per_bar

        events = self.decode_tokens(tokens, skip_special=True)

        # Group notes by instrument
        instrument_notes: Dict[int, List[muspy.Note]] = {}

        current_bar = 0
        current_position = 0
        current_instrument = 0
        current_pitch = None
        current_duration = None
        current_velocity = 64

        for event_type, value in events:
            if event_type == 'bar':
                current_bar = value
            elif event_type == 'position':
                current_position = value
            elif event_type == 'instrument':
                current_instrument = value
            elif event_type == 'pitch':
                current_pitch = value
            elif event_type == 'duration':
                current_duration = value
            elif event_type == 'velocity':
                current_velocity = value

                # After velocity, we have a complete note
                if current_pitch is not None and current_duration is not None:
                    time = current_bar * ticks_per_bar + current_position * ticks_per_position
                    duration = current_duration * ticks_per_position

                    if current_instrument not in instrument_notes:
                        instrument_notes[current_instrument] = []

                    instrument_notes[current_instrument].append(muspy.Note(
                        time=time,
                        pitch=current_pitch,
                        duration=duration,
                        velocity=current_velocity,
                    ))

                    # Reset for next note
                    current_pitch = None
                    current_duration = None

        # Create tracks
        tracks = []
        for inst_id in sorted(instrument_notes.keys()):
            notes = instrument_notes[inst_id]
            is_drum = (inst_id == 128)
            program = 0 if is_drum else inst_id

            track = muspy.Track(
                program=program,
                is_drum=is_drum,
                name=f"Track_{inst_id}" if not is_drum else "Drums",
                notes=sorted(notes, key=lambda n: n.time),
            )
            tracks.append(track)

        return muspy.Music(
            resolution=resolution,
            tempos=[muspy.Tempo(time=0, qpm=tempo)],
            tracks=tracks,
        )

    def create_conditioning_tokens(
        self,
        genre_id: int = 0,
        instrument_id: int = 0,  # Ignored for multi-track
        instruments: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Create conditioning tokens for generation.

        Sequence: [BOS, genre, inst_hint_0, inst_hint_1, ..., (bar added by caller)]

        Args:
            genre_id: Genre ID
            instrument_id: Ignored (for compatibility with base class)
            instruments: Optional list of MIDI program IDs (0-127 melodic, 128 drums)
                         to hint which instruments should appear. These are injected
                         as instrument tokens right after the genre token, biasing
                         the model toward those programs.

        Returns:
            Array of conditioning tokens
        """
        tokens = [self.BOS_TOKEN]

        if genre_id is not None:
            tokens.append(self.genre_token(genre_id))

        if instruments:
            for inst_id in instruments:
                if 0 <= inst_id <= 128:
                    tokens.append(self.INSTRUMENT_OFFSET + inst_id)

        return np.array(tokens, dtype=np.int32)

    def get_state(self) -> Dict[str, Any]:
        """Get encoder state for serialization."""
        return {
            'encoder_type': 'multitrack',
            'num_genres': self.num_genres,
            'resolution': self.resolution,
            'max_bars': self.max_bars,
            'positions_per_bar': self.positions_per_bar,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'MultiTrackEncoder':
        """Create encoder from serialized state."""
        return cls(
            num_genres=state.get('num_genres', 10),
            resolution=state.get('resolution', 24),
            max_bars=state.get('max_bars', 64),
            positions_per_bar=state.get('positions_per_bar', 16),
        )

    def __repr__(self) -> str:
        return (
            f"MultiTrackEncoder(vocab_size={self.vocab_size}, "
            f"num_genres={self.num_genres}, resolution={self.resolution})"
        )
