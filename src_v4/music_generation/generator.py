"""
Music generator using trained models.

Clean autoregressive generation with various sampling strategies.
Converts tokens to MIDI via muspy.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import muspy

from ..model_training.architectures.base_model import BaseMusicModel
from ..data_preprocessing.encoders.base_encoder import BaseEncoder

if TYPE_CHECKING:
    from ..model_training.model_bundle import ModelBundle


@dataclass
class GenerationConfig:
    """Configuration for music generation."""

    # Sequence length
    max_length: int = 1024

    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    # MIDI output
    resolution: int = 24  # Ticks per beat
    tempo: float = 120.0  # BPM

    # Quality thresholds
    min_notes: int = 10  # Minimum notes per track
    min_bars: int = 4  # Minimum bars (song length)
    max_retries: int = 3  # Max attempts to meet thresholds

    # Random seed
    seed: Optional[int] = None


class MusicGenerator:
    """
    Generate music using trained models.

    Works with both Transformer and LSTM models that extend BaseMusicModel,
    and any encoder that extends BaseEncoder.
    """

    def __init__(
        self,
        model: BaseMusicModel,
        encoder: BaseEncoder,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize generator.

        Args:
            model: Trained model (TransformerModel or LSTMModel)
            encoder: Encoder instance (EventEncoder or REMIEncoder)
            config: Generation configuration
        """
        self.model = model
        self.encoder = encoder
        self.config = config or GenerationConfig()

    @classmethod
    def from_bundle(
        cls,
        bundle: "ModelBundle",
        config: Optional[GenerationConfig] = None,
    ) -> "MusicGenerator":
        """
        Create generator from a ModelBundle.

        Args:
            bundle: ModelBundle containing model and encoder
            config: Generation configuration (optional)

        Returns:
            MusicGenerator instance
        """
        return cls(
            model=bundle.model,
            encoder=bundle.encoder,
            config=config,
        )

    @classmethod
    def from_bundle_path(
        cls,
        filepath: Union[str, Path],
        config: Optional[GenerationConfig] = None,
    ) -> "MusicGenerator":
        """
        Load generator from a saved bundle file.

        Args:
            filepath: Path to the .h5 bundle file
            config: Generation configuration (optional)

        Returns:
            MusicGenerator instance
        """
        from ..model_training.model_bundle import load_model_bundle
        bundle = load_model_bundle(filepath)
        return cls.from_bundle(bundle, config)

    def generate_tokens(
        self,
        genre_id: Optional[int] = None,
        instrument_id: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a sequence of tokens.

        Args:
            genre_id: Genre ID for conditioning (optional)
            instrument_id: Instrument ID for conditioning (optional)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            seed: Random seed

        Returns:
            Array of generated token IDs
        """
        # Use config defaults if not specified
        max_length = max_length or self.config.max_length
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        seed = seed if seed is not None else self.config.seed

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        import tensorflow as tf

        # Create conditioning tokens
        start_tokens = self.encoder.create_conditioning_tokens(
            genre_id=genre_id,
            instrument_id=instrument_id,
        )

        # Convert to 2D tensor with batch dimension
        start_tokens = tf.convert_to_tensor(start_tokens, dtype=tf.int32)
        if len(start_tokens.shape) == 1:
            start_tokens = tf.expand_dims(start_tokens, 0)  # Add batch dim

        # Generate using model's generate method
        tokens = self.model.generate(
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.encoder.eos_token_id,
        )

        # Return as numpy array without batch dimension
        if hasattr(tokens, 'numpy'):
            tokens = tokens.numpy()
        if len(tokens.shape) == 2 and tokens.shape[0] == 1:
            tokens = tokens[0]

        return tokens

    def generate_events(
        self,
        genre_id: Optional[int] = None,
        instrument_id: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[str, int]]:
        """
        Generate music and decode to events.

        Args:
            genre_id: Genre ID for conditioning
            instrument_id: Instrument ID for conditioning
            **kwargs: Additional arguments for generate_tokens

        Returns:
            List of (event_type, value) tuples
        """
        tokens = self.generate_tokens(
            genre_id=genre_id,
            instrument_id=instrument_id,
            **kwargs,
        )

        events = self.encoder.decode_tokens(tokens, skip_special=True)
        return events

    def generate_notes(
        self,
        genre_id: Optional[int] = None,
        instrument_id: Optional[int] = None,
        **kwargs,
    ) -> List[muspy.Note]:
        """
        Generate music and convert to muspy Notes.

        Args:
            genre_id: Genre ID for conditioning
            instrument_id: Instrument ID for conditioning
            **kwargs: Additional arguments for generate_tokens

        Returns:
            List of muspy.Note objects
        """
        events = self.generate_events(
            genre_id=genre_id,
            instrument_id=instrument_id,
            **kwargs,
        )

        notes = self._events_to_notes(events)
        return notes

    def generate_track(
        self,
        genre_id: Optional[int] = None,
        instrument_id: Optional[int] = None,
        program: int = 0,
        is_drum: bool = False,
        name: str = "",
        min_notes: Optional[int] = None,
        min_bars: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> muspy.Track:
        """
        Generate a single track with quality thresholds.

        Args:
            genre_id: Genre ID for conditioning
            instrument_id: Instrument ID for conditioning
            program: MIDI program number (0-127)
            is_drum: Whether this is a drum track
            name: Track name
            min_notes: Minimum notes required (uses config if None)
            min_bars: Minimum bars required (uses config if None)
            max_retries: Max retry attempts (uses config if None)
            **kwargs: Additional arguments for generate_tokens

        Returns:
            muspy.Track object
        """
        min_notes = min_notes if min_notes is not None else self.config.min_notes
        min_bars = min_bars if min_bars is not None else self.config.min_bars
        max_retries = max_retries if max_retries is not None else self.config.max_retries

        ticks_per_bar = self.config.resolution * 4  # 4 beats per bar
        min_duration = min_bars * ticks_per_bar

        best_track = None
        best_score = 0

        for attempt in range(max_retries):
            notes = self.generate_notes(
                genre_id=genre_id,
                instrument_id=instrument_id,
                **kwargs,
            )

            # Calculate track metrics
            note_count = len(notes)
            duration = max((n.time + n.duration for n in notes), default=0) if notes else 0
            bars = duration / ticks_per_bar if ticks_per_bar > 0 else 0

            # Score this attempt
            score = note_count + (bars * 10)

            if score > best_score:
                best_score = score
                best_track = notes

            # Check if meets thresholds
            if note_count >= min_notes and bars >= min_bars:
                break

        return muspy.Track(
            program=program,
            is_drum=is_drum,
            name=name,
            notes=best_track or [],
        )

    def validate_track(self, track: muspy.Track) -> Dict[str, any]:
        """
        Get validation metrics for a track.

        Args:
            track: muspy.Track to validate

        Returns:
            Dict with note_count, duration_ticks, bars, meets_min_notes, meets_min_bars
        """
        ticks_per_bar = self.config.resolution * 4
        note_count = len(track.notes)
        duration = max((n.time + n.duration for n in track.notes), default=0) if track.notes else 0
        bars = duration / ticks_per_bar if ticks_per_bar > 0 else 0

        return {
            "note_count": note_count,
            "duration_ticks": duration,
            "bars": round(bars, 1),
            "meets_min_notes": note_count >= self.config.min_notes,
            "meets_min_bars": bars >= self.config.min_bars,
        }

    def generate_music(
        self,
        genre_id: Optional[int] = None,
        instrument_ids: Optional[List[int]] = None,
        programs: Optional[List[int]] = None,
        include_drums: bool = True,
        resolution: Optional[int] = None,
        tempo: Optional[float] = None,
        **kwargs,
    ) -> muspy.Music:
        """
        Generate a complete music piece with multiple tracks.

        Args:
            genre_id: Genre ID for conditioning
            instrument_ids: List of instrument IDs to generate
            programs: MIDI program numbers for each instrument
            include_drums: Whether to include a drum track
            resolution: Ticks per beat
            tempo: Tempo in BPM
            **kwargs: Additional arguments for generate_tokens

        Returns:
            muspy.Music object with all tracks
        """
        resolution = resolution or self.config.resolution
        tempo = tempo or self.config.tempo

        tracks = []

        # Generate drum track first if requested
        if include_drums:
            drum_track = self.generate_track(
                genre_id=genre_id,
                instrument_id=128,  # Standard drum ID
                program=0,
                is_drum=True,
                name="Drums",
                **kwargs,
            )
            tracks.append(drum_track)

        # Generate other instrument tracks
        if instrument_ids is not None:
            programs = programs or instrument_ids  # Use instrument ID as program if not specified

            for i, inst_id in enumerate(instrument_ids):
                program = programs[i] if i < len(programs) else inst_id
                track = self.generate_track(
                    genre_id=genre_id,
                    instrument_id=inst_id,
                    program=program,
                    is_drum=False,
                    name=f"Track_{inst_id}",
                    **kwargs,
                )
                tracks.append(track)

        # Create Music object
        music = muspy.Music(
            resolution=resolution,
            tempos=[muspy.Tempo(time=0, qpm=tempo)],
            tracks=tracks,
        )

        return music

    def generate_midi(
        self,
        output_path: Union[str, Path],
        genre_id: Optional[int] = None,
        instrument_ids: Optional[List[int]] = None,
        programs: Optional[List[int]] = None,
        include_drums: bool = True,
        **kwargs,
    ) -> muspy.Music:
        """
        Generate music and save to MIDI file.

        Args:
            output_path: Path to save MIDI file
            genre_id: Genre ID for conditioning
            instrument_ids: List of instrument IDs to generate
            programs: MIDI program numbers for each instrument
            include_drums: Whether to include drums
            **kwargs: Additional arguments for generate_music

        Returns:
            Generated muspy.Music object
        """
        music = self.generate_music(
            genre_id=genre_id,
            instrument_ids=instrument_ids,
            programs=programs,
            include_drums=include_drums,
            **kwargs,
        )

        # Save to MIDI
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        music.write_midi(str(output_path))

        return music

    def _events_to_notes(
        self,
        events: List[Tuple[str, int]],
    ) -> List[muspy.Note]:
        """
        Convert event sequence to muspy Notes.

        Handles both event-based (note_on/note_off/time_shift) and
        REMI-based (bar/position/pitch/duration/velocity) encodings.

        Args:
            events: List of (event_type, value) tuples

        Returns:
            List of muspy.Note objects
        """
        notes = []
        current_time = 0
        active_notes: Dict[int, Tuple[int, int]] = {}  # pitch -> (start_time, velocity)

        # REMI state
        current_bar = 0
        current_position = 0
        current_pitch = None
        current_duration = None
        current_velocity = 64

        for event_type, value in events:
            if event_type == 'time_shift':
                # Event-based: advance time
                current_time += value

            elif event_type == 'note_on':
                # Event-based: start a note
                if value not in active_notes:
                    active_notes[value] = (current_time, 64)  # Default velocity

            elif event_type == 'note_off':
                # Event-based: end a note
                if value in active_notes:
                    start_time, velocity = active_notes.pop(value)
                    duration = current_time - start_time
                    if duration > 0:
                        notes.append(muspy.Note(
                            time=start_time,
                            pitch=value,
                            duration=duration,
                            velocity=velocity,
                        ))

            elif event_type == 'velocity':
                # Update velocity for subsequent notes
                current_velocity = value
                # Update velocity for active notes
                for pitch in active_notes:
                    start, _ = active_notes[pitch]
                    active_notes[pitch] = (start, value)

            # REMI events
            elif event_type == 'bar':
                current_bar = value
                current_position = 0

            elif event_type == 'position':
                current_position = value

            elif event_type == 'pitch':
                current_pitch = value

            elif event_type == 'duration':
                current_duration = value

                # In REMI, duration comes last - create note now
                if current_pitch is not None and current_duration is not None:
                    # Calculate absolute time from bar and position
                    # Assuming 4/4 time signature and positions are in ticks
                    ticks_per_bar = self.config.resolution * 4  # 4 beats per bar
                    time = current_bar * ticks_per_bar + current_position

                    notes.append(muspy.Note(
                        time=time,
                        pitch=current_pitch,
                        duration=current_duration,
                        velocity=current_velocity,
                    ))

                    # Reset for next note
                    current_pitch = None
                    current_duration = None

        # Close any remaining active notes (event-based)
        for pitch, (start_time, velocity) in active_notes.items():
            duration = current_time - start_time
            if duration > 0:
                notes.append(muspy.Note(
                    time=start_time,
                    pitch=pitch,
                    duration=duration,
                    velocity=velocity,
                ))

        # Sort by time
        notes.sort(key=lambda n: (n.time, n.pitch))

        return notes

    def tokens_to_pianoroll(
        self,
        tokens: np.ndarray,
        num_time_steps: int = 2048,
    ) -> np.ndarray:
        """
        Convert tokens to pianoroll representation.

        Args:
            tokens: Array of token IDs
            num_time_steps: Length of output pianoroll

        Returns:
            Pianoroll array (128, num_time_steps)
        """
        events = self.encoder.decode_tokens(tokens, skip_special=True)
        notes = self._events_to_notes(events)

        pianoroll = np.zeros((128, num_time_steps), dtype=np.float32)

        for note in notes:
            if 0 <= note.pitch < 128:
                start = min(note.time, num_time_steps - 1)
                end = min(note.time + note.duration, num_time_steps)
                pianoroll[note.pitch, start:end] = 1.0

        return pianoroll
