"""
Monophonic Music Generator.

Generates single tracks one at a time using event-based or REMI encoding.
Each track is generated independently.

Author: Murilo de Freitas Spinelli, Ryan
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import muspy

from .base_generator import BaseGenerator, GeneratorConfig

if TYPE_CHECKING:
    from ..models.base_model import BaseMusicModel
    from ..data.encoders.base_encoder import BaseEncoder
    from ..models.model_bundle import ModelBundle


@dataclass
class MonoGeneratorConfig(GeneratorConfig):
    """Configuration for monophonic generation."""
    pass  # Inherits all from GeneratorConfig


class MonoGenerator(BaseGenerator):
    """
    Generate music one track at a time.

    Works with event-based or REMI encoders where each track
    is generated independently. Suitable for single-instrument
    or layered multi-track generation.
    """

    def __init__(
        self,
        model: "BaseMusicModel",
        encoder: "BaseEncoder",
        config: Optional[MonoGeneratorConfig] = None,
    ):
        """
        Initialize monophonic generator.

        Args:
            model: Trained model (TransformerModel or LSTMModel)
            encoder: Encoder instance (EventEncoder or REMIEncoder)
            config: Generation configuration
        """
        super().__init__(model, encoder, config or MonoGeneratorConfig())

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
        Generate a sequence of tokens for a single track.

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
        params = self._apply_sampling_defaults(max_length, temperature, top_k, top_p, seed)

        # Create conditioning tokens
        start_tokens = self.encoder.create_conditioning_tokens(
            genre_id=genre_id,
            instrument_id=instrument_id,
        )

        # Prepare for model
        start_tokens = self._prepare_start_tokens(start_tokens)

        # Generate
        tokens = self.model.generate(
            start_tokens=start_tokens,
            max_length=params['max_length'],
            temperature=params['temperature'],
            top_k=params['top_k'],
            top_p=params['top_p'],
            eos_token_id=self.encoder.eos_token_id,
        )

        return self._extract_tokens(tokens)

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
        return self.encoder.decode_tokens(tokens, skip_special=True)

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
        return self._events_to_notes(events)

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
            min_notes: Minimum notes required
            min_bars: Minimum bars required
            max_retries: Max retry attempts
            **kwargs: Additional arguments for generate_tokens

        Returns:
            muspy.Track object
        """
        min_notes = min_notes if min_notes is not None else self.config.min_notes
        min_bars = min_bars if min_bars is not None else self.config.min_bars
        max_retries = max_retries if max_retries is not None else self.config.max_retries

        ticks_per_bar = self.config.resolution * 4

        best_track = None
        best_score = 0

        for attempt in range(max_retries):
            notes = self.generate_notes(
                genre_id=genre_id,
                instrument_id=instrument_id,
                **kwargs,
            )

            note_count = len(notes)
            duration = max((n.time + n.duration for n in notes), default=0) if notes else 0
            bars = duration / ticks_per_bar if ticks_per_bar > 0 else 0

            score = note_count + (bars * 10)

            if score > best_score:
                best_score = score
                best_track = notes

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
            Dict with metrics
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

        Each track is generated independently (they don't know about each other).

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
                instrument_id=128,
                program=0,
                is_drum=True,
                name="Drums",
                **kwargs,
            )
            tracks.append(drum_track)

        # Generate other instrument tracks
        if instrument_ids is not None:
            programs = programs or instrument_ids

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

        return muspy.Music(
            resolution=resolution,
            tempos=[muspy.Tempo(time=0, qpm=tempo)],
            tracks=tracks,
        )

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
        active_notes: Dict[int, Tuple[int, int]] = {}

        # REMI state
        current_bar = 0
        current_position = 0
        current_pitch = None
        current_duration = None
        current_velocity = 64

        for event_type, value in events:
            if event_type == 'time_shift':
                current_time += value

            elif event_type == 'note_on':
                if value not in active_notes:
                    active_notes[value] = (current_time, 64)

            elif event_type == 'note_off':
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
                current_velocity = value
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

                if current_pitch is not None and current_duration is not None:
                    ticks_per_bar = self.config.resolution * 4
                    time = current_bar * ticks_per_bar + current_position

                    notes.append(muspy.Note(
                        time=time,
                        pitch=current_pitch,
                        duration=current_duration,
                        velocity=current_velocity,
                    ))

                    current_pitch = None
                    current_duration = None

        # Close remaining active notes
        for pitch, (start_time, velocity) in active_notes.items():
            duration = current_time - start_time
            if duration > 0:
                notes.append(muspy.Note(
                    time=start_time,
                    pitch=pitch,
                    duration=duration,
                    velocity=velocity,
                ))

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


# Backwards compatibility aliases
MusicGenerator = MonoGenerator
GenerationConfig = MonoGeneratorConfig
