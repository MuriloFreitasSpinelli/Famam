"""
Polyphonic Multi-Track Music Generator.

Generates all instruments simultaneously using interleaved encoding.
All instruments are aware of each other during generation.

Author: Murilo de Freitas Spinelli
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import muspy

from .base_generator import BaseGenerator, GeneratorConfig

if TYPE_CHECKING:
    from ..models.base_model import BaseMusicModel
    from ..data.encoders.multitrack_encoder import MultiTrackEncoder
    from ..models.model_bundle import ModelBundle


@dataclass
class PolyGeneratorConfig(GeneratorConfig):
    """Configuration for polyphonic multi-track generation."""

    # Multi-track specific
    num_bars: int = 8
    min_notes_per_track: int = 5
    min_total_notes: int = 20


class PolyGenerator(BaseGenerator):
    """
    Generate music with all tracks aware of each other.

    Uses interleaved encoding where all instruments are generated
    together in a single sequence, sorted by time. The model sees
    what every instrument plays at each moment.
    """

    def __init__(
        self,
        model: "BaseMusicModel",
        encoder: "MultiTrackEncoder",
        config: Optional[PolyGeneratorConfig] = None,
    ):
        """
        Initialize polyphonic generator.

        Args:
            model: Trained model (TransformerModel or LSTMModel)
            encoder: MultiTrackEncoder instance
            config: Generation configuration
        """
        super().__init__(model, encoder, config or PolyGeneratorConfig())

    def generate_tokens(
        self,
        genre_id: int = 0,
        instruments: Optional[List[int]] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate interleaved multi-track token sequence.

        Args:
            genre_id: Genre conditioning ID
            instruments: List of instrument IDs to generate (optional hint)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            seed: Random seed

        Returns:
            Array of generated tokens (interleaved multi-track)
        """
        import tensorflow as tf

        params = self._apply_sampling_defaults(max_length, temperature, top_k, top_p, seed)

        if params['seed'] is not None:
            tf.random.set_seed(params['seed'])

        # Create start tokens
        start_tokens = self.encoder.create_conditioning_tokens(
            genre_id=genre_id,
            instruments=instruments,
        )

        # Add initial bar token to help the model
        start_tokens = np.append(start_tokens, self.encoder.bar_token(0))

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

    def generate_music(
        self,
        genre_id: int = 0,
        instruments: Optional[List[int]] = None,
        num_bars: Optional[int] = None,
        temperature: Optional[float] = None,
        min_notes_per_track: Optional[int] = None,
        min_total_notes: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> muspy.Music:
        """
        Generate a complete multi-track music piece.

        All instruments are generated together - they know about each other.

        Args:
            genre_id: Genre conditioning ID
            instruments: Hint for which instruments to include
            num_bars: Number of bars to generate
            temperature: Sampling temperature
            min_notes_per_track: Minimum notes per track
            min_total_notes: Minimum total notes
            max_retries: Maximum generation attempts

        Returns:
            muspy.Music object with all tracks
        """
        num_bars = num_bars or self.config.num_bars
        min_notes_per_track = min_notes_per_track or self.config.min_notes_per_track
        min_total_notes = min_total_notes or self.config.min_total_notes
        max_retries = max_retries or self.config.max_retries

        # Estimate sequence length based on bars
        estimated_length = min(num_bars * 100, self.config.max_length)

        best_music = None
        best_score = 0

        for attempt in range(max_retries):
            tokens = self.generate_tokens(
                genre_id=genre_id,
                instruments=instruments,
                max_length=estimated_length,
                temperature=temperature,
            )

            music = self.encoder.decode_to_music(
                tokens,
                resolution=self.config.resolution,
                tempo=self.config.tempo,
            )

            total_notes = sum(len(t.notes) for t in music.tracks)
            num_tracks_with_notes = sum(
                1 for t in music.tracks
                if len(t.notes) >= min_notes_per_track
            )

            score = total_notes + (num_tracks_with_notes * 50)

            if score > best_score:
                best_score = score
                best_music = music

            if total_notes >= min_total_notes and num_tracks_with_notes >= 2:
                break

        return best_music or muspy.Music(resolution=self.config.resolution)

    def generate_with_prompt(
        self,
        prompt_music: muspy.Music,
        genre_id: int = 0,
        continuation_bars: int = 4,
        temperature: Optional[float] = None,
    ) -> muspy.Music:
        """
        Generate continuation of existing music.

        The model sees the prompt and generates more in the same style,
        with all instruments continuing coherently.

        Args:
            prompt_music: Existing music to continue from
            genre_id: Genre conditioning ID
            continuation_bars: Bars to generate after prompt
            temperature: Sampling temperature

        Returns:
            muspy.Music with prompt + generated continuation
        """
        import tensorflow as tf

        # Encode the prompt
        prompt_encoded = self.encoder.encode_music(
            prompt_music,
            genre_id=genre_id,
            max_length=self.config.max_length // 2,
        )

        # Remove EOS and padding from prompt
        prompt_tokens = prompt_encoded.token_ids
        prompt_tokens = prompt_tokens[prompt_tokens != self.encoder.eos_token_id]
        prompt_tokens = prompt_tokens[prompt_tokens != self.encoder.pad_token_id]

        # Generate continuation
        prompt_tensor = self._prepare_start_tokens(prompt_tokens)

        continuation_length = continuation_bars * 50
        total_length = min(len(prompt_tokens) + continuation_length, self.config.max_length)

        tokens = self.model.generate(
            start_tokens=prompt_tensor,
            max_length=total_length,
            temperature=temperature or self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            eos_token_id=self.encoder.eos_token_id,
        )

        tokens = self._extract_tokens(tokens)

        return self.encoder.decode_to_music(
            tokens,
            resolution=self.config.resolution,
            tempo=self.config.tempo,
        )

    def generate_extended(
        self,
        genre_id: int = 0,
        instruments: Optional[List[int]] = None,
        num_segments: int = 3,
        segment_length: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> muspy.Music:
        """
        Generate longer music by concatenating multiple generations.

        Args:
            genre_id: Genre conditioning ID
            instruments: Instrument hints
            num_segments: Number of segments to concatenate
            segment_length: Tokens per segment
            temperature: Sampling temperature

        Returns:
            muspy.Music with concatenated segments
        """
        segment_length = segment_length or self.config.max_length

        all_tracks: Dict[int, List[muspy.Note]] = {}
        total_time_offset = 0

        for seg_idx in range(num_segments):
            tokens = self.generate_tokens(
                genre_id=genre_id,
                instruments=instruments,
                max_length=segment_length,
                temperature=temperature,
            )

            music = self.encoder.decode_to_music(
                tokens,
                resolution=self.config.resolution,
                tempo=self.config.tempo,
            )

            # Get segment duration
            segment_duration = 0
            for track in music.tracks:
                if track.notes:
                    track_end = max(n.time + n.duration for n in track.notes)
                    segment_duration = max(segment_duration, track_end)

            # Add notes with time offset
            for track in music.tracks:
                program = 128 if track.is_drum else track.program

                if program not in all_tracks:
                    all_tracks[program] = []

                for note in track.notes:
                    offset_note = muspy.Note(
                        time=note.time + total_time_offset,
                        pitch=note.pitch,
                        duration=note.duration,
                        velocity=note.velocity,
                    )
                    all_tracks[program].append(offset_note)

            total_time_offset += segment_duration

        # Build final music
        final_music = muspy.Music(
            resolution=self.config.resolution,
            tempos=[muspy.Tempo(time=0, qpm=self.config.tempo)],
            tracks=[],
        )

        for program, notes in all_tracks.items():
            is_drum = (program == 128)
            track = muspy.Track(
                program=0 if is_drum else program,
                is_drum=is_drum,
                name="Drums" if is_drum else f"Program {program}",
                notes=sorted(notes, key=lambda n: n.time),
            )
            final_music.tracks.append(track)

        return final_music


# Backwards compatibility aliases
MultiTrackGenerator = PolyGenerator
MultiTrackConfig = PolyGeneratorConfig
