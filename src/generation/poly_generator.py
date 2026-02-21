"""
Polyphonic Multi-Track Music Generator.

Generates all instruments simultaneously using interleaved encoding.
All instruments are aware of each other during generation.

Author: Murilo de Freitas Spinelli
"""

import math
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
        ignore_eos: bool = False,
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
            ignore_eos: If True, suppress EOS stopping so generation runs to
                        max_length. Use this when a specific bar count is required.

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
        eos_id = None if ignore_eos else self.encoder.eos_token_id
        tokens = self.model.generate(
            start_tokens=start_tokens,
            max_length=params['max_length'],
            temperature=params['temperature'],
            top_k=params['top_k'],
            top_p=params['top_p'],
            eos_token_id=eos_id,
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

        # Interleaved multitrack sequences are dense: ~200-300 tokens per bar.
        # Ignore EOS so the model is forced to fill the full token budget,
        # then trim the decoded music to exactly num_bars afterward.
        ticks_per_bar = self.config.resolution * 4
        estimated_length = min(num_bars * 250, self.config.max_length)

        best_music = None
        best_score = 0

        for attempt in range(max_retries):
            tokens = self.generate_tokens(
                genre_id=genre_id,
                instruments=instruments,
                max_length=estimated_length,
                temperature=temperature,
                ignore_eos=True,
            )

            music = self.encoder.decode_to_music(
                tokens,
                resolution=self.config.resolution,
                tempo=self.config.tempo,
            )

            # Hard-trim to the requested bar count
            music = self._trim_to_bars(music, num_bars, ticks_per_bar)

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

        # Generate continuation — same token budget as generate_music (~250/bar),
        # and suppress EOS so the model is forced to fill the requested bars.
        prompt_tensor = self._prepare_start_tokens(prompt_tokens)

        continuation_length = continuation_bars * 250
        total_length = min(len(prompt_tokens) + continuation_length, self.config.max_length)

        tokens = self.model.generate(
            start_tokens=prompt_tensor,
            max_length=total_length,
            temperature=temperature or self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            eos_token_id=None,
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

            # Get segment duration, snapped up to the next full bar boundary
            # so every segment starts exactly on a bar line.
            ticks_per_bar = self.config.resolution * 4
            segment_duration = 0
            for track in music.tracks:
                if track.notes:
                    track_end = max(n.time + n.duration for n in track.notes)
                    segment_duration = max(segment_duration, track_end)
            if segment_duration > 0:
                bars = math.ceil(segment_duration / ticks_per_bar)
                segment_duration = bars * ticks_per_bar

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

    def generate_continuation(
        self,
        total_bars: int = 32,
        initial_bars: int = 16,
        chunk_bars: int = 4,
        context_bars: int = 8,
        genre_id: int = 0,
        instruments: Optional[List[int]] = None,
        temperature: Optional[float] = None,
    ) -> muspy.Music:
        """
        Generate a long piece by extending in aligned chunks with sliding context.

        Generates an initial seed, then repeatedly feeds the last N bars as
        a prompt to produce the next chunk. Each chunk is bar-aligned so the
        piece stays in a 4-by-4 phrase structure.

        Args:
            total_bars: Target total length of the piece in bars
            initial_bars: Bars to generate in the initial seed
            chunk_bars: How many new bars to generate per extension step
            context_bars: How many bars of history the model sees as prompt
            genre_id: Genre conditioning ID
            instruments: Instrument hints
            temperature: Sampling temperature

        Returns:
            muspy.Music with the full extended piece
        """
        ticks_per_bar = self.config.resolution * 4

        # A single model pass can fit roughly max_length // 250 bars.
        # Cap the seed to that so generate_music always fills its full budget,
        # then the continuation loop extends to initial_bars and beyond to total_bars.
        _TOKENS_PER_BAR = 250
        max_safe_seed_bars = max(4, self.config.max_length // _TOKENS_PER_BAR)
        seed_bars = min(initial_bars, max_safe_seed_bars)

        seed_music = self.generate_music(
            genre_id=genre_id,
            instruments=instruments,
            num_bars=seed_bars,
            temperature=temperature,
        )

        current_music = seed_music

        # Measure actual seed length — the loop handles everything from here to total_bars
        all_ends = [n.time + n.duration for t in current_music.tracks for n in t.notes]
        if all_ends:
            current_bars = math.ceil(max(all_ends) / ticks_per_bar)
        else:
            current_bars = 0

        while current_bars < total_bars:
            # Extract the last `context_bars` bars as the prompt
            prompt = self._extract_last_n_bars(current_music, context_bars, ticks_per_bar)

            # Generate continuation (model sees prompt, produces chunk_bars more)
            continued = self.generate_with_prompt(
                prompt_music=prompt,
                genre_id=genre_id,
                continuation_bars=chunk_bars,
                temperature=temperature,
            )

            # Slice out only the newly generated notes (after the prompt region)
            context_ticks = context_bars * ticks_per_bar
            new_notes = self._extract_notes_after(continued, context_ticks)

            # Append new notes to the main piece at the correct time offset
            append_offset = current_bars * ticks_per_bar
            self._append_notes(current_music, new_notes, append_offset, context_ticks)

            current_bars += chunk_bars

        return current_music

    @staticmethod
    def _trim_to_bars(
        music: muspy.Music,
        num_bars: int,
        ticks_per_bar: int,
    ) -> muspy.Music:
        """Remove notes that start at or after num_bars, clip durations that overshoot."""
        cutoff = num_bars * ticks_per_bar
        for track in music.tracks:
            kept = []
            for n in track.notes:
                if n.time >= cutoff:
                    continue
                if n.time + n.duration > cutoff:
                    n = muspy.Note(
                        time=n.time,
                        pitch=n.pitch,
                        duration=cutoff - n.time,
                        velocity=n.velocity,
                    )
                kept.append(n)
            track.notes = kept
        return music

    @staticmethod
    def _extract_last_n_bars(
        music: muspy.Music,
        n_bars: int,
        ticks_per_bar: int,
    ) -> muspy.Music:
        """Extract only the last n_bars from a Music object, rebased to time 0."""
        all_ends = [n.time + n.duration for t in music.tracks for n in t.notes]
        if not all_ends:
            return music

        total_ticks = max(all_ends)
        total_bars = total_ticks / ticks_per_bar
        cut_bar = max(0, int(total_bars) - n_bars)
        cut_tick = cut_bar * ticks_per_bar

        new_tracks = []
        for track in music.tracks:
            new_notes = [
                muspy.Note(
                    time=int(n.time - cut_tick),
                    pitch=n.pitch,
                    duration=n.duration,
                    velocity=n.velocity,
                )
                for n in track.notes
                if n.time >= cut_tick
            ]
            new_tracks.append(muspy.Track(
                program=track.program,
                is_drum=track.is_drum,
                name=track.name,
                notes=sorted(new_notes, key=lambda n: n.time),
            ))

        return muspy.Music(
            resolution=music.resolution,
            tempos=music.tempos,
            tracks=new_tracks,
        )

    @staticmethod
    def _extract_notes_after(
        music: muspy.Music,
        after_tick: int,
    ) -> Dict[tuple, list]:
        """Get notes starting at or after after_tick, grouped by (program, is_drum)."""
        result = {}
        for track in music.tracks:
            key = (track.program, track.is_drum)
            notes = [n for n in track.notes if n.time >= after_tick]
            if notes:
                result[key] = (track, notes)
        return result

    @staticmethod
    def _append_notes(
        music: muspy.Music,
        new_notes_by_track: Dict[tuple, list],
        append_offset: int,
        context_ticks: int,
    ) -> None:
        """Append new notes to existing music, shifted to the correct global time."""
        for (program, is_drum), (src_track, notes) in new_notes_by_track.items():
            # Find the matching track in the existing piece
            target = None
            for track in music.tracks:
                if track.program == program and track.is_drum == is_drum:
                    target = track
                    break

            if target is None:
                target = muspy.Track(
                    program=program,
                    is_drum=is_drum,
                    name=src_track.name,
                    notes=[],
                )
                music.tracks.append(target)

            for note in notes:
                target.notes.append(muspy.Note(
                    time=int(note.time - context_ticks + append_offset),
                    pitch=note.pitch,
                    duration=note.duration,
                    velocity=note.velocity,
                ))

            target.notes.sort(key=lambda n: n.time)


# Backwards compatibility aliases
MultiTrackGenerator = PolyGenerator
MultiTrackConfig = PolyGeneratorConfig
