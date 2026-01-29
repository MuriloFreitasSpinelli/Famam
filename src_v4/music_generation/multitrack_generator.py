"""
Multi-Track Music Generator.

Generates all instruments simultaneously using the interleaved
multi-track encoding. All instruments are aware of each other.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import muspy

from ..data_preprocessing.encoders.multitrack_encoder import MultiTrackEncoder
from ..model_training.architectures.base_model import BaseMusicModel

if TYPE_CHECKING:
    from ..model_training.model_bundle import ModelBundle


@dataclass
class MultiTrackConfig:
    """Configuration for multi-track generation."""

    # Sequence parameters
    max_length: int = 2048

    # Sampling parameters
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.92

    # Music parameters
    resolution: int = 24
    tempo: float = 120.0
    num_bars: int = 8

    # Quality thresholds
    min_notes_per_track: int = 5
    min_total_notes: int = 20
    max_retries: int = 3

    # Random seed
    seed: Optional[int] = None


class MultiTrackGenerator:
    """
    Generate music with all tracks aware of each other.

    Uses interleaved encoding where all instruments are generated
    together in a single sequence, sorted by time. The model sees
    what every instrument plays at each moment.
    """

    def __init__(
        self,
        model: BaseMusicModel,
        encoder: MultiTrackEncoder,
        config: Optional[MultiTrackConfig] = None,
    ):
        """
        Initialize multi-track generator.

        Args:
            model: Trained model (TransformerModel or LSTMModel)
            encoder: MultiTrackEncoder instance
            config: Generation configuration
        """
        self.model = model
        self.encoder = encoder
        self.config = config or MultiTrackConfig()

    @classmethod
    def from_bundle(
        cls,
        bundle: "ModelBundle",
        config: Optional[MultiTrackConfig] = None,
    ) -> "MultiTrackGenerator":
        """Create generator from ModelBundle."""
        return cls(
            model=bundle.model,
            encoder=bundle.encoder,
            config=config,
        )

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

        # Use config defaults
        max_length = max_length or self.config.max_length
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        seed = seed if seed is not None else self.config.seed

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # Create start tokens
        start_tokens = self.encoder.create_conditioning_tokens(
            genre_id=genre_id,
            instruments=instruments,
        )

        # Add initial bar token to help the model
        start_tokens = np.append(start_tokens, self.encoder.bar_token(0))

        # Convert to tensor with batch dimension
        start_tokens = tf.convert_to_tensor(start_tokens, dtype=tf.int32)
        if len(start_tokens.shape) == 1:
            start_tokens = tf.expand_dims(start_tokens, 0)

        # Generate
        tokens = self.model.generate(
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.encoder.eos_token_id,
        )

        # Remove batch dimension
        if hasattr(tokens, 'numpy'):
            tokens = tokens.numpy()
        if len(tokens.shape) == 2:
            tokens = tokens[0]

        return tokens

    def generate_music(
        self,
        genre_id: int = 0,
        instruments: Optional[List[int]] = None,
        num_bars: Optional[int] = None,
        temperature: Optional[float] = None,
        min_notes_per_track: Optional[int] = None,
        min_total_notes: Optional[int] = None,
        max_retries: Optional[int] = None,
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
        # Rough estimate: ~20-30 tokens per bar per instrument
        estimated_length = min(num_bars * 100, self.config.max_length)

        best_music = None
        best_score = 0

        for attempt in range(max_retries):
            # Generate tokens
            tokens = self.generate_tokens(
                genre_id=genre_id,
                instruments=instruments,
                max_length=estimated_length,
                temperature=temperature,
            )

            # Decode to music
            music = self.encoder.decode_to_music(
                tokens,
                resolution=self.config.resolution,
                tempo=self.config.tempo,
            )

            # Score this attempt
            total_notes = sum(len(t.notes) for t in music.tracks)
            num_tracks_with_notes = sum(1 for t in music.tracks if len(t.notes) >= min_notes_per_track)

            score = total_notes + (num_tracks_with_notes * 50)

            if score > best_score:
                best_score = score
                best_music = music

            # Check if meets thresholds
            if total_notes >= min_total_notes and num_tracks_with_notes >= 2:
                break

        return best_music or muspy.Music(resolution=self.config.resolution)

    def generate_midi(
        self,
        output_path: Union[str, Path],
        genre_id: int = 0,
        instruments: Optional[List[int]] = None,
        **kwargs,
    ) -> muspy.Music:
        """
        Generate music and save to MIDI file.

        Args:
            output_path: Path to save MIDI
            genre_id: Genre conditioning ID
            instruments: Instrument hints
            **kwargs: Additional generation arguments

        Returns:
            Generated muspy.Music object
        """
        music = self.generate_music(
            genre_id=genre_id,
            instruments=instruments,
            **kwargs,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        muspy.write_midi(str(output_path), music)

        return music

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
        prompt_tensor = tf.convert_to_tensor(prompt_tokens, dtype=tf.int32)
        prompt_tensor = tf.expand_dims(prompt_tensor, 0)

        continuation_length = continuation_bars * 50  # Estimate
        total_length = min(len(prompt_tokens) + continuation_length, self.config.max_length)

        tokens = self.model.generate(
            start_tokens=prompt_tensor,
            max_length=total_length,
            temperature=temperature or self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            eos_token_id=self.encoder.eos_token_id,
        )

        if hasattr(tokens, 'numpy'):
            tokens = tokens.numpy()
        if len(tokens.shape) == 2:
            tokens = tokens[0]

        # Decode full sequence (prompt + continuation)
        return self.encoder.decode_to_music(
            tokens,
            resolution=self.config.resolution,
            tempo=self.config.tempo,
        )

    def get_generation_stats(self, music: muspy.Music) -> Dict:
        """
        Get statistics about generated music.

        Args:
            music: Generated muspy.Music

        Returns:
            Dict with statistics
        """
        ticks_per_bar = self.config.resolution * 4

        stats = {
            'num_tracks': len(music.tracks),
            'total_notes': sum(len(t.notes) for t in music.tracks),
            'tracks': [],
        }

        for track in music.tracks:
            if track.notes:
                max_time = max(n.time + n.duration for n in track.notes)
                bars = max_time / ticks_per_bar
            else:
                bars = 0

            track_stats = {
                'name': track.name,
                'program': track.program,
                'is_drum': track.is_drum,
                'num_notes': len(track.notes),
                'bars': round(bars, 1),
            }
            stats['tracks'].append(track_stats)

        # Overall duration
        if stats['total_notes'] > 0:
            all_end_times = [
                n.time + n.duration
                for t in music.tracks
                for n in t.notes
            ]
            stats['duration_bars'] = round(max(all_end_times) / ticks_per_bar, 1)
        else:
            stats['duration_bars'] = 0

        return stats

    def print_generation_stats(self, music: muspy.Music) -> None:
        """Print formatted generation statistics."""
        stats = self.get_generation_stats(music)

        print("\n" + "=" * 50)
        print("  Multi-Track Generation Results")
        print("=" * 50)
        print(f"  Tracks: {stats['num_tracks']}")
        print(f"  Total Notes: {stats['total_notes']}")
        print(f"  Duration: {stats['duration_bars']} bars")
        print("-" * 50)

        for track in stats['tracks']:
            drum_marker = " [DRUMS]" if track['is_drum'] else ""
            print(f"  {track['name']}{drum_marker}")
            print(f"    Notes: {track['num_notes']}, Bars: {track['bars']}")

        print("=" * 50)
