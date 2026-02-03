"""
Abstract base class for music generators.

Provides common functionality for both monophonic and polyphonic generators.

Author: Murilo de Freitas Spinelli
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import muspy

if TYPE_CHECKING:
    from ..models.base_model import BaseMusicModel
    from ..data.encoders.base_encoder import BaseEncoder
    from ..models.model_bundle import ModelBundle


@dataclass
class GeneratorConfig:
    """Base configuration for music generation."""

    # Sequence length
    max_length: int = 2048

    # Sampling parameters
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.92

    # MIDI output
    resolution: int = 24  # Ticks per beat
    tempo: float = 120.0  # BPM

    # Quality thresholds
    min_notes: int = 10
    min_bars: int = 4
    max_retries: int = 3

    # Random seed
    seed: Optional[int] = None


class BaseGenerator(ABC):
    """
    Abstract base class for music generators.

    Provides common functionality for token generation, MIDI export,
    and statistics. Subclasses implement specific generation strategies.
    """

    def __init__(
        self,
        model: "BaseMusicModel",
        encoder: "BaseEncoder",
        config: Optional[GeneratorConfig] = None,
    ):
        """
        Initialize generator.

        Args:
            model: Trained model (TransformerModel or LSTMModel)
            encoder: Encoder instance
            config: Generation configuration
        """
        self.model = model
        self.encoder = encoder
        self.config = config or GeneratorConfig()

    @classmethod
    def from_bundle(
        cls,
        bundle: "ModelBundle",
        config: Optional[GeneratorConfig] = None,
    ) -> "BaseGenerator":
        """
        Create generator from a ModelBundle.

        Args:
            bundle: ModelBundle containing model and encoder
            config: Generation configuration (optional)

        Returns:
            Generator instance
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
        config: Optional[GeneratorConfig] = None,
    ) -> "BaseGenerator":
        """
        Load generator from a saved bundle file.

        Args:
            filepath: Path to the .h5 bundle file
            config: Generation configuration (optional)

        Returns:
            Generator instance
        """
        from ..models.model_bundle import load_model_bundle
        bundle = load_model_bundle(filepath)
        return cls.from_bundle(bundle, config)

    @abstractmethod
    def generate_tokens(self, **kwargs) -> np.ndarray:
        """
        Generate a sequence of tokens.

        Returns:
            Array of generated token IDs
        """
        pass

    @abstractmethod
    def generate_music(self, **kwargs) -> muspy.Music:
        """
        Generate a complete music piece.

        Returns:
            muspy.Music object
        """
        pass

    def generate_midi(
        self,
        output_path: Union[str, Path],
        **kwargs,
    ) -> muspy.Music:
        """
        Generate music and save to MIDI file.

        Args:
            output_path: Path to save MIDI file
            **kwargs: Additional arguments for generate_music

        Returns:
            Generated muspy.Music object
        """
        music = self.generate_music(**kwargs)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        muspy.write_midi(str(output_path), music)

        return music

    def _apply_sampling_defaults(
        self,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Apply default values from config for sampling parameters.

        Returns:
            Dict with resolved parameter values
        """
        resolved_seed = seed if seed is not None else self.config.seed
        if resolved_seed is not None:
            np.random.seed(resolved_seed)

        return {
            'max_length': max_length or self.config.max_length,
            'temperature': temperature if temperature is not None else self.config.temperature,
            'top_k': top_k if top_k is not None else self.config.top_k,
            'top_p': top_p if top_p is not None else self.config.top_p,
            'seed': resolved_seed,
        }

    def _prepare_start_tokens(self, start_tokens: np.ndarray) -> "np.ndarray":
        """
        Prepare start tokens for model input (add batch dimension).

        Args:
            start_tokens: 1D array of token IDs

        Returns:
            2D tensor with batch dimension
        """
        import tensorflow as tf

        tokens = tf.convert_to_tensor(start_tokens, dtype=tf.int32)
        if len(tokens.shape) == 1:
            tokens = tf.expand_dims(tokens, 0)
        return tokens

    def _extract_tokens(self, tokens) -> np.ndarray:
        """
        Extract tokens from model output (remove batch dimension).

        Args:
            tokens: Model output (tensor or array)

        Returns:
            1D numpy array
        """
        if hasattr(tokens, 'numpy'):
            tokens = tokens.numpy()
        if len(tokens.shape) == 2 and tokens.shape[0] == 1:
            tokens = tokens[0]
        return tokens

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
        print("  Generation Results")
        print("=" * 50)
        print(f"  Tracks: {stats['num_tracks']}")
        print(f"  Total Notes: {stats['total_notes']}")
        print(f"  Duration: {stats['duration_bars']} bars")
        print("-" * 50)

        for track in stats['tracks']:
            drum_marker = " [DRUMS]" if track['is_drum'] else ""
            print(f"  {track['name']}{drum_marker}")
            print(f"    Program: {track['program']}, Notes: {track['num_notes']}, Bars: {track['bars']}")

        print("=" * 50)
