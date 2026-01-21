"""
Music generator using trained LSTM models.

Generates pianoroll predictions and converts them to MIDI via muspy.
Supports genre conditioning for style-specific generation.
"""

import numpy as np  # type: ignore
from typing import Optional, List, Tuple
from pathlib import Path

import muspy  # type: ignore

from ..core.model_bundle import ModelBundle
from ..core.vocabulary import Vocabulary


class MusicGenerator:
    """
    Generate music using a trained LSTM model.

    Uses ModelBundle which contains the trained model, vocabulary,
    and configuration needed for generation.
    """

    def __init__(self, bundle: ModelBundle):
        """
        Initialize generator with a trained model bundle.

        Args:
            bundle: ModelBundle containing model, vocabulary, and config
        """
        self.bundle = bundle
        self.model = bundle.model
        self.vocabulary = bundle.vocabulary
        self.input_shape = bundle.input_shape

    @classmethod
    def from_bundle_path(cls, filepath: str) -> 'MusicGenerator':
        """
        Load generator from saved bundle file.

        Args:
            filepath: Path to the .h5 bundle file

        Returns:
            MusicGenerator instance
        """
        bundle = ModelBundle.load(filepath)
        return cls(bundle)

    def list_genres(self) -> List[str]:
        """List available genres for conditioning."""
        return self.bundle.list_genres()

    def generate(
        self,
        genre: str,
        seed: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Generate a pianoroll conditioned on genre.

        Args:
            genre: Genre name for conditioning
            seed: Optional seed pianoroll (128, time_steps). If None, uses zeros.
            temperature: Sampling temperature (higher = more random)
            threshold: Threshold for binarizing output (0-1)

        Returns:
            Generated pianoroll array (128, time_steps)
        """
        # Create seed if not provided
        if seed is None:
            seed = np.zeros(self.input_shape, dtype=np.float32)

        # Validate seed shape
        if seed.shape != self.input_shape:
            raise ValueError(
                f"Seed shape {seed.shape} doesn't match expected {self.input_shape}"
            )

        # Get prediction from model
        prediction = self.bundle.predict(seed, genre)

        # Apply temperature scaling
        if temperature != 1.0:
            prediction = self._apply_temperature(prediction, temperature)

        # Binarize using threshold
        if threshold > 0:
            prediction = (prediction > threshold).astype(np.float32)

        return prediction

    def generate_continuation(
        self,
        seed: np.ndarray,
        genre: str,
        num_steps: int = 1,
        temperature: float = 1.0,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Generate continuation from a seed pianoroll.

        Iteratively generates new segments, using each output as the next input.

        Args:
            seed: Initial pianoroll (128, time_steps)
            genre: Genre for conditioning
            num_steps: Number of generation iterations
            temperature: Sampling temperature
            threshold: Binarization threshold

        Returns:
            Final generated pianoroll (128, time_steps)
        """
        current = seed.copy()

        for _ in range(num_steps):
            current = self.generate(
                genre=genre,
                seed=current,
                temperature=temperature,
                threshold=threshold,
            )

        return current

    def generate_multiple(
        self,
        genre: str,
        count: int = 4,
        temperature: float = 1.0,
        threshold: float = 0.5,
        use_random_seeds: bool = True,
    ) -> List[np.ndarray]:
        """
        Generate multiple variations.

        Args:
            genre: Genre for conditioning
            count: Number of variations to generate
            temperature: Sampling temperature
            threshold: Binarization threshold
            use_random_seeds: If True, use random noise seeds

        Returns:
            List of generated pianorolls
        """
        results = []

        for i in range(count):
            if use_random_seeds:
                # Random sparse seed (mostly zeros with some notes)
                seed = np.random.random(self.input_shape).astype(np.float32)
                seed = (seed > 0.95).astype(np.float32)  # ~5% notes
            else:
                seed = None

            result = self.generate(
                genre=genre,
                seed=seed,
                temperature=temperature,
                threshold=threshold,
            )
            results.append(result)

        return results

    def _apply_temperature(
        self,
        prediction: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling to predictions."""
        # Clip to avoid log(0)
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

        # Convert to logits, scale, convert back
        logits = np.log(prediction / (1 - prediction))
        scaled_logits = logits / temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def to_music(
        self,
        pianoroll: np.ndarray,
        resolution: int = 24,
        tempo: float = 120.0,
        program: int = 0,
    ) -> muspy.Music:
        """
        Convert pianoroll to muspy Music object.

        Args:
            pianoroll: Pianoroll array (128, time_steps)
            resolution: Ticks per beat
            tempo: Tempo in BPM
            program: MIDI program number (instrument)

        Returns:
            muspy.Music object
        """
        # Transpose to muspy format: (time_steps, 128)
        pianoroll_t = pianoroll.T

        # Convert to bool for muspy (required when encode_velocity=False)
        pianoroll_bool = pianoroll_t.astype(bool)

        # Create Music from pianoroll
        music = muspy.from_pianoroll_representation(
            pianoroll_bool,
            resolution=resolution,
            encode_velocity=False,
        )

        # Set tempo
        music.tempos = [muspy.Tempo(time=0, qpm=tempo)]

        # Set program for all tracks
        for track in music.tracks:
            track.program = program

        return music

    def generate_midi(
        self,
        genre: str,
        output_path: str,
        seed: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
        resolution: int = 24,
        tempo: float = 120.0,
        program: int = 0,
    ) -> muspy.Music:
        """
        Generate music and save directly to MIDI file.

        Args:
            genre: Genre for conditioning
            output_path: Path to save MIDI file
            seed: Optional seed pianoroll
            temperature: Sampling temperature
            threshold: Binarization threshold
            resolution: Ticks per beat
            tempo: Tempo in BPM
            program: MIDI program number

        Returns:
            Generated muspy.Music object
        """
        # Generate pianoroll
        pianoroll = self.generate(
            genre=genre,
            seed=seed,
            temperature=temperature,
            threshold=threshold,
        )

        # Convert to Music
        music = self.to_music(
            pianoroll,
            resolution=resolution,
            tempo=tempo,
            program=program,
        )

        # Save to MIDI
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        music.write_midi(str(output_path))

        print(f"Saved MIDI to: {output_path}")
        return music

    def generate_batch_midi(
        self,
        genre: str,
        output_dir: str,
        count: int = 4,
        temperature: float = 1.0,
        threshold: float = 0.5,
        resolution: int = 24,
        tempo: float = 120.0,
        program: int = 0,
        prefix: str = "generated",
    ) -> List[str]:
        """
        Generate multiple MIDI files.

        Args:
            genre: Genre for conditioning
            output_dir: Directory to save MIDI files
            count: Number of files to generate
            temperature: Sampling temperature
            threshold: Binarization threshold
            resolution: Ticks per beat
            tempo: Tempo in BPM
            program: MIDI program number
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pianorolls = self.generate_multiple(
            genre=genre,
            count=count,
            temperature=temperature,
            threshold=threshold,
        )

        paths = []
        for i, pianoroll in enumerate(pianorolls):
            music = self.to_music(
                pianoroll,
                resolution=resolution,
                tempo=tempo,
                program=program,
            )

            filepath = output_dir / f"{prefix}_{genre}_{i:03d}.mid"
            music.write_midi(str(filepath))
            paths.append(str(filepath))

        print(f"Saved {count} MIDI files to: {output_dir}")
        return paths

    def summary(self) -> str:
        """Get summary of generator configuration."""
        return self.bundle.summary()


def generate_from_bundle(
    bundle_path: str,
    genre: str,
    output_path: str,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> muspy.Music:
    """
    Convenience function to generate MIDI from a saved bundle.

    Args:
        bundle_path: Path to ModelBundle .h5 file
        genre: Genre for conditioning
        output_path: Path to save MIDI
        temperature: Sampling temperature
        threshold: Binarization threshold

    Returns:
        Generated muspy.Music object
    """
    generator = MusicGenerator.from_bundle_path(bundle_path)
    return generator.generate_midi(
        genre=genre,
        output_path=output_path,
        temperature=temperature,
        threshold=threshold,
    )