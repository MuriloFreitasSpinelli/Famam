"""
Music generator using trained LSTM models.

Generates pianoroll predictions and converts them to MIDI via muspy.
Supports genre and instrument conditioning for style-specific generation.
Can generate individual instrument tracks and combine them into a full piece.
"""

import numpy as np  # type: ignore
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import muspy  # type: ignore

from ..core.model_bundle import ModelBundle
from ..core.vocabulary import Vocabulary, GENERAL_MIDI_INSTRUMENTS


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

    def list_instruments(self) -> List[str]:
        """List all available instruments (General MIDI)."""
        return self.bundle.list_instruments()

    def list_active_instruments(self) -> List[str]:
        """List instruments used in training data."""
        return self.bundle.list_active_instruments()

    def get_instruments_for_genre(self, genre: str) -> List[str]:
        """Get instruments commonly used in a specific genre."""
        return self.bundle.get_instruments_for_genre(genre)

    def get_top_instruments_for_genre(
        self,
        genre: str,
        top_n: int = 3,
        exclude_drums: bool = True
    ) -> List[str]:
        """
        Get top N most frequently used instruments for a genre.

        Args:
            genre: Genre name to get instruments for
            top_n: Number of instruments to return
            exclude_drums: If True, exclude drums from the selection

        Returns:
            List of instrument names sorted by frequency (most frequent first)
        """
        return self.bundle.get_top_instruments_for_genre(genre, top_n, exclude_drums)

    def generate(
        self,
        genre: str,
        instrument: str = "Acoustic Grand Piano",
        seed: Optional[np.ndarray] = None,
        drum_track: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Generate a pianoroll conditioned on genre and instrument.

        Args:
            genre: Genre name for conditioning
            instrument: Instrument name for conditioning (General MIDI name)
            seed: Optional seed pianoroll (128, time_steps). If None, uses zeros.
            drum_track: Optional drum pianoroll for alignment (128, time_steps)
            temperature: Sampling temperature (higher = more random)
            threshold: Threshold for binarizing output (0-1)

        Returns:
            Generated pianoroll array (128, time_steps)
        """
        # Create seed if not provided
        if seed is None:
            seed = np.zeros(self.input_shape, dtype=np.float32)

        # Validate seed shape
        if seed.shape != self.input_shape: # type: ignore
            raise ValueError(
                f"Seed shape {seed.shape} doesn't match expected {self.input_shape}" # type: ignore
            )

        # Get prediction from model
        prediction = self.bundle.predict(
            seed,
            instrument=instrument,
            genre=genre,
            drum_pianoroll=drum_track,
        )

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
        instrument: str = "Acoustic Grand Piano",
        drum_track: Optional[np.ndarray] = None,
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
            instrument: Instrument for conditioning
            drum_track: Optional drum pianoroll for alignment
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
                instrument=instrument,
                seed=current,
                drum_track=drum_track,
                temperature=temperature,
                threshold=threshold,
            )

        return current

    def generate_multiple(
        self,
        genre: str,
        instrument: str = "Acoustic Grand Piano",
        count: int = 4,
        drum_track: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
        use_random_seeds: bool = True,
    ) -> List[np.ndarray]:
        """
        Generate multiple variations of a single instrument.

        Args:
            genre: Genre for conditioning
            instrument: Instrument for conditioning
            count: Number of variations to generate
            drum_track: Optional drum pianoroll for alignment
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
                instrument=instrument,
                seed=seed,
                drum_track=drum_track,
                temperature=temperature,
                threshold=threshold,
            )
            results.append(result)

        return results

    def generate_song(
        self,
        genre: str,
        temperature: float = 1.0,
        threshold: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete song with drums and other instruments.

        This is the main entry point for generating full multi-instrument pieces.
        First generates a drum track, then uses it to align other instrument generation.

        Args:
            genre: Genre for conditioning (e.g., "rock", "jazz", "classical")
            temperature: Sampling temperature (higher = more random/creative)
            threshold: Threshold for binarizing output (0-1)

        Returns:
            Dict mapping instrument names to their generated pianorolls
        """
        tracks: Dict[str, np.ndarray] = {}

        # === Step 1: Generate drum track first for rhythmic foundation ===
        print(f"Generating drum track for '{genre}'...")
        drum_track = self.generate(
            genre=genre,
            instrument="Drums",
            seed=None,
            drum_track=None,
            temperature=temperature,
            threshold=threshold,
        )
        tracks["Drums"] = drum_track

        # === Step 2: Get top 3 most frequently used instruments for this genre ===
        # Uses global instrument frequency to select the most representative instruments
        genre_instruments = self.get_top_instruments_for_genre(
            genre, top_n=3, exclude_drums=True
        )

        # If no instruments found for genre, use sensible defaults
        if not genre_instruments:
            genre_instruments = [
                "Acoustic Grand Piano",
                "Electric Bass (finger)",
                "Acoustic Guitar (steel)",
            ]

        # === Step 3: Generate each instrument aligned to the drum track ===
        for instrument in genre_instruments:
            print(f"Generating {instrument}...")
            instrument_track = self.generate(
                genre=genre,
                instrument=instrument,
                seed=None,
                drum_track=drum_track,
                temperature=temperature,
                threshold=threshold,
            )
            tracks[instrument] = instrument_track

        print(f"Generated {len(tracks)} tracks: {list(tracks.keys())}")
        return tracks

    def generate_song_midi(
        self,
        genre: str,
        output_path: str,
        temperature: float = 1.0,
        threshold: float = 0.5,
        resolution: int = 24,
        tempo: float = 120.0,
    ) -> muspy.Music:
        """
        Generate a complete song and save to MIDI file.

        Args:
            genre: Genre for conditioning
            output_path: Path to save MIDI file
            temperature: Sampling temperature
            threshold: Binarization threshold
            resolution: Ticks per beat
            tempo: Tempo in BPM

        Returns:
            Generated muspy.Music object with all tracks
        """
        # Generate all tracks
        tracks = self.generate_song(
            genre=genre,
            temperature=temperature,
            threshold=threshold,
        )

        # Combine into Music object
        music = self.combine_tracks_to_music(
            tracks=tracks,
            resolution=resolution,
            tempo=tempo,
        )

        # Save to MIDI
        output_path = Path(output_path) # type: ignore
        output_path.parent.mkdir(parents=True, exist_ok=True) # type: ignore
        music.write_midi(str(output_path))

        print(f"Saved song to: {output_path}")
        return music

    def generate_multi_instrument(
        self,
        genre: str,
        instruments: Optional[List[str]] = None,
        drum_track: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
        generate_drums_first: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a multi-instrument piece by generating each instrument track separately.

        Args:
            genre: Genre for conditioning
            instruments: List of instrument names. If None, uses common instruments for genre.
            drum_track: Optional pre-existing drum track. If None and generate_drums_first=True,
                        generates a drum track first for alignment.
            temperature: Sampling temperature
            threshold: Binarization threshold
            generate_drums_first: If True and no drum_track provided, generate drums first

        Returns:
            Dict mapping instrument names to their generated pianorolls
        """
        result_tracks: Dict[str, np.ndarray] = {}

        # Get instruments for this genre if not specified
        if instruments is None:
            instruments = self.get_instruments_for_genre(genre)
            if not instruments:
                # Fall back to common instruments
                instruments = ["Acoustic Grand Piano", "Electric Bass (finger)", "Drums"]

        # Generate drums first if requested and not provided
        if generate_drums_first and drum_track is None and "Drums" in instruments:
            print("Generating drum track for alignment...")
            drum_track = self.generate(
                genre=genre,
                instrument="Drums",
                temperature=temperature,
                threshold=threshold,
            )
            result_tracks["Drums"] = drum_track
            instruments = [i for i in instruments if i != "Drums"]

        # Generate each instrument
        for instrument in instruments:
            print(f"Generating {instrument}...")
            pianoroll = self.generate(
                genre=genre,
                instrument=instrument,
                drum_track=drum_track,
                temperature=temperature,
                threshold=threshold,
            )
            result_tracks[instrument] = pianoroll

        return result_tracks

    def combine_tracks_to_music(
        self,
        tracks: Dict[str, np.ndarray],
        resolution: int = 24,
        tempo: float = 120.0,
    ) -> muspy.Music:
        """
        Combine multiple instrument tracks into a single Music object.

        Args:
            tracks: Dict mapping instrument names to pianorolls
            resolution: Ticks per beat
            tempo: Tempo in BPM

        Returns:
            muspy.Music object with all tracks
        """
        from ..core.vocabulary import INSTRUMENT_NAME_TO_ID

        music = muspy.Music(
            resolution=resolution,
            tempos=[muspy.Tempo(time=0, qpm=tempo)],
        )

        for instrument_name, pianoroll in tracks.items():
            # Get program number
            program = INSTRUMENT_NAME_TO_ID.get(instrument_name, 0)
            is_drum = (instrument_name == "Drums" or program == 128)

            # Convert pianoroll to track
            pianoroll_t = pianoroll.T.astype(bool)
            temp_music = muspy.from_pianoroll_representation(
                pianoroll_t,
                resolution=resolution,
                encode_velocity=False,
            )

            # Add tracks with correct program
            for track in temp_music.tracks:
                track.program = 0 if is_drum else program
                track.is_drum = is_drum
                track.name = instrument_name
                music.tracks.append(track)

        return music

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
        instrument: str = "Acoustic Grand Piano",
        seed: Optional[np.ndarray] = None,
        drum_track: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
        resolution: int = 24,
        tempo: float = 120.0,
    ) -> muspy.Music:
        """
        Generate single-instrument music and save directly to MIDI file.

        Args:
            genre: Genre for conditioning
            output_path: Path to save MIDI file
            instrument: Instrument for conditioning
            seed: Optional seed pianoroll
            drum_track: Optional drum pianoroll for alignment
            temperature: Sampling temperature
            threshold: Binarization threshold
            resolution: Ticks per beat
            tempo: Tempo in BPM

        Returns:
            Generated muspy.Music object
        """
        from ..core.vocabulary import INSTRUMENT_NAME_TO_ID

        # Generate pianoroll
        pianoroll = self.generate(
            genre=genre,
            instrument=instrument,
            seed=seed,
            drum_track=drum_track,
            temperature=temperature,
            threshold=threshold,
        )

        # Get program number
        program = INSTRUMENT_NAME_TO_ID.get(instrument, 0)

        # Convert to Music
        music = self.to_music(
            pianoroll,
            resolution=resolution,
            tempo=tempo,
            program=program,
        )

        # Save to MIDI
        output_path = Path(output_path) # type: ignore
        output_path.parent.mkdir(parents=True, exist_ok=True) # type: ignore
        music.write_midi(str(output_path))

        print(f"Saved MIDI to: {output_path}")
        return music

    def generate_multi_instrument_midi(
        self,
        genre: str,
        output_path: str,
        instruments: Optional[List[str]] = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
        resolution: int = 24,
        tempo: float = 120.0,
        generate_drums_first: bool = True,
    ) -> muspy.Music:
        """
        Generate multi-instrument music and save to MIDI file.

        Args:
            genre: Genre for conditioning
            output_path: Path to save MIDI file
            instruments: List of instruments. If None, uses genre-appropriate instruments.
            temperature: Sampling temperature
            threshold: Binarization threshold
            resolution: Ticks per beat
            tempo: Tempo in BPM
            generate_drums_first: If True, generate drums first for alignment

        Returns:
            Generated muspy.Music object with all tracks
        """
        # Generate all instrument tracks
        tracks = self.generate_multi_instrument(
            genre=genre,
            instruments=instruments,
            temperature=temperature,
            threshold=threshold,
            generate_drums_first=generate_drums_first,
        )

        # Combine into Music object
        music = self.combine_tracks_to_music(
            tracks=tracks,
            resolution=resolution,
            tempo=tempo,
        )

        # Save to MIDI
        output_path = Path(output_path) # type: ignore
        output_path.parent.mkdir(parents=True, exist_ok=True) # type: ignore
        music.write_midi(str(output_path))

        print(f"Saved multi-instrument MIDI to: {output_path}")
        print(f"  Tracks: {list(tracks.keys())}")
        return music

    def generate_batch_midi(
        self,
        genre: str,
        output_dir: str,
        instrument: str = "Acoustic Grand Piano",
        count: int = 4,
        drum_track: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
        resolution: int = 24,
        tempo: float = 120.0,
        prefix: str = "generated",
    ) -> List[str]:
        """
        Generate multiple MIDI files for a single instrument.

        Args:
            genre: Genre for conditioning
            output_dir: Directory to save MIDI files
            instrument: Instrument for conditioning
            count: Number of files to generate
            drum_track: Optional drum pianoroll for alignment
            temperature: Sampling temperature
            threshold: Binarization threshold
            resolution: Ticks per beat
            tempo: Tempo in BPM
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        from ..core.vocabulary import INSTRUMENT_NAME_TO_ID

        output_dir = Path(output_dir) # type: ignore
        output_dir.mkdir(parents=True, exist_ok=True) # type: ignore

        program = INSTRUMENT_NAME_TO_ID.get(instrument, 0)

        pianorolls = self.generate_multiple(
            genre=genre,
            instrument=instrument,
            count=count,
            drum_track=drum_track,
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

            # Clean instrument name for filename
            inst_clean = instrument.replace(" ", "_").replace("(", "").replace(")", "")
            filepath = output_dir / f"{prefix}_{genre}_{inst_clean}_{i:03d}.mid" # type: ignore
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
    instrument: str = "Acoustic Grand Piano",
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> muspy.Music:
    """
    Convenience function to generate single-instrument MIDI from a saved bundle.

    Args:
        bundle_path: Path to ModelBundle .h5 file
        genre: Genre for conditioning
        output_path: Path to save MIDI
        instrument: Instrument for conditioning
        temperature: Sampling temperature
        threshold: Binarization threshold

    Returns:
        Generated muspy.Music object
    """
    generator = MusicGenerator.from_bundle_path(bundle_path)
    return generator.generate_midi(
        genre=genre,
        output_path=output_path,
        instrument=instrument,
        temperature=temperature,
        threshold=threshold,
    )


def generate_multi_instrument_from_bundle(
    bundle_path: str,
    genre: str,
    output_path: str,
    instruments: Optional[List[str]] = None,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> muspy.Music:
    """
    Convenience function to generate multi-instrument MIDI from a saved bundle.

    Args:
        bundle_path: Path to ModelBundle .h5 file
        genre: Genre for conditioning
        output_path: Path to save MIDI
        instruments: List of instruments to generate. If None, uses genre-appropriate instruments.
        temperature: Sampling temperature
        threshold: Binarization threshold

    Returns:
        Generated muspy.Music object with multiple tracks
    """
    generator = MusicGenerator.from_bundle_path(bundle_path)
    return generator.generate_multi_instrument_midi(
        genre=genre,
        output_path=output_path,
        instruments=instruments,
        temperature=temperature,
        threshold=threshold,
    )


def generate_song_from_bundle(
    bundle_path: str,
    genre: str,
    output_path: str,
    temperature: float = 1.0,
) -> muspy.Music:
    """
    Convenience function to generate a complete song from a saved bundle.

    This is the simplest way to generate music - just provide genre and temperature.
    Drums are generated first, then other instruments are generated aligned to the drums.

    Args:
        bundle_path: Path to ModelBundle .h5 file
        genre: Genre for conditioning (e.g., "rock", "jazz", "classical")
        output_path: Path to save MIDI
        temperature: Sampling temperature (higher = more creative)

    Returns:
        Generated muspy.Music object with drums and instruments
    """
    generator = MusicGenerator.from_bundle_path(bundle_path)
    return generator.generate_song_midi(
        genre=genre,
        output_path=output_path,
        temperature=temperature,
    )