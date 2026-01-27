"""
Music generator using trained LSTM and Transformer models.

Generates pianoroll predictions and converts them to MIDI via muspy.
Supports genre and instrument conditioning for style-specific generation.
Can generate individual instrument tracks and combine them into a full piece.

For Transformer models, uses autoregressive token generation with various
sampling strategies (temperature, top-k, top-p).
"""

import numpy as np  # type: ignore
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path

import tensorflow as tf  # type: ignore
import muspy  # type: ignore

from ..core.model_bundle import ModelBundle, TransformerModelBundle, load_model_bundle
from ..core.vocabulary import Vocabulary, GENERAL_MIDI_INSTRUMENTS
from ..core.event_vocabulary import EventVocabulary


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
        use_dynamic_threshold: bool = True,
        seed_density: float = 0.02,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Generate a pianoroll conditioned on genre and instrument.

        Args:
            genre: Genre name for conditioning
            instrument: Instrument name for conditioning (General MIDI name)
            seed: Optional seed pianoroll (128, time_steps). If None, generates random sparse seed.
            drum_track: Optional drum pianoroll for alignment (128, time_steps)
            temperature: Sampling temperature (higher = more random)
            threshold: Threshold for binarizing output (0-1). If use_dynamic_threshold
                       is True, this is used as a percentile target instead.
            use_dynamic_threshold: If True, use adaptive thresholding based on the
                                   actual prediction distribution to ensure notes are generated.
            seed_density: Density of random seed notes (0.0 = zeros, 0.02 = 2% random notes)
            verbose: If True, print diagnostic info about predictions.

        Returns:
            Generated pianoroll array (128, time_steps)
        """
        is_drum = (instrument == "Drums")

        # Create seed if not provided
        if seed is None:
            if seed_density > 0:
                # Use rhythmic pulse seed - structured pattern instead of random noise
                # This creates a more musical foundation for the model to build on
                seed = self._create_rhythmic_seed(seed_density)
            else:
                seed = np.zeros(self.input_shape, dtype=np.float32)

        # Validate seed shape
        if seed.shape != self.input_shape: # type: ignore
            raise ValueError(
                f"Seed shape {seed.shape} doesn't match expected {self.input_shape}" # type: ignore
            )

        if verbose:
            seed_notes = int(seed.sum())
            print(f"  Seed: {seed_notes} note activations across full time range")

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

        if verbose:
            print(f"  Raw prediction stats - min: {prediction.min():.4f}, "
                  f"max: {prediction.max():.4f}, mean: {prediction.mean():.4f}")

        # Binarize using threshold
        # Use higher density for drums to ensure consistent patterns
        target_density = 0.08 if is_drum else 0.03

        if use_dynamic_threshold:
            # Use dynamic threshold: take top N% of activations as notes
            dynamic_thresh = np.percentile(prediction, 100 * (1 - target_density))
            # Ensure threshold is meaningful (at least 10% of max value)
            min_threshold = 0.1 * prediction.max()
            effective_threshold = max(dynamic_thresh, min_threshold)
            if verbose:
                print(f"  Dynamic threshold: {effective_threshold:.4f} "
                      f"(targeting top {target_density*100:.0f}% of activations)")
            prediction = (prediction > effective_threshold).astype(np.float32)
        elif threshold > 0:
            prediction = (prediction > threshold).astype(np.float32)

        # Post-processing to reduce noise and create cleaner output
        # 1. Remove isolated single-timestep notes (noise)
        prediction = self._remove_isolated_notes(prediction, min_neighbors=1)

        # 2. Merge short choppy notes into sustained notes
        if is_drum:
            # For drums: shorter notes are OK, but ensure coverage
            prediction = self._merge_short_notes(prediction, min_note_length=2, max_gap=1)
            prediction = self._ensure_drum_coverage(prediction, min_density=0.03)
        else:
            # For melodic instruments: longer sustained notes
            prediction = self._merge_short_notes(prediction, min_note_length=6, max_gap=3)

        # 3. Limit simultaneous notes to reduce noise
        max_simultaneous = 8 if is_drum else 4
        prediction = self._limit_simultaneous_notes(prediction, max_notes=max_simultaneous)

        if verbose:
            note_count = int(prediction.sum())
            print(f"  Generated {note_count} note activations (after cleanup)")

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
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete song with drums and other instruments.

        This is the main entry point for generating full multi-instrument pieces.
        First generates a drum track, then uses it to align other instrument generation.

        Args:
            genre: Genre for conditioning (e.g., "rock", "jazz", "classical")
            temperature: Sampling temperature (higher = more random/creative)
            threshold: Threshold for binarizing output (0-1)
            verbose: If True, print diagnostic info during generation

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
            verbose=verbose,
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
                verbose=verbose,
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
        verbose: bool = True,
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
            verbose: If True, print diagnostic info during generation

        Returns:
            Generated muspy.Music object with all tracks
        """
        # Generate all tracks
        tracks = self.generate_song(
            genre=genre,
            temperature=temperature,
            threshold=threshold,
            verbose=verbose,
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

    def generate_extended_song(
        self,
        genre: str,
        num_segments: int = 4,
        temperature: float = 1.0,
        threshold: float = 0.5,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate an extended song by creating multiple segments and concatenating them.

        Each segment uses the end of the previous segment as a seed for continuity.

        Args:
            genre: Genre for conditioning
            num_segments: Number of segments to generate (total length = num_segments * max_time_steps)
            temperature: Sampling temperature
            threshold: Binarization threshold
            verbose: Print progress info

        Returns:
            Dict mapping instrument names to their concatenated pianorolls
        """
        # Get instruments for this genre
        genre_instruments = self.get_top_instruments_for_genre(
            genre, top_n=3, exclude_drums=True
        )
        if not genre_instruments:
            genre_instruments = [
                "Acoustic Grand Piano",
                "Electric Bass (finger)",
                "Acoustic Guitar (steel)",
            ]

        all_instruments = ["Drums"] + genre_instruments
        extended_tracks: Dict[str, List[np.ndarray]] = {inst: [] for inst in all_instruments}

        # Generate first segment (no seed)
        if verbose:
            print(f"Generating segment 1/{num_segments}...")

        # Generate drums first
        drum_segment = self.generate(
            genre=genre,
            instrument="Drums",
            seed=None,
            drum_track=None,
            temperature=temperature,
            threshold=threshold,
            verbose=verbose,
        )
        extended_tracks["Drums"].append(drum_segment)

        # Generate other instruments aligned to drums
        for instrument in genre_instruments:
            if verbose:
                print(f"  {instrument}...")
            segment = self.generate(
                genre=genre,
                instrument=instrument,
                seed=None,
                drum_track=drum_segment,
                temperature=temperature,
                threshold=threshold,
                verbose=verbose,
            )
            extended_tracks[instrument].append(segment)

        # Generate subsequent segments using end of previous as seed
        time_steps = self.input_shape[1]
        seed_length = time_steps // 4  # Use last 25% as seed for continuity

        for seg_idx in range(1, num_segments):
            if verbose:
                print(f"\nGenerating segment {seg_idx + 1}/{num_segments}...")

            # Use end of previous drum segment as seed
            prev_drum = extended_tracks["Drums"][-1]
            drum_seed = np.zeros(self.input_shape, dtype=np.float32)
            drum_seed[:, :seed_length] = prev_drum[:, -seed_length:]

            drum_segment = self.generate(
                genre=genre,
                instrument="Drums",
                seed=drum_seed,
                drum_track=None,
                temperature=temperature,
                threshold=threshold,
                verbose=verbose,
            )
            extended_tracks["Drums"].append(drum_segment)

            # Generate other instruments
            for instrument in genre_instruments:
                if verbose:
                    print(f"  {instrument}...")

                prev_segment = extended_tracks[instrument][-1]
                inst_seed = np.zeros(self.input_shape, dtype=np.float32)
                inst_seed[:, :seed_length] = prev_segment[:, -seed_length:]

                segment = self.generate(
                    genre=genre,
                    instrument=instrument,
                    seed=inst_seed,
                    drum_track=drum_segment,
                    temperature=temperature,
                    threshold=threshold,
                    verbose=verbose,
                )
                extended_tracks[instrument].append(segment)

        # Concatenate all segments
        result: Dict[str, np.ndarray] = {}
        for instrument, segments in extended_tracks.items():
            result[instrument] = np.concatenate(segments, axis=1)

        if verbose:
            total_steps = num_segments * time_steps
            print(f"\nGenerated extended song: {total_steps} time steps ({num_segments} segments)")

        return result

    def generate_extended_song_midi(
        self,
        genre: str,
        output_path: str,
        num_segments: int = 4,
        temperature: float = 1.0,
        threshold: float = 0.5,
        resolution: int = 24,
        tempo: float = 120.0,
        verbose: bool = True,
    ) -> muspy.Music:
        """
        Generate an extended song and save to MIDI.

        Args:
            genre: Genre for conditioning
            output_path: Path to save MIDI file
            num_segments: Number of segments (total length multiplier)
            temperature: Sampling temperature
            threshold: Binarization threshold
            resolution: Ticks per beat
            tempo: Tempo in BPM
            verbose: Print progress info

        Returns:
            Generated muspy.Music object
        """
        tracks = self.generate_extended_song(
            genre=genre,
            num_segments=num_segments,
            temperature=temperature,
            threshold=threshold,
            verbose=verbose,
        )

        # Combine into Music object
        music = self.combine_tracks_to_music(
            tracks=tracks,
            resolution=resolution,
            tempo=tempo,
        )

        # Save to MIDI
        output_path = Path(output_path)  # type: ignore
        output_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        music.write_midi(str(output_path))

        print(f"Saved extended song to: {output_path}")
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

    def _merge_short_notes(
        self,
        pianoroll: np.ndarray,
        min_note_length: int = 4,
        max_gap: int = 2,
    ) -> np.ndarray:
        """
        Merge short choppy notes into longer sustained notes.

        This fixes the issue where the model outputs sparse activations
        that get converted to many tiny notes instead of musical phrases.

        Args:
            pianoroll: Binary pianoroll (128, time_steps)
            min_note_length: Minimum note length in time steps. Shorter notes are extended.
            max_gap: Maximum gap (in time steps) to bridge between notes of same pitch.

        Returns:
            Cleaned pianoroll with merged notes
        """
        num_pitches, time_steps = pianoroll.shape
        result = pianoroll.copy()

        for pitch in range(num_pitches):
            row = result[pitch]

            # Find note onsets and offsets
            # Pad to detect edges
            padded = np.concatenate([[0], row, [0]])
            onsets = np.where(np.diff(padded) == 1)[0]
            offsets = np.where(np.diff(padded) == -1)[0]

            if len(onsets) == 0:
                continue

            # First pass: extend short notes to minimum length
            for onset, offset in zip(onsets, offsets):
                note_length = offset - onset
                if note_length < min_note_length:
                    # Extend note to minimum length
                    new_offset = min(onset + min_note_length, time_steps)
                    result[pitch, onset:new_offset] = 1.0

            # Second pass: bridge small gaps between notes
            row = result[pitch]
            padded = np.concatenate([[0], row, [0]])
            onsets = np.where(np.diff(padded) == 1)[0]
            offsets = np.where(np.diff(padded) == -1)[0]

            for i in range(len(offsets) - 1):
                gap = onsets[i + 1] - offsets[i]
                if gap <= max_gap:
                    # Bridge the gap
                    result[pitch, offsets[i]:onsets[i + 1]] = 1.0

        return result

    def _remove_isolated_notes(
        self,
        pianoroll: np.ndarray,
        min_neighbors: int = 1,
    ) -> np.ndarray:
        """
        Remove isolated note activations that are likely noise.

        A note activation is considered isolated if it doesn't have enough
        neighboring activations (in time or pitch).

        Args:
            pianoroll: Binary pianoroll (128, time_steps)
            min_neighbors: Minimum number of neighbors required to keep a note

        Returns:
            Cleaned pianoroll
        """
        from scipy import ndimage  # type: ignore

        # Create a kernel that counts neighbors (3x3 window excluding center)
        kernel = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        # Count neighbors for each cell
        neighbor_count = ndimage.convolve(pianoroll.astype(float), kernel, mode='constant')

        # Keep only notes with enough neighbors
        result = pianoroll * (neighbor_count >= min_neighbors)

        return result.astype(np.float32)

    def _limit_simultaneous_notes(
        self,
        pianoroll: np.ndarray,
        max_notes: int = 4,
    ) -> np.ndarray:
        """
        Limit the number of simultaneous notes at each time step.

        Keeps the highest-activation notes (or lowest pitches for ties).

        Args:
            pianoroll: Binary pianoroll (128, time_steps)
            max_notes: Maximum simultaneous notes allowed

        Returns:
            Filtered pianoroll
        """
        result = pianoroll.copy()
        num_pitches, time_steps = pianoroll.shape

        for t in range(time_steps):
            active_pitches = np.where(result[:, t] > 0)[0]

            if len(active_pitches) > max_notes:
                # Keep the lowest pitches (bass notes are usually more important)
                # and some higher ones for melody
                pitches_to_keep = np.concatenate([
                    active_pitches[:max_notes // 2],  # Keep lowest
                    active_pitches[-(max_notes - max_notes // 2):]  # Keep highest
                ])
                pitches_to_remove = np.setdiff1d(active_pitches, pitches_to_keep)
                result[pitches_to_remove, t] = 0

        return result

    def _ensure_drum_coverage(
        self,
        pianoroll: np.ndarray,
        min_density: float = 0.05,
    ) -> np.ndarray:
        """
        Ensure drums have adequate coverage throughout the duration.

        Args:
            pianoroll: Drum pianoroll (128, time_steps)
            min_density: Minimum note density to ensure

        Returns:
            Enhanced drum pianoroll
        """
        num_pitches, time_steps = pianoroll.shape
        result = pianoroll.copy()

        # Common drum pitches (General MIDI)
        kick = 36
        snare = 38
        closed_hihat = 42
        open_hihat = 46

        # Check if drums are too sparse
        current_density = result.sum() / (num_pitches * time_steps)

        if current_density < min_density:
            # Add basic drum pattern as foundation
            ticks_per_beat = 24  # Assuming standard resolution
            beats = time_steps // ticks_per_beat

            for beat in range(beats):
                beat_pos = beat * ticks_per_beat

                # Kick on 1 and 3 (every 2 beats in 4/4)
                if beat % 2 == 0:
                    result[kick, beat_pos:beat_pos + 6] = 1.0

                # Snare on 2 and 4
                if beat % 2 == 1:
                    result[snare, beat_pos:beat_pos + 4] = 1.0

                # Hi-hat on every beat
                result[closed_hihat, beat_pos:beat_pos + 2] = 1.0

                # Hi-hat on off-beats too for groove
                half_beat = beat_pos + ticks_per_beat // 2
                if half_beat < time_steps:
                    result[closed_hihat, half_beat:half_beat + 2] = 1.0

        return result

    def _create_rhythmic_seed(self, density: float = 0.02) -> np.ndarray:
        """
        Create a structured rhythmic seed pattern instead of random noise.

        This creates a more musical foundation with:
        - Regular rhythmic pulses on beat positions
        - Notes in common pitch ranges (bass, mid, high)
        - Some variation to trigger different model responses

        Args:
            density: Approximate note density (0.01-0.05 typical)

        Returns:
            Seed pianoroll (128, time_steps)
        """
        num_pitches, time_steps = self.input_shape
        seed = np.zeros((num_pitches, time_steps), dtype=np.float32)

        # Assume 24 ticks per beat (standard resolution)
        ticks_per_beat = 24
        beats = time_steps // ticks_per_beat

        # Create rhythmic pulses every beat and half-beat
        for beat in range(beats):
            beat_pos = beat * ticks_per_beat

            # Strong beat - bass note (kick drum area for drums)
            if beat % 4 == 0:  # Every 4 beats (downbeat)
                # Bass range: 36-48 (C2-C3)
                bass_pitch = np.random.randint(36, 48)
                seed[bass_pitch, beat_pos:beat_pos + 4] = 1.0
                # Kick drum pitch (35-36)
                seed[36, beat_pos:beat_pos + 2] = 1.0

            # Backbeat - snare area
            if beat % 4 == 2:  # Beats 2 and 4
                seed[38, beat_pos:beat_pos + 2] = 1.0  # Snare

            # Hi-hat on every beat
            if beat % 2 == 0:
                seed[42, beat_pos:beat_pos + 1] = 1.0  # Closed hi-hat

            # Add some melodic content in mid range
            if np.random.random() < density * 10:
                # Mid range: 60-72 (C4-C5)
                mid_pitch = np.random.randint(60, 72)
                note_length = np.random.randint(4, 12)
                end_pos = min(beat_pos + note_length, time_steps)
                seed[mid_pitch, beat_pos:end_pos] = 1.0

        # Add some random sparse notes for variation
        random_mask = np.random.random((num_pitches, time_steps)) > (1 - density * 0.5)
        seed = np.maximum(seed, random_mask.astype(np.float32))

        return seed

    def to_music(
        self,
        pianoroll: np.ndarray,
        resolution: int = 24,
        tempo: float = 120.0,
        program: int = 0,
        is_drum: bool = False,
    ) -> muspy.Music:
        """
        Convert pianoroll to muspy Music object.

        Args:
            pianoroll: Pianoroll array (128, time_steps)
            resolution: Ticks per beat
            tempo: Tempo in BPM
            program: MIDI program number (instrument), 128 for drums
            is_drum: Whether this is a drum track

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

        # Handle drums: program 128 is internal, MIDI needs 0-127
        is_drum = is_drum or program == 128

        # Set program for all tracks
        for track in music.tracks:
            track.program = 0 if is_drum else program
            track.is_drum = is_drum

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
        is_drum = (instrument == "Drums" or program == 128)

        # Convert to Music
        music = self.to_music(
            pianoroll,
            resolution=resolution,
            tempo=tempo,
            program=program,
            is_drum=is_drum,
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
        is_drum = (instrument == "Drums" or program == 128)

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
                is_drum=is_drum,
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


# ============================================================================
# Transformer Music Generator (Autoregressive)
# ============================================================================

class TransformerMusicGenerator:
    """
    Generate music using a trained Transformer model with autoregressive decoding.

    Uses TransformerModelBundle which contains the trained model, event vocabulary,
    and configuration needed for generation.
    """

    def __init__(self, bundle: TransformerModelBundle):
        """
        Initialize generator with a trained transformer bundle.

        Args:
            bundle: TransformerModelBundle containing model, vocabularies, and config
        """
        self.bundle = bundle
        self.model = bundle.model
        self.event_vocab = bundle.event_vocabulary
        self.vocabulary = bundle.vocabulary
        self.max_seq_length = bundle.max_seq_length

    @classmethod
    def from_bundle_path(cls, filepath: str) -> 'TransformerMusicGenerator':
        """
        Load generator from saved bundle file.

        Args:
            filepath: Path to the .h5 bundle file

        Returns:
            TransformerMusicGenerator instance
        """
        bundle = TransformerModelBundle.load(filepath)
        return cls(bundle)

    def list_genres(self) -> List[str]:
        """List available genres for conditioning."""
        return self.bundle.list_genres()

    def list_instruments(self) -> List[str]:
        """List all available instruments (General MIDI)."""
        return self.bundle.list_instruments()

    def list_active_instruments(self) -> List[str]:
        """List instruments that are actually used in training data."""
        active_ids = [
            i for i in range(129)
            if len(self.vocabulary.instrument_to_songs.get(i, set())) > 0
        ]
        return [GENERAL_MIDI_INSTRUMENTS[i] for i in active_ids if i in GENERAL_MIDI_INSTRUMENTS]

    def get_instruments_for_genre(self, genre: str) -> List[str]:
        """Get instruments commonly used in a specific genre."""
        instrument_ids = self.vocabulary.get_instruments_for_genre(genre)
        return [GENERAL_MIDI_INSTRUMENTS[i] for i in instrument_ids if i in GENERAL_MIDI_INSTRUMENTS]

    def get_top_instruments_for_genre(
        self,
        genre: str,
        top_n: int = 3,
        exclude_drums: bool = True
    ) -> List[str]:
        """Get top N most frequently used instruments for a genre."""
        instrument_ids = self.vocabulary.get_top_instruments_for_genre(
            genre, top_n, exclude_drums
        )
        return [GENERAL_MIDI_INSTRUMENTS[i] for i in instrument_ids if i in GENERAL_MIDI_INSTRUMENTS]

    def generate(
        self,
        genre: str,
        instrument: str = "Acoustic Grand Piano",
        max_length: Optional[int] = None,
        min_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        eos_penalty: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Generate a sequence of music events autoregressively.

        Args:
            genre: Genre name for conditioning
            instrument: Instrument name for conditioning (General MIDI name)
            max_length: Maximum sequence length (default: model's max_seq_length)
            min_length: Minimum sequence length before allowing EOS (prevents early stopping)
            temperature: Sampling temperature (higher = more random)
            top_k: If > 0, only sample from top k tokens
            top_p: If > 0, use nucleus sampling with this probability threshold
            repetition_penalty: Penalty for repeated tokens (> 1 reduces repetition)
            eos_penalty: Penalty for EOS token before min_length (higher = longer sequences)
            seed: Random seed for reproducibility
            verbose: If True, print generation progress

        Returns:
            Array of generated token IDs
        """
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        if max_length is None:
            max_length = self.max_seq_length

        # Create starting sequence: [BOS, GENRE, INSTRUMENT]
        start_tokens = self.bundle.create_start_sequence(genre, instrument)
        generated = list(start_tokens)

        if verbose:
            print(f"Generating {instrument} in {genre} style...")
            print(f"  Max length: {max_length}, Min length: {min_length}, Temperature: {temperature}")

        # Generate tokens one at a time
        for step in range(len(start_tokens), max_length):
            # Prepare input (batch of 1)
            input_ids = np.array([generated], dtype=np.int32)

            # Get logits from model
            logits = self.model.predict(input_ids, verbose=0)

            # Get logits for next token (last position)
            next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in set(generated):
                    next_token_logits[token] /= repetition_penalty

            # Apply EOS penalty before min_length to prevent early stopping
            # This is critical for models trained on short sequences
            if len(generated) < min_length and eos_penalty > 1.0:
                next_token_logits[self.event_vocab.EOS_TOKEN] /= eos_penalty

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Convert to probabilities
            probs = self._softmax(next_token_logits)

            # Apply top-k filtering
            if top_k > 0:
                probs = self._top_k_filter(probs, top_k)

            # Apply nucleus (top-p) sampling
            if top_p > 0:
                probs = self._top_p_filter(probs, top_p)

            # Re-normalize probabilities
            probs = probs / probs.sum()

            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(int(next_token))

            # Check for EOS - only stop if we've reached min_length
            if next_token == self.event_vocab.EOS_TOKEN:
                if len(generated) >= min_length:
                    if verbose:
                        print(f"  EOS reached at step {step}")
                    break
                else:
                    # Remove the EOS and continue generating
                    generated.pop()
                    if verbose and step % 50 == 0:
                        print(f"  Ignoring early EOS at step {step}, continuing...")

            # Progress indicator
            if verbose and step % 100 == 0:
                print(f"  Generated {step}/{max_length} tokens...")

        if verbose:
            print(f"  Generated {len(generated)} tokens total")

        return np.array(generated, dtype=np.int32)

    def generate_pianoroll(
        self,
        genre: str,
        instrument: str = "Acoustic Grand Piano",
        max_length: Optional[int] = None,
        min_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        eos_penalty: float = 2.0,
        resolution: int = 24,
        time_steps: int = 2048,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Generate music and convert to pianoroll representation.

        Args:
            genre: Genre for conditioning
            instrument: Instrument for conditioning
            max_length: Maximum token sequence length
            min_length: Minimum sequence length before allowing EOS
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeated tokens
            eos_penalty: Penalty for EOS token before min_length
            resolution: Ticks per beat (for time conversion)
            time_steps: Length of output pianoroll
            seed: Random seed
            verbose: Print progress

        Returns:
            Pianoroll array (128, time_steps)
        """
        # Generate token sequence
        tokens = self.generate(
            genre=genre,
            instrument=instrument,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_penalty=eos_penalty,
            seed=seed,
            verbose=verbose,
        )

        # Convert tokens to pianoroll
        pianoroll = self._tokens_to_pianoroll(tokens, time_steps)

        return pianoroll

    def generate_midi(
        self,
        genre: str,
        output_path: str,
        instrument: str = "Acoustic Grand Piano",
        max_length: Optional[int] = None,
        min_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        eos_penalty: float = 2.0,
        resolution: int = 24,
        tempo: float = 120.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> muspy.Music:
        """
        Generate music and save to MIDI file.

        Args:
            genre: Genre for conditioning
            output_path: Path to save MIDI file
            instrument: Instrument for conditioning
            max_length: Maximum token sequence length
            min_length: Minimum sequence length before allowing EOS
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeated tokens
            resolution: Ticks per beat
            tempo: Tempo in BPM
            seed: Random seed
            verbose: Print progress

        Returns:
            Generated muspy.Music object
        """
        from ..core.vocabulary import INSTRUMENT_NAME_TO_ID

        # Generate token sequence
        tokens = self.generate(
            genre=genre,
            instrument=instrument,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_penalty=eos_penalty,
            seed=seed,
            verbose=verbose,
        )

        # Convert to Music object
        music = self._tokens_to_music(
            tokens=tokens,
            instrument=instrument,
            resolution=resolution,
            tempo=tempo,
        )

        # Save to MIDI
        output_path = Path(output_path)  # type: ignore
        output_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        music.write_midi(str(output_path))

        if verbose:
            print(f"Saved MIDI to: {output_path}")

        return music

    def generate_song(
        self,
        genre: str,
        min_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_penalty: float = 2.0,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete song with multiple instruments.

        Args:
            genre: Genre for conditioning
            min_length: Minimum sequence length before allowing EOS
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            eos_penalty: Penalty for EOS token before min_length
            verbose: Print progress

        Returns:
            Dict mapping instrument names to token sequences
        """
        tracks: Dict[str, np.ndarray] = {}

        # Generate drums first
        if verbose:
            print(f"Generating drum track for '{genre}'...")
        tracks["Drums"] = self.generate(
            genre=genre,
            instrument="Drums",
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_penalty=eos_penalty,
            verbose=verbose,
        )

        # Get top instruments for this genre by frequency (same as LSTM generator)
        genre_instrument_ids = self.vocabulary.get_top_instruments_for_genre(
            genre, top_n=3, exclude_drums=True
        )
        if not genre_instrument_ids:
            # Fallback to common rock/pop instruments
            genre_instrument_ids = [25, 33, 0]  # Guitar, Bass, Piano

        for inst_id in genre_instrument_ids:
            instrument = GENERAL_MIDI_INSTRUMENTS.get(inst_id, "Acoustic Grand Piano")
            if verbose:
                print(f"\nGenerating {instrument}...")
            tracks[instrument] = self.generate(
                genre=genre,
                instrument=instrument,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_penalty=eos_penalty,
                verbose=verbose,
            )

        return tracks

    def generate_song_midi(
        self,
        genre: str,
        output_path: str,
        min_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_penalty: float = 2.0,
        resolution: int = 24,
        tempo: float = 120.0,
        verbose: bool = True,
    ) -> muspy.Music:
        """
        Generate a complete song and save to MIDI file.

        Args:
            genre: Genre for conditioning
            output_path: Path to save MIDI file
            min_length: Minimum sequence length before allowing EOS
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            eos_penalty: Penalty for EOS token before min_length
            resolution: Ticks per beat
            tempo: Tempo in BPM
            verbose: Print progress

        Returns:
            Generated muspy.Music object
        """
        # Generate all instrument tracks
        tracks = self.generate_song(
            genre=genre,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_penalty=eos_penalty,
            verbose=verbose,
        )

        # Create Music object
        music = muspy.Music(
            resolution=resolution,
            tempos=[muspy.Tempo(time=0, qpm=tempo)],
        )

        # Convert each track
        for instrument, tokens in tracks.items():
            track_music = self._tokens_to_music(
                tokens=tokens,
                instrument=instrument,
                resolution=resolution,
                tempo=tempo,
            )
            for track in track_music.tracks:
                music.tracks.append(track)

        # Save to MIDI
        output_path = Path(output_path)  # type: ignore
        output_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        music.write_midi(str(output_path))

        if verbose:
            print(f"\nSaved song to: {output_path}")

        return music

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()

    def _top_k_filter(self, probs: np.ndarray, k: int) -> np.ndarray:
        """Keep only top-k probability tokens."""
        if k <= 0 or k >= len(probs):
            return probs

        # Find k-th largest value
        threshold = np.partition(probs, -k)[-k]

        # Zero out everything below threshold
        filtered = probs.copy()
        filtered[probs < threshold] = 0

        return filtered

    def _top_p_filter(self, probs: np.ndarray, p: float) -> np.ndarray:
        """Apply nucleus (top-p) sampling."""
        if p <= 0 or p >= 1:
            return probs

        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find cutoff where cumulative probability exceeds p
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, p)

        # Zero out tokens beyond cutoff
        filtered = probs.copy()
        cutoff_indices = sorted_indices[cutoff_idx + 1:]
        filtered[cutoff_indices] = 0

        return filtered

    def _tokens_to_pianoroll(
        self,
        tokens: np.ndarray,
        time_steps: int = 2048,
    ) -> np.ndarray:
        """
        Convert token sequence to pianoroll representation.

        Args:
            tokens: Array of token IDs
            time_steps: Length of output pianoroll

        Returns:
            Pianoroll array (128, time_steps)
        """
        pianoroll = np.zeros((128, time_steps), dtype=np.float32)
        current_time = 0
        active_notes: Dict[int, int] = {}  # pitch -> start_time

        for token in tokens:
            event_type, value = self.event_vocab.decode_token(int(token))

            if event_type == 'note_on':
                pitch = value
                if pitch < 128 and current_time < time_steps:
                    active_notes[pitch] = current_time

            elif event_type == 'note_off':
                pitch = value
                if pitch in active_notes:
                    start = active_notes.pop(pitch)
                    end = min(current_time, time_steps)
                    if start < end:
                        pianoroll[pitch, start:end] = 1.0

            elif event_type == 'time_shift':
                current_time += value
                if current_time >= time_steps:
                    break

            elif event_type == 'eos':
                break

        # Close any remaining active notes
        for pitch, start in active_notes.items():
            end = min(current_time, time_steps)
            if start < end and pitch < 128:
                pianoroll[pitch, start:end] = 1.0

        return pianoroll

    def _tokens_to_music(
        self,
        tokens: np.ndarray,
        instrument: str = "Acoustic Grand Piano",
        resolution: int = 24,
        tempo: float = 120.0,
    ) -> muspy.Music:
        """
        Convert token sequence to muspy Music object.

        Args:
            tokens: Array of token IDs
            instrument: Instrument name
            resolution: Ticks per beat
            tempo: Tempo in BPM

        Returns:
            muspy.Music object
        """
        from ..core.vocabulary import INSTRUMENT_NAME_TO_ID

        notes = []
        current_time = 0
        active_notes: Dict[int, Tuple[int, int]] = {}  # pitch -> (start_time, velocity)

        for token in tokens:
            event_type, value = self.event_vocab.decode_token(int(token))

            if event_type == 'note_on':
                pitch = value
                if pitch < 128:
                    active_notes[pitch] = (current_time, 80)  # Default velocity

            elif event_type == 'note_off':
                pitch = value
                if pitch in active_notes:
                    start, velocity = active_notes.pop(pitch)
                    duration = current_time - start
                    if duration > 0:
                        notes.append(muspy.Note(
                            time=start,
                            pitch=pitch,
                            duration=duration,
                            velocity=velocity,
                        ))

            elif event_type == 'time_shift':
                current_time += value

            elif event_type == 'velocity':
                # Update velocity for next notes
                pass  # Could store and use for next note-on

            elif event_type == 'eos':
                break

        # Close any remaining active notes
        for pitch, (start, velocity) in active_notes.items():
            duration = max(1, current_time - start)
            notes.append(muspy.Note(
                time=start,
                pitch=pitch,
                duration=duration,
                velocity=velocity,
            ))

        # Create track
        program = INSTRUMENT_NAME_TO_ID.get(instrument, 0)
        is_drum = (instrument == "Drums" or program == 128)

        track = muspy.Track(
            program=0 if is_drum else program,
            is_drum=is_drum,
            name=instrument,
            notes=sorted(notes, key=lambda n: n.time),
        )

        # Create Music object
        music = muspy.Music(
            resolution=resolution,
            tempos=[muspy.Tempo(time=0, qpm=tempo)],
            tracks=[track],
        )

        return music

    def summary(self) -> str:
        """Get summary of generator configuration."""
        return self.bundle.summary()


# ============================================================================
# Factory Function for Creating Generators
# ============================================================================

def create_generator(
    bundle_path: str
) -> Union[MusicGenerator, TransformerMusicGenerator]:
    """
    Create appropriate generator based on model bundle type.

    This is the recommended way to create a generator when you don't know
    the model type ahead of time.

    Args:
        bundle_path: Path to the model bundle .h5 file

    Returns:
        MusicGenerator for LSTM models, TransformerMusicGenerator for Transformers

    Example:
        generator = create_generator("models/my_model.h5")
        if isinstance(generator, TransformerMusicGenerator):
            # Use autoregressive generation
            generator.generate_midi(genre="rock", output_path="out.mid")
        else:
            # Use LSTM generation
            generator.generate_song_midi(genre="rock", output_path="out.mid")
    """
    bundle = load_model_bundle(bundle_path)

    if isinstance(bundle, TransformerModelBundle):
        return TransformerMusicGenerator(bundle)
    else:
        return MusicGenerator(bundle)


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