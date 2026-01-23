from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import json
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import h5py # type: ignore
import muspy # type: ignore

from .vocabulary import Vocabulary

if TYPE_CHECKING:
    from ..data_processing.configs import MusicDatasetConfig

DRUM_PROGRAM_ID = 128


@dataclass
class MusicEntry:
    """A music object with its associated genre and cached drum pianoroll."""
    music: muspy.Music
    genre: str
    song_id: str = ""  # Unique identifier for the song (e.g., "Artist/SongName")
    drum_pianoroll: Optional[np.ndarray] = None  # Cached drum track for alignment (128, max_time_steps)


class MusicDataset:
    """
    Dataset of Music objects that converts to per-track TensorFlow tensors.

    Each track becomes a separate training sample with:
        - pianoroll: (128, time_steps) - the track's notes
        - instrument_id: int32 - MIDI program number (0-127) or 128 for drums
        - genre_id: int32 - genre vocabulary ID
    """

    def __init__(
        self,
        config: Optional["MusicDatasetConfig"] = None,
        resolution: int = 24,
        max_time_steps: int = 512,
    ):
        """
        Args:
            config: Optional MusicDatasetConfig (overrides resolution/max_time_steps)
            resolution: Ticks per beat (used if config not provided)
            max_time_steps: Fixed time dimension (used if config not provided)
        """
        self.config = config
        self.entries: List[MusicEntry] = []
        self.vocabulary: Vocabulary = Vocabulary()

        # Use config values if provided, otherwise use direct args
        if config is not None:
            self.resolution = config.resolution
            self.max_time_steps = config.max_time_steps
        else:
            self.resolution = resolution
            self.max_time_steps = max_time_steps

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, music: muspy.Music, genre: str, song_id: str = "") -> None:
        """
        Add a music object with its genre.

        Pre-computes and caches the drum pianoroll for this segment.

        Args:
            music: MusPy Music object (may be a segment from preprocessing)
            genre: Genre string
            song_id: Unique identifier for the song (e.g., "Artist/SongName")
        """
        # Generate song_id if not provided
        if not song_id:
            song_id = f"entry_{len(self.entries)}"

        # Pre-compute drum pianoroll for this segment
        drum_pianoroll = self._compute_drum_pianoroll(music)

        self.entries.append(MusicEntry(
            music=music,
            genre=genre,
            song_id=song_id,
            drum_pianoroll=drum_pianoroll,
        ))
        self.vocabulary.add_genre(genre)

        # Register instrument usage for each track
        for track in music.tracks:
            instrument_id = self._get_instrument_id(track)
            self.vocabulary.register_instrument_usage(instrument_id, song_id, genre)

    def _track_to_pianoroll(self, track: muspy.Track, music: muspy.Music) -> np.ndarray:
        """
        Convert a single track to a pianoroll array using MusPy's built-in.

        Args:
            track: MusPy Track object
            music: Parent Music object (for resolution info)

        Returns:
            np.ndarray of shape (128, max_time_steps)
        """
        # Create temporary Music with just this track to use MusPy's pianoroll
        temp_music = muspy.Music(
            resolution=music.resolution,
            tempos=music.tempos,
            tracks=[track],
        )

        # Use MusPy's built-in pianoroll representation (T, 128)
        pianoroll = temp_music.to_representation("pianoroll")

        # Transpose to (128, T) and pad/truncate to fixed size
        pianoroll = pianoroll.T.astype(np.float32)

        # Pad or truncate to max_time_steps
        current_steps = pianoroll.shape[1]
        if current_steps > self.max_time_steps:
            pianoroll = pianoroll[:, :self.max_time_steps]
        elif current_steps < self.max_time_steps:
            padding = np.zeros((128, self.max_time_steps - current_steps), dtype=np.float32)
            pianoroll = np.concatenate([pianoroll, padding], axis=1)

        return pianoroll

    def _get_instrument_id(self, track: muspy.Track) -> int:
        """Get instrument ID for a track (program number or 128 for drums)."""
        if track.is_drum:
            return DRUM_PROGRAM_ID
        return track.program

    def count_tracks(self) -> int:
        """Count total number of tracks across all entries."""
        return sum(len(entry.music.tracks) for entry in self.entries)

    def _compute_drum_pianoroll(self, music: muspy.Music) -> Optional[np.ndarray]:
        """
        Compute the combined drum pianoroll for a Music object.

        This is called at add() time to pre-compute and cache the drum pianoroll
        for each segment.

        Args:
            music: MusPy Music object (possibly a segment)

        Returns:
            np.ndarray of shape (128, max_time_steps) or None if no drums
        """
        drum_tracks = [t for t in music.tracks if t.is_drum]
        if not drum_tracks:
            return None

        # Create combined drum pianoroll
        combined = np.zeros((128, self.max_time_steps), dtype=np.float32)

        for track in drum_tracks:
            pianoroll = self._track_to_pianoroll(track, music)
            combined = np.maximum(combined, pianoroll)

        return combined

    def _get_drum_pianoroll(self, entry: MusicEntry) -> np.ndarray:
        """
        Get the drum pianoroll for an entry, using cached version if available.

        Args:
            entry: MusicEntry with cached drum_pianoroll

        Returns:
            np.ndarray of shape (128, max_time_steps), zeros if no drums
        """
        if entry.drum_pianoroll is not None:
            return entry.drum_pianoroll

        # Fallback: compute from music (for backward compatibility with old datasets)
        computed = self._compute_drum_pianoroll(entry.music)
        if computed is not None:
            return computed

        return np.zeros((128, self.max_time_steps), dtype=np.float32)

    def _get_non_drum_tracks(self, music: muspy.Music) -> List[muspy.Track]:
        """Get all non-drum tracks from a Music object."""
        return [t for t in music.tracks if not t.is_drum]

    def _get_drum_tracks(self, music: muspy.Music) -> List[muspy.Track]:
        """Get all drum tracks from a Music object."""
        return [t for t in music.tracks if t.is_drum]

    def get_instruments_in_entry(self, entry_idx: int) -> List[Tuple[int, str]]:
        """
        Get list of instruments used in an entry.

        Returns:
            List of (instrument_id, instrument_name) tuples
        """
        entry = self.entries[entry_idx]
        instruments = []
        for track in entry.music.tracks:
            inst_id = self._get_instrument_id(track)
            inst_name = self.vocabulary.get_instrument_name(inst_id)
            instruments.append((inst_id, inst_name))
        return instruments

    def _get_base_song_id(self, song_id: str) -> str:
        """
        Extract the base song ID by removing segment suffixes like '_seg0', '_seg1'.

        This ensures all segments from the same original song are grouped together.

        Args:
            song_id: The song_id which may include segment suffix

        Returns:
            Base song ID without segment suffix
        """
        import re
        # Remove segment suffix like "_seg0", "_seg1", etc.
        return re.sub(r"_seg\d+$", "", song_id)

    def to_tensorflow_dataset(
        self,
        splits: Optional[Tuple[float, float, float]] = None,
        random_state: Optional[int] = None,
        include_drums: bool = True,
    ) -> Union[tf.data.Dataset, Dict[str, tf.data.Dataset]]:
        """
        Convert to TensorFlow dataset with per-track samples.

        Each sample contains:
            - 'pianoroll': float32 (128, max_time_steps) - the track's notes
            - 'instrument_id': int32 scalar - MIDI program number (0-127) or 128 for drums
            - 'genre_id': int32 scalar - genre vocabulary ID
            - 'drum_pianoroll': float32 (128, max_time_steps) - combined drum track (zeros if no drums)

        Args:
            splits: Optional (train, val, test) proportions, e.g. (0.8, 0.1, 0.1)
            random_state: Random seed for reproducible splits
            include_drums: Whether to include drum_pianoroll in output

        Returns:
            Single dataset if splits is None, otherwise dict with 'train', 'validation', 'test'

        Note:
            When using splits, entries are grouped by base song ID to ensure that
            segments from the same song never appear in different splits (preventing data leakage).
        """
        # Pre-flatten all tracks into a list of (entry_idx, track_idx)
        track_indices: List[Tuple[int, int]] = []
        for entry_idx, entry in enumerate(self.entries):
            for track_idx in range(len(entry.music.tracks)):
                track_indices.append((entry_idx, track_idx))

        def generator(indices: List[Tuple[int, int]]):
            for entry_idx, track_idx in indices:
                entry = self.entries[entry_idx]
                track = entry.music.tracks[track_idx]

                # Skip empty tracks
                if len(track.notes) == 0:
                    continue

                pianoroll = self._track_to_pianoroll(track, entry.music)
                instrument_id = self._get_instrument_id(track)
                genre_id = self.vocabulary.get_genre_id(entry.genre)

                sample = {
                    "pianoroll": pianoroll,
                    "instrument_id": np.int32(instrument_id),
                    "genre_id": np.int32(genre_id),
                }

                if include_drums:
                    # Get cached drum pianoroll for this segment
                    sample["drum_pianoroll"] = self._get_drum_pianoroll(entry)

                yield sample

        output_signature = {
            "pianoroll": tf.TensorSpec(shape=(128, self.max_time_steps), dtype=tf.float32),  # type: ignore
            "instrument_id": tf.TensorSpec(shape=(), dtype=tf.int32),  # type: ignore
            "genre_id": tf.TensorSpec(shape=(), dtype=tf.int32),  # type: ignore
        }

        if include_drums:
            output_signature["drum_pianoroll"] = tf.TensorSpec(
                shape=(128, self.max_time_steps), dtype=tf.float32  # type: ignore
            )

        if splits is None:
            return tf.data.Dataset.from_generator(
                lambda: generator(track_indices),
                output_signature=output_signature,
            )

        # Handle splits - group by base song ID to prevent data leakage
        # All segments from the same song must be in the same split
        if random_state is not None:
            np.random.seed(random_state)

        # Group entry indices by base song ID
        song_to_entries: Dict[str, List[int]] = {}
        for entry_idx, entry in enumerate(self.entries):
            base_song_id = self._get_base_song_id(entry.song_id)
            if base_song_id not in song_to_entries:
                song_to_entries[base_song_id] = []
            song_to_entries[base_song_id].append(entry_idx)

        # Shuffle the unique song IDs
        unique_songs = list(song_to_entries.keys())
        np.random.shuffle(unique_songs)

        # Split at the song level
        n_songs = len(unique_songs)
        train_end = int(n_songs * splits[0])
        val_end = train_end + int(n_songs * splits[1])

        train_songs = set(unique_songs[:train_end])
        val_songs = set(unique_songs[train_end:val_end])
        test_songs = set(unique_songs[val_end:])

        # Collect track indices for each split based on song membership
        train_idx: List[Tuple[int, int]] = []
        val_idx: List[Tuple[int, int]] = []
        test_idx: List[Tuple[int, int]] = []

        for entry_idx, entry in enumerate(self.entries):
            base_song_id = self._get_base_song_id(entry.song_id)
            for track_idx in range(len(entry.music.tracks)):
                track_tuple = (entry_idx, track_idx)
                if base_song_id in train_songs:
                    train_idx.append(track_tuple)
                elif base_song_id in val_songs:
                    val_idx.append(track_tuple)
                else:
                    test_idx.append(track_tuple)

        # Shuffle within each split
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        return {
            "train": tf.data.Dataset.from_generator(
                lambda: generator(train_idx),
                output_signature=output_signature,
            ),
            "validation": tf.data.Dataset.from_generator(
                lambda: generator(val_idx),
                output_signature=output_signature,
            ),
            "test": tf.data.Dataset.from_generator(
                lambda: generator(test_idx),
                output_signature=output_signature,
            ),
        }

    def save(self, filepath: str) -> None:
        """
        Save dataset to HDF5 file.

        Saves:
            - Dataset metadata (resolution, max_time_steps)
            - Vocabulary (genre mappings, instrument mappings)
            - All music entries (pickled Music objects + genre strings + song_id)

        Args:
            filepath: Path to save the .h5 file
        """
        filepath = Path(filepath)  # type: ignore
        filepath.parent.mkdir(parents=True, exist_ok=True)  # type: ignore

        with h5py.File(filepath, "w") as f:
            # Save metadata
            meta = f.create_group("metadata")
            meta.attrs["resolution"] = self.resolution
            meta.attrs["max_time_steps"] = self.max_time_steps

            # Save vocabulary (full serialization including instruments)
            vocab_group = f.create_group("vocabulary")
            vocab_group.attrs["genre_to_id"] = json.dumps(self.vocabulary.genre_to_id)
            vocab_group.attrs["artist_to_id"] = json.dumps(self.vocabulary.artist_to_id)
            vocab_group.attrs["instrument_to_songs"] = json.dumps(
                {str(k): list(v) for k, v in self.vocabulary.instrument_to_songs.items()}
            )
            vocab_group.attrs["genre_to_instruments"] = json.dumps(
                {k: list(v) for k, v in self.vocabulary.genre_to_instruments.items()}
            )

            # Save entries
            entries_group = f.create_group("entries")
            entries_group.attrs["count"] = len(self.entries)

            for idx, entry in enumerate(self.entries):
                entry_group = entries_group.create_group(str(idx))

                # Save genre and song_id as attributes
                entry_group.attrs["genre"] = entry.genre
                entry_group.attrs["song_id"] = entry.song_id

                # Save Music object as pickled bytes
                music_bytes = pickle.dumps(entry.music)
                entry_group.create_dataset(
                    "music_pickle",
                    data=np.frombuffer(music_bytes, dtype=np.uint8),
                )

                # Save cached drum pianoroll if present
                if entry.drum_pianoroll is not None:
                    entry_group.create_dataset(
                        "drum_pianoroll",
                        data=entry.drum_pianoroll,
                        compression="gzip",
                    )

    @classmethod
    def load(cls, filepath: str) -> "MusicDataset":
        """
        Load dataset from HDF5 file.

        Args:
            filepath: Path to the .h5 file

        Returns:
            MusicDataset instance
        """
        with h5py.File(filepath, "r") as f:
            # Load metadata
            resolution = int(f["metadata"].attrs["resolution"])  # type: ignore
            max_time_steps = int(f["metadata"].attrs["max_time_steps"])  # type: ignore

            # Create dataset
            dataset = cls(resolution=resolution, max_time_steps=max_time_steps)

            # Load vocabulary
            dataset.vocabulary.genre_to_id = json.loads(f["vocabulary"].attrs["genre_to_id"])  # type: ignore
            dataset.vocabulary.artist_to_id = json.loads(f["vocabulary"].attrs["artist_to_id"])  # type: ignore

            # Load instrument mappings (if present, for backward compatibility)
            if "instrument_to_songs" in f["vocabulary"].attrs:
                inst_to_songs = json.loads(f["vocabulary"].attrs["instrument_to_songs"])  # type: ignore
                for k, v in inst_to_songs.items():
                    dataset.vocabulary.instrument_to_songs[int(k)] = set(v)

            if "genre_to_instruments" in f["vocabulary"].attrs:
                genre_to_inst = json.loads(f["vocabulary"].attrs["genre_to_instruments"])  # type: ignore
                for k, v in genre_to_inst.items():
                    dataset.vocabulary.genre_to_instruments[k] = set(v)

            # Load entries
            entries_group = f["entries"]
            count = int(entries_group.attrs["count"])  # type: ignore

            for idx in range(count):
                entry_group = entries_group[str(idx)]  # type: ignore

                genre = entry_group.attrs["genre"]
                song_id = entry_group.attrs.get("song_id", f"entry_{idx}")
                music_bytes = entry_group["music_pickle"][:].tobytes()  # type: ignore
                music = pickle.loads(music_bytes)

                # Load cached drum pianoroll if present
                drum_pianoroll = None
                if "drum_pianoroll" in entry_group:
                    drum_pianoroll = entry_group["drum_pianoroll"][:]  # type: ignore

                dataset.entries.append(MusicEntry(
                    music=music,
                    genre=genre,
                    song_id=song_id,
                    drum_pianoroll=drum_pianoroll,
                ))

        return dataset
