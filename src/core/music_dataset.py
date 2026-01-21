from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
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
    """A music object with its associated genre."""
    music: muspy.Music
    genre: str


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
        max_time_steps: int = 1000,
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

    def add(self, music: muspy.Music, genre: str) -> None:
        """
        Add a music object with its genre.

        Args:
            music: MusPy Music object
            genre: Genre string
        """
        self.entries.append(MusicEntry(music=music, genre=genre))
        self.vocabulary.add_genre(genre)

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

    def to_tensorflow_dataset(
        self,
        splits: Optional[Tuple[float, float, float]] = None,
        random_state: Optional[int] = None,
    ) -> tf.data.Dataset | Dict[str, tf.data.Dataset]:
        """
        Convert to TensorFlow dataset with per-track samples.

        Each sample contains:
            - 'pianoroll': float32 (128, max_time_steps)
            - 'instrument_id': int32 scalar
            - 'genre_id': int32 scalar

        Args:
            splits: Optional (train, val, test) proportions, e.g. (0.8, 0.1, 0.1)
            random_state: Random seed for reproducible splits

        Returns:
            Single dataset if splits is None, otherwise dict with 'train', 'validation', 'test'
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

                yield {
                    "pianoroll": pianoroll,
                    "instrument_id": np.int32(instrument_id),
                    "genre_id": np.int32(genre_id),
                }

        output_signature = {
            "pianoroll": tf.TensorSpec(shape=(128, self.max_time_steps), dtype=tf.float32), # type: ignore
            "instrument_id": tf.TensorSpec(shape=(), dtype=tf.int32), # type: ignore
            "genre_id": tf.TensorSpec(shape=(), dtype=tf.int32), # type: ignore
        }

        if splits is None:
            return tf.data.Dataset.from_generator(
                lambda: generator(track_indices),
                output_signature=output_signature,
            )

        # Handle splits
        if random_state is not None:
            np.random.seed(random_state)

        indices_array = np.array(track_indices)
        np.random.shuffle(indices_array)
        shuffled_indices = [tuple(x) for x in indices_array.tolist()]

        n = len(shuffled_indices)
        train_end = int(n * splits[0])
        val_end = train_end + int(n * splits[1])

        train_idx = shuffled_indices[:train_end]
        val_idx = shuffled_indices[train_end:val_end]
        test_idx = shuffled_indices[val_end:]

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
            - Vocabulary (genre mappings)
            - All music entries (pickled Music objects + genre strings)

        Args:
            filepath: Path to save the .h5 file
        """
        filepath = Path(filepath) # type: ignore
        filepath.parent.mkdir(parents=True, exist_ok=True) # type: ignore

        with h5py.File(filepath, "w") as f:
            # Save metadata
            meta = f.create_group("metadata")
            meta.attrs["resolution"] = self.resolution
            meta.attrs["max_time_steps"] = self.max_time_steps

            # Save vocabulary
            vocab_group = f.create_group("vocabulary")
            vocab_group.attrs["genre_to_id"] = json.dumps(self.vocabulary.genre_to_id)
            vocab_group.attrs["artist_to_id"] = json.dumps(self.vocabulary.artist_to_id)

            # Save entries
            entries_group = f.create_group("entries")
            entries_group.attrs["count"] = len(self.entries)

            for idx, entry in enumerate(self.entries):
                entry_group = entries_group.create_group(str(idx))

                # Save genre as attribute
                entry_group.attrs["genre"] = entry.genre

                # Save Music object as pickled bytes
                music_bytes = pickle.dumps(entry.music)
                entry_group.create_dataset(
                    "music_pickle",
                    data=np.frombuffer(music_bytes, dtype=np.uint8),
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
            resolution = int(f["metadata"].attrs["resolution"]) # type: ignore
            max_time_steps = int(f["metadata"].attrs["max_time_steps"]) # type: ignore

            # Create dataset
            dataset = cls(resolution=resolution, max_time_steps=max_time_steps)

            # Load vocabulary
            dataset.vocabulary.genre_to_id = json.loads(f["vocabulary"].attrs["genre_to_id"]) # type: ignore
            dataset.vocabulary.artist_to_id = json.loads(f["vocabulary"].attrs["artist_to_id"]) # type: ignore

            # Load entries
            entries_group = f["entries"]
            count = int(entries_group.attrs["count"]) # type: ignore

            for idx in range(count):
                entry_group = entries_group[str(idx)] # type: ignore

                genre = entry_group.attrs["genre"]
                music_bytes = entry_group["music_pickle"][:].tobytes() # type: ignore
                music = pickle.loads(music_bytes)

                dataset.entries.append(MusicEntry(music=music, genre=genre)) # type: ignore

        return dataset
