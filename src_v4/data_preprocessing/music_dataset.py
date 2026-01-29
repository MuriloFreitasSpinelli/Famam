"""
MusicDataset for storing and converting music data to training format.

Stores muspy.Music objects with metadata and converts them to
TensorFlow datasets using pluggable encoders (EventEncoder, REMIEncoder).
"""

from typing import List, Tuple, Optional, Dict, Union, Any, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
import re

import numpy as np
import h5py

from .vocabulary import Vocabulary, DRUM_PROGRAM_ID
from .encoders import BaseEncoder, EncodedSequence

if TYPE_CHECKING:
    import muspy
    import tensorflow as tf


@dataclass
class MusicEntry:
    """
    A music entry with associated metadata.

    Attributes:
        music: muspy.Music object containing one or more tracks
        genre: Genre string
        song_id: Unique identifier (e.g., "Artist/SongName" or "Artist/SongName_seg0")
    """
    music: Any  # muspy.Music
    genre: str
    song_id: str


class MusicDataset:
    """
    Dataset of music tracks that converts to TensorFlow datasets via encoders.

    Each entry is a single track with metadata (genre_id, instrument_id).

    Features:
        - Pluggable encoder architecture
        - Song-level train/val/test splits (prevents data leakage)
        - HDF5 persistence
        - Vocabulary management for genres/instruments
    """

    def __init__(
        self,
        entries: Optional[List[MusicEntry]] = None,
        vocabulary: Optional[Vocabulary] = None,
        resolution: int = 24,
        max_seq_length: int = 2048,
    ):
        """
        Initialize the dataset.

        Args:
            entries: List of MusicEntry objects (optional)
            vocabulary: Vocabulary instance (optional)
            resolution: Ticks per beat (quarter note)
            max_seq_length: Maximum token sequence length for encoding
        """
        self.resolution = resolution
        self.max_seq_length = max_seq_length
        self.entries: List[MusicEntry] = entries or []
        self.vocabulary = vocabulary or Vocabulary()

    def __len__(self) -> int:
        """Number of music entries (songs/segments)."""
        return len(self.entries)

    def add_entry(self, entry: MusicEntry) -> None:
        """
        Add a MusicEntry to the dataset.

        Args:
            entry: MusicEntry object
        """
        self.entries.append(entry)

    def add(
        self,
        music: "muspy.Music",
        genre: str,
        song_id: str = "",
    ) -> None:
        """
        Add a Music object with metadata.

        Args:
            music: muspy.Music object
            genre: Genre string
            song_id: Unique identifier (auto-generated if not provided)
        """
        if not song_id:
            song_id = f"entry_{len(self.entries)}"

        # Register instruments in vocabulary
        for track in music.tracks:
            instrument_id = DRUM_PROGRAM_ID if track.is_drum else track.program
            self.vocabulary.register_instrument_usage(instrument_id, song_id, genre)

        self.entries.append(MusicEntry(
            music=music,
            genre=genre,
            song_id=song_id,
        ))

    def count_tracks(self) -> int:
        """Count total number of tracks across all entries."""
        return sum(len(entry.music.tracks) for entry in self.entries)

    def _get_instrument_id(self, track: "muspy.Track") -> int:
        """Get instrument ID from a track."""
        return DRUM_PROGRAM_ID if track.is_drum else track.program

    def _get_base_song_id(self, song_id: str) -> str:
        """
        Extract base song ID by removing segment suffixes like '_seg0' or '_0'.

        Ensures all segments from the same song are grouped together for splits.
        """
        return re.sub(r"_\d+$", "", song_id)

    # === Encoder-based conversion ===

    def to_tensorflow_dataset(
        self,
        encoder: BaseEncoder,
        splits: Optional[Tuple[float, float, float]] = None,
        random_state: Optional[int] = None,
        include_metadata: bool = False,
    ) -> Union["tf.data.Dataset", Dict[str, "tf.data.Dataset"]]:
        """
        Convert to TensorFlow dataset using the provided encoder.

        Each track becomes a sample with:
            - 'input_ids': int32 (max_seq_length,) - token sequence
            - 'attention_mask': int32 (max_seq_length,) - 1=real, 0=padding
            - 'labels': int32 (max_seq_length,) - shifted for next-token prediction

        Optionally includes metadata:
            - 'genre_id': int32 scalar
            - 'instrument_id': int32 scalar

        Args:
            encoder: BaseEncoder instance (EventEncoder or REMIEncoder)
            splits: Optional (train, val, test) proportions, e.g., (0.8, 0.1, 0.1)
            random_state: Random seed for reproducible splits
            include_metadata: Whether to include genre_id and instrument_id

        Returns:
            Single dataset if splits is None, otherwise dict with 'train', 'validation', 'test'
        """
        import tensorflow as tf

        # Pre-flatten all tracks into (entry_idx, track_idx) tuples
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

                # Get IDs
                genre_id = self.vocabulary.get_genre_id(entry.genre)
                instrument_id = self._get_instrument_id(track)

                # Encode track
                encoded = encoder.encode_track(
                    track=track,
                    genre_id=genre_id,
                    instrument_id=instrument_id,
                    max_length=self.max_seq_length,
                )

                # Create labels (shifted by 1)
                labels = encoder.create_labels(encoded.token_ids)

                sample = {
                    "input_ids": encoded.token_ids,
                    "attention_mask": encoded.attention_mask,
                    "labels": labels,
                }

                if include_metadata:
                    sample["genre_id"] = np.int32(genre_id)
                    sample["instrument_id"] = np.int32(instrument_id)

                yield sample

        # Build output signature
        output_signature = {
            "input_ids": tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
            "labels": tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
        }

        if include_metadata:
            output_signature["genre_id"] = tf.TensorSpec(shape=(), dtype=tf.int32)
            output_signature["instrument_id"] = tf.TensorSpec(shape=(), dtype=tf.int32)

        if splits is None:
            return tf.data.Dataset.from_generator(
                lambda: generator(track_indices),
                output_signature=output_signature,
            )

        # Handle splits - group by base song ID to prevent data leakage
        if random_state is not None:
            np.random.seed(random_state)

        # Group entry indices by base song ID
        song_to_entries: Dict[str, List[int]] = {}
        for entry_idx, entry in enumerate(self.entries):
            base_song_id = self._get_base_song_id(entry.song_id)
            if base_song_id not in song_to_entries:
                song_to_entries[base_song_id] = []
            song_to_entries[base_song_id].append(entry_idx)

        # Shuffle unique song IDs
        unique_songs = list(song_to_entries.keys())
        np.random.shuffle(unique_songs)

        # Split at song level
        n_songs = len(unique_songs)
        train_end = int(n_songs * splits[0])
        val_end = train_end + int(n_songs * splits[1])

        train_songs = set(unique_songs[:train_end])
        val_songs = set(unique_songs[train_end:val_end])
        test_songs = set(unique_songs[val_end:])

        # Collect track indices for each split
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

    def to_multitrack_dataset(
        self,
        encoder: "MultiTrackEncoder",
        splits: Optional[Tuple[float, float, float]] = None,
        random_state: Optional[int] = None,
        min_tracks: int = 2,
    ) -> Union["tf.data.Dataset", Dict[str, "tf.data.Dataset"]]:
        """
        Convert to TensorFlow dataset using multi-track encoding.

        Each entry (full song) becomes a sample with all tracks interleaved.
        This allows the model to learn relationships between all instruments.

        Args:
            encoder: MultiTrackEncoder instance
            splits: Optional (train, val, test) proportions
            random_state: Random seed for reproducible splits
            min_tracks: Minimum tracks required per entry

        Returns:
            Single dataset or dict with train/validation/test
        """
        import tensorflow as tf
        from .encoders import MultiTrackEncoder

        # Filter entries with enough tracks
        valid_indices = [
            i for i, entry in enumerate(self.entries)
            if len(entry.music.tracks) >= min_tracks
            and sum(len(t.notes) for t in entry.music.tracks) > 0
        ]

        def generator(indices: List[int]):
            for entry_idx in indices:
                entry = self.entries[entry_idx]

                # Get genre ID
                genre_id = self.vocabulary.get_genre_id(entry.genre)
                if genre_id < 0:
                    genre_id = 0

                # Encode full multi-track music
                encoded = encoder.encode_music(
                    music=entry.music,
                    genre_id=genre_id,
                    max_length=self.max_seq_length,
                )

                # Create labels (shifted by 1)
                labels = encoder.create_labels(encoded.token_ids)

                yield {
                    "input_ids": encoded.token_ids,
                    "attention_mask": encoded.attention_mask,
                    "labels": labels,
                }

        # Output signature
        output_signature = {
            "input_ids": tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
            "labels": tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
        }

        if splits is None:
            return tf.data.Dataset.from_generator(
                lambda: generator(valid_indices),
                output_signature=output_signature,
            )

        # Handle splits by song
        if random_state is not None:
            np.random.seed(random_state)

        # Group by base song ID
        song_to_indices: Dict[str, List[int]] = {}
        for idx in valid_indices:
            base_id = self._get_base_song_id(self.entries[idx].song_id)
            if base_id not in song_to_indices:
                song_to_indices[base_id] = []
            song_to_indices[base_id].append(idx)

        # Shuffle and split
        unique_songs = list(song_to_indices.keys())
        np.random.shuffle(unique_songs)

        n = len(unique_songs)
        train_end = int(n * splits[0])
        val_end = train_end + int(n * splits[1])

        train_songs = set(unique_songs[:train_end])
        val_songs = set(unique_songs[train_end:val_end])
        test_songs = set(unique_songs[val_end:])

        # Collect indices
        train_idx = [i for s in train_songs for i in song_to_indices[s]]
        val_idx = [i for s in val_songs for i in song_to_indices[s]]
        test_idx = [i for s in test_songs for i in song_to_indices[s]]

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

    # === HDF5 Persistence ===

    def save(self, filepath: str) -> None:
        """
        Save dataset to HDF5 file.

        Saves:
            - Metadata (resolution, max_seq_length)
            - Vocabulary (genres, instruments, mappings)
            - All music entries (pickled Music objects)

        Args:
            filepath: Path to save the .h5 file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "w") as f:
            # Save metadata
            meta = f.create_group("metadata")
            meta.attrs["resolution"] = self.resolution
            meta.attrs["max_seq_length"] = self.max_seq_length

            # Save vocabulary
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
                entry_group.attrs["genre"] = entry.genre
                entry_group.attrs["song_id"] = entry.song_id

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
            resolution = int(f["metadata"].attrs["resolution"])
            max_seq_length = int(f["metadata"].attrs["max_seq_length"])

            # Create dataset
            dataset = cls(resolution=resolution, max_seq_length=max_seq_length)

            # Load vocabulary
            dataset.vocabulary.genre_to_id = json.loads(f["vocabulary"].attrs["genre_to_id"])
            dataset.vocabulary.artist_to_id = json.loads(f["vocabulary"].attrs["artist_to_id"])

            if "instrument_to_songs" in f["vocabulary"].attrs:
                inst_to_songs = json.loads(f["vocabulary"].attrs["instrument_to_songs"])
                for k, v in inst_to_songs.items():
                    dataset.vocabulary.instrument_to_songs[int(k)] = set(v)

            if "genre_to_instruments" in f["vocabulary"].attrs:
                genre_to_inst = json.loads(f["vocabulary"].attrs["genre_to_instruments"])
                for k, v in genre_to_inst.items():
                    dataset.vocabulary.genre_to_instruments[k] = set(v)

            # Load entries
            entries_group = f["entries"]
            count = int(entries_group.attrs["count"])

            for idx in range(count):
                entry_group = entries_group[str(idx)]
                genre = entry_group.attrs["genre"]
                song_id = entry_group.attrs.get("song_id", f"entry_{idx}")
                music_bytes = entry_group["music_pickle"][:].tobytes()
                music = pickle.loads(music_bytes)

                dataset.entries.append(MusicEntry(
                    music=music,
                    genre=genre,
                    song_id=song_id,
                ))

        return dataset

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_notes = 0
        total_duration = 0

        for entry in self.entries:
            for track in entry.music.tracks:
                total_notes += len(track.notes)
                if track.notes:
                    max_time = max(n.time + n.duration for n in track.notes)
                    total_duration = max(total_duration, max_time)

        return {
            "num_entries": len(self.entries),
            "num_tracks": self.count_tracks(),
            "num_genres": self.vocabulary.num_genres,
            "genres": self.vocabulary.genres,
            "num_active_instruments": self.vocabulary.num_active_instruments,
            "total_notes": total_notes,
            "resolution": self.resolution,
            "max_seq_length": self.max_seq_length,
        }

    def get_instrument_stats(self, top_n: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Get instrument usage statistics sorted by frequency.

        Args:
            top_n: If provided, only return the top N instruments

        Returns:
            List of (instrument_name, song_count) tuples sorted by count (descending)
        """
        stats = self.vocabulary.get_instrument_stats()
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)

        if top_n is not None:
            return sorted_stats[:top_n]
        return sorted_stats

    def print_instrument_stats(self, top_n: int = 20) -> None:
        """
        Print a formatted table of instrument usage statistics.

        Args:
            top_n: Number of top instruments to display
        """
        stats = self.get_instrument_stats(top_n)

        if not stats:
            print("No instrument usage data available.")
            return

        print("\n" + "=" * 50)
        print("  Instrument Usage Statistics")
        print("=" * 50)
        print(f"{'Rank':<6} {'Instrument':<30} {'Songs':<10}")
        print("-" * 50)

        for rank, (instrument, count) in enumerate(stats, 1):
            print(f"{rank:<6} {instrument:<30} {count:<10}")

        total_instruments = self.vocabulary.num_active_instruments
        if total_instruments > top_n:
            print(f"\n... and {total_instruments - top_n} more instruments")

        print("=" * 50)

    def __repr__(self) -> str:
        return (
            f"MusicDataset(entries={len(self.entries)}, "
            f"tracks={self.count_tracks()}, "
            f"genres={self.vocabulary.num_genres}, "
            f"resolution={self.resolution})"
        )
