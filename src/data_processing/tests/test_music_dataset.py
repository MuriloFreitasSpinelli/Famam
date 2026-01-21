"""Tests for MusicDataset class."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import muspy

from ...core import MusicDataset, DRUM_PROGRAM_ID


class TestMusicDataset:
    """Tests for MusicDataset core functionality."""

    def _create_mock_music(self, num_tracks=2, notes_per_track=10, resolution=24):
        """Helper to create a mock Music object."""
        music = muspy.Music(resolution=resolution)
        music.tempos = [muspy.Tempo(time=0, qpm=120)]

        for i in range(num_tracks):
            track = muspy.Track(program=i, is_drum=(i == 0))  # First track is drums
            for j in range(notes_per_track):
                track.notes.append(
                    muspy.Note(time=j * 24, pitch=60 + (j % 12), duration=24, velocity=64)
                )
            music.tracks.append(track)

        return music

    def test_create_empty_dataset(self):
        """Test creating an empty dataset."""
        dataset = MusicDataset(resolution=24, max_time_steps=500)

        assert len(dataset) == 0
        assert dataset.resolution == 24
        assert dataset.max_time_steps == 500
        assert dataset.count_tracks() == 0

    def test_add_music(self):
        """Test adding music to dataset."""
        dataset = MusicDataset()
        music = self._create_mock_music(num_tracks=3)

        dataset.add(music, "Rock")

        assert len(dataset) == 1
        assert dataset.count_tracks() == 3
        assert "Rock" in dataset.vocabulary.genre_to_id

    def test_add_multiple_genres(self):
        """Test adding music with different genres."""
        dataset = MusicDataset()

        dataset.add(self._create_mock_music(), "Rock")
        dataset.add(self._create_mock_music(), "Jazz")
        dataset.add(self._create_mock_music(), "Rock")  # Duplicate genre

        assert len(dataset) == 3
        assert dataset.vocabulary.num_genres == 2
        assert dataset.vocabulary.get_genre_id("Rock") == 0
        assert dataset.vocabulary.get_genre_id("Jazz") == 1

    def test_instrument_id_for_drums(self):
        """Test that drum tracks get instrument_id = 128."""
        dataset = MusicDataset()
        music = self._create_mock_music()

        # First track is drum in our mock
        drum_track = music.tracks[0]
        assert drum_track.is_drum is True

        instrument_id = dataset._get_instrument_id(drum_track)
        assert instrument_id == DRUM_PROGRAM_ID  # 128

    def test_instrument_id_for_regular_track(self):
        """Test that regular tracks get their program number."""
        dataset = MusicDataset()
        music = self._create_mock_music()

        # Second track is not drum, has program=1
        regular_track = music.tracks[1]
        assert regular_track.is_drum is False

        instrument_id = dataset._get_instrument_id(regular_track)
        assert instrument_id == regular_track.program


class TestMusicDatasetSaveLoad:
    """Tests for MusicDataset save/load functionality."""

    def _create_mock_music(self, num_tracks=2, notes_per_track=10, resolution=24):
        """Helper to create a mock Music object."""
        music = muspy.Music(resolution=resolution)
        music.tempos = [muspy.Tempo(time=0, qpm=120)]

        for i in range(num_tracks):
            track = muspy.Track(program=i, is_drum=False)
            for j in range(notes_per_track):
                track.notes.append(
                    muspy.Note(time=j * 24, pitch=60 + (j % 12), duration=24, velocity=64)
                )
            music.tracks.append(track)

        return music

    def test_save_and_load_empty_dataset(self):
        """Test saving and loading an empty dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty.h5"

            # Save empty dataset
            dataset = MusicDataset(resolution=48, max_time_steps=2000)
            dataset.save(str(filepath))

            assert filepath.exists()

            # Load and verify
            loaded = MusicDataset.load(str(filepath))
            assert len(loaded) == 0
            assert loaded.resolution == 48
            assert loaded.max_time_steps == 2000

    def test_save_and_load_with_entries(self):
        """Test saving and loading dataset with music entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "dataset.h5"

            # Create and populate dataset
            dataset = MusicDataset(resolution=24, max_time_steps=1000)
            dataset.add(self._create_mock_music(num_tracks=2), "Rock")
            dataset.add(self._create_mock_music(num_tracks=3), "Jazz")
            dataset.add(self._create_mock_music(num_tracks=1), "Rock")

            dataset.save(str(filepath))

            # Load and verify
            loaded = MusicDataset.load(str(filepath))

            assert len(loaded) == 3
            assert loaded.resolution == 24
            assert loaded.max_time_steps == 1000

            # Check vocabulary
            assert loaded.vocabulary.num_genres == 2
            assert "Rock" in loaded.vocabulary.genre_to_id
            assert "Jazz" in loaded.vocabulary.genre_to_id

            # Check entries
            assert loaded.entries[0].genre == "Rock"
            assert loaded.entries[1].genre == "Jazz"
            assert len(loaded.entries[0].music.tracks) == 2
            assert len(loaded.entries[1].music.tracks) == 3

    def test_save_and_load_preserves_notes(self):
        """Test that notes are preserved after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "notes_test.h5"

            # Create dataset with specific notes
            dataset = MusicDataset()
            music = self._create_mock_music(num_tracks=1, notes_per_track=5)
            original_notes = [(n.time, n.pitch, n.duration) for n in music.tracks[0].notes]
            dataset.add(music, "Test")

            dataset.save(str(filepath))
            loaded = MusicDataset.load(str(filepath))

            # Verify notes match
            loaded_notes = [
                (n.time, n.pitch, n.duration)
                for n in loaded.entries[0].music.tracks[0].notes
            ]
            assert loaded_notes == original_notes

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "path" / "dataset.h5"

            dataset = MusicDataset()
            dataset.save(str(filepath))

            assert filepath.exists()


class TestMusicDatasetTensorflow:
    """Tests for TensorFlow dataset conversion."""

    def _create_mock_music(self, num_tracks=2, notes_per_track=10, resolution=24):
        """Helper to create a mock Music object."""
        music = muspy.Music(resolution=resolution)
        music.tempos = [muspy.Tempo(time=0, qpm=120)]

        for i in range(num_tracks):
            track = muspy.Track(program=i * 10, is_drum=False)
            for j in range(notes_per_track):
                track.notes.append(
                    muspy.Note(time=j * 24, pitch=60 + (j % 12), duration=24, velocity=64)
                )
            music.tracks.append(track)

        return music

    def test_to_tensorflow_dataset_basic(self):
        """Test basic TensorFlow dataset conversion."""
        dataset = MusicDataset(resolution=24, max_time_steps=500)
        dataset.add(self._create_mock_music(num_tracks=2), "Rock")

        tf_dataset = dataset.to_tensorflow_dataset()

        # Should yield 2 samples (one per track)
        samples = list(tf_dataset.take(10))
        assert len(samples) == 2

        # Check sample structure
        sample = samples[0]
        assert "pianoroll" in sample
        assert "instrument_id" in sample
        assert "genre_id" in sample

        # Check shapes
        assert sample["pianoroll"].shape == (128, 500)
        assert sample["instrument_id"].shape == ()
        assert sample["genre_id"].shape == ()

    def test_to_tensorflow_dataset_with_splits(self):
        """Test TensorFlow dataset with train/val/test splits."""
        dataset = MusicDataset(resolution=24, max_time_steps=500)

        # Add enough entries for meaningful splits
        for i in range(10):
            dataset.add(self._create_mock_music(num_tracks=2), "Rock")

        splits = dataset.to_tensorflow_dataset(
            splits=(0.6, 0.2, 0.2),
            random_state=42,
        )

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

        # Count samples in each split
        train_count = len(list(splits["train"]))
        val_count = len(list(splits["validation"]))
        test_count = len(list(splits["test"]))

        # Total should be 20 (10 entries * 2 tracks each)
        total = train_count + val_count + test_count
        assert total == 20

    def test_empty_tracks_are_skipped(self):
        """Test that tracks with no notes are skipped."""
        dataset = MusicDataset(resolution=24, max_time_steps=500)

        # Create music with one empty track
        music = muspy.Music(resolution=24)
        music.tempos = [muspy.Tempo(time=0, qpm=120)]

        # Track with notes
        track1 = muspy.Track(program=0, is_drum=False)
        track1.notes.append(muspy.Note(time=0, pitch=60, duration=24, velocity=64))
        music.tracks.append(track1)

        # Empty track
        track2 = muspy.Track(program=1, is_drum=False)
        music.tracks.append(track2)

        dataset.add(music, "Test")

        tf_dataset = dataset.to_tensorflow_dataset()
        samples = list(tf_dataset)

        # Should only yield 1 sample (empty track skipped)
        assert len(samples) == 1
