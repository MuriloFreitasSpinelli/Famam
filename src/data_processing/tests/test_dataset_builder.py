"""Tests for dataset_builder functions."""

import pytest
import tempfile
from pathlib import Path
import muspy

from ..dataset_builder import (
    load_genre_map,
    find_midi_files,
    get_genre_for_file,
    passes_filters,
    build_dataset,
)
from ..configs import MusicDatasetConfig


class TestLoadGenreMap:
    """Tests for load_genre_map function."""

    def test_load_valid_tsv(self):
        """Test loading a valid genre TSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "genre.tsv"
            tsv_path.write_text(
                "Artist1/Song1\tRock\n"
                "Artist1/Song2\tJazz\n"
                "Artist2/Song1\tPop\n"
            )

            genre_map = load_genre_map(str(tsv_path))

            assert genre_map["Artist1/Song1"] == "Rock"
            assert genre_map["Artist1/Song2"] == "Jazz"
            assert genre_map["Artist2/Song1"] == "Pop"
            assert len(genre_map) == 3

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file returns empty dict."""
        genre_map = load_genre_map("/nonexistent/path/genre.tsv")
        assert genre_map == {}

    def test_load_empty_lines(self):
        """Test that empty lines are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "genre.tsv"
            tsv_path.write_text(
                "Artist1/Song1\tRock\n"
                "\n"
                "Artist2/Song1\tJazz\n"
                "\n"
            )

            genre_map = load_genre_map(str(tsv_path))
            assert len(genre_map) == 2


class TestFindMidiFiles:
    """Tests for find_midi_files function."""

    def test_find_files_in_directory(self):
        """Test finding MIDI files recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            artist_dir = Path(tmpdir) / "Artist1"
            artist_dir.mkdir()

            # Create dummy MIDI files
            (artist_dir / "song1.mid").write_bytes(b"dummy")
            (artist_dir / "song2.midi").write_bytes(b"dummy")
            (artist_dir / "readme.txt").write_bytes(b"not midi")

            midi_files = find_midi_files([tmpdir])

            assert len(midi_files) == 2
            assert all(f.suffix.lower() in [".mid", ".midi"] for f in midi_files)

    def test_find_files_with_limit(self):
        """Test limiting number of files returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                (Path(tmpdir) / f"song{i}.mid").write_bytes(b"dummy")

            midi_files = find_midi_files([tmpdir], max_files=3)
            assert len(midi_files) == 3

    def test_find_files_nonexistent_dir(self):
        """Test that nonexistent directories are skipped."""
        midi_files = find_midi_files(["/nonexistent/path"])
        assert midi_files == []


class TestGetGenreForFile:
    """Tests for get_genre_for_file function."""

    def test_get_genre_basic(self):
        """Test basic genre lookup."""
        genre_map = {
            "Artist1/Song1": "Rock",
            "Artist2/Song2": "Jazz",
        }

        filepath = Path("/data/Artist1/Song1.mid")
        genre = get_genre_for_file(filepath, genre_map)
        assert genre == "Rock"

    def test_get_genre_with_version_suffix(self):
        """Test genre lookup strips version suffix like .1, .2."""
        genre_map = {"Artist1/Song1": "Rock"}

        filepath = Path("/data/Artist1/Song1.1.mid")
        genre = get_genre_for_file(filepath, genre_map)
        assert genre == "Rock"

        filepath2 = Path("/data/Artist1/Song1.10.mid")
        genre2 = get_genre_for_file(filepath2, genre_map)
        assert genre2 == "Rock"

    def test_get_genre_not_found(self):
        """Test returns None when genre not in map."""
        genre_map = {"Artist1/Song1": "Rock"}

        filepath = Path("/data/Artist2/Unknown.mid")
        genre = get_genre_for_file(filepath, genre_map)
        assert genre is None


class TestPassesFilters:
    """Tests for passes_filters function."""

    def _create_mock_music(self, num_tracks=2, notes_per_track=10, resolution=24):
        """Helper to create a mock Music object."""
        music = muspy.Music(resolution=resolution)
        music.tempos = [muspy.Tempo(time=0, qpm=120)]

        for i in range(num_tracks):
            track = muspy.Track(program=i, is_drum=False)
            for j in range(notes_per_track):
                track.notes.append(
                    muspy.Note(time=j * 24, pitch=60 + j, duration=24, velocity=64)
                )
            music.tracks.append(track)

        return music

    def test_passes_all_filters(self):
        """Test music that passes all filters."""
        music = self._create_mock_music(num_tracks=4, notes_per_track=20)
        config = MusicDatasetConfig(
            name="test",
            input_dirs=["dir"],
            output_path="out.h5",
            min_tracks=1,
            max_tracks=8,
            min_notes_per_track=5,
        )

        passes, reason = passes_filters(music, config)
        assert passes is True
        assert reason == ""

    def test_fails_too_few_tracks(self):
        """Test filtering by minimum tracks."""
        music = self._create_mock_music(num_tracks=1)
        config = MusicDatasetConfig(
            name="test",
            input_dirs=["dir"],
            output_path="out.h5",
            min_tracks=3,
        )

        passes, reason = passes_filters(music, config)
        assert passes is False
        assert "too_few_tracks" in reason

    def test_fails_too_many_tracks(self):
        """Test filtering by maximum tracks."""
        music = self._create_mock_music(num_tracks=10)
        config = MusicDatasetConfig(
            name="test",
            input_dirs=["dir"],
            output_path="out.h5",
            max_tracks=5,
        )

        passes, reason = passes_filters(music, config)
        assert passes is False
        assert "too_many_tracks" in reason

    def test_fails_too_few_notes_per_track(self):
        """Test filtering by minimum notes per track."""
        music = self._create_mock_music(num_tracks=2, notes_per_track=3)
        config = MusicDatasetConfig(
            name="test",
            input_dirs=["dir"],
            output_path="out.h5",
            min_notes_per_track=10,
        )

        passes, reason = passes_filters(music, config)
        assert passes is False
        assert "too_few_notes" in reason


class TestBuildDataset:
    """Integration tests for build_dataset function."""

    def test_build_empty_directory(self):
        """Test building from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MusicDatasetConfig(
                name="empty_test",
                input_dirs=[tmpdir],
                output_path=str(Path(tmpdir) / "out.h5"),
                verbose=False,
            )

            dataset = build_dataset(config)
            assert len(dataset) == 0

    def test_build_with_genre_filter(self):
        """Test building with genre filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create genre TSV
            tsv_path = Path(tmpdir) / "genre.tsv"
            tsv_path.write_text(
                "TestArtist/RockSong\tRock\n"
                "TestArtist/JazzSong\tJazz\n"
            )

            config = MusicDatasetConfig(
                name="genre_test",
                input_dirs=[tmpdir],
                output_path=str(Path(tmpdir) / "out.h5"),
                genre_tsv_path=str(tsv_path),
                allowed_genres=["Rock"],  # Only Rock
                verbose=False,
            )

            # No actual MIDI files, but config should work
            dataset = build_dataset(config)
            assert len(dataset) == 0  # No files, but no errors
