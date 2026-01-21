"""Tests for MusicDatasetConfig."""

import pytest
import tempfile
import json
from pathlib import Path

from ..configs import MusicDatasetConfig


class TestMusicDatasetConfig:
    """Tests for MusicDatasetConfig creation and validation."""

    def test_create_basic_config(self):
        """Test creating a config with required fields."""
        config = MusicDatasetConfig(
            name="test_dataset",
            input_dirs=["data/midi"],
            output_path="data/output.h5",
        )

        assert config.name == "test_dataset"
        assert config.input_dirs == ["data/midi"]
        assert config.output_path == "data/output.h5"
        assert config.resolution == 24  # default
        assert config.max_time_steps == 1000  # default

    def test_create_config_with_all_options(self):
        """Test creating a config with all options specified."""
        config = MusicDatasetConfig(
            name="full_config",
            input_dirs=["dir1", "dir2"],
            output_path="output.h5",
            genre_tsv_path="genre.tsv",
            allowed_genres=["Rock", "Jazz"],
            min_tracks=2,
            max_tracks=8,
            min_notes_per_track=10,
            min_duration_seconds=30.0,
            max_duration_seconds=300.0,
            resolution=48,
            max_time_steps=2000,
            max_samples=100,
            random_seed=42,
            verbose=False,
        )

        assert config.allowed_genres == ["Rock", "Jazz"]
        assert config.min_tracks == 2
        assert config.max_tracks == 8
        assert config.resolution == 48
        assert config.max_time_steps == 2000

    def test_validation_min_tracks(self):
        """Test that min_tracks must be at least 1."""
        with pytest.raises(ValueError, match="min_tracks must be at least 1"):
            MusicDatasetConfig(
                name="test",
                input_dirs=["dir"],
                output_path="out.h5",
                min_tracks=0,
            )

    def test_validation_max_tracks_less_than_min(self):
        """Test that max_tracks must be >= min_tracks."""
        with pytest.raises(ValueError, match="max_tracks must be >= min_tracks"):
            MusicDatasetConfig(
                name="test",
                input_dirs=["dir"],
                output_path="out.h5",
                min_tracks=5,
                max_tracks=3,
            )

    def test_validation_negative_notes_per_track(self):
        """Test that min_notes_per_track must be non-negative."""
        with pytest.raises(ValueError, match="min_notes_per_track must be non-negative"):
            MusicDatasetConfig(
                name="test",
                input_dirs=["dir"],
                output_path="out.h5",
                min_notes_per_track=-1,
            )

    def test_validation_duration_constraints(self):
        """Test that max_duration must be >= min_duration."""
        with pytest.raises(ValueError, match="max_duration_seconds must be >= min_duration_seconds"):
            MusicDatasetConfig(
                name="test",
                input_dirs=["dir"],
                output_path="out.h5",
                min_duration_seconds=100.0,
                max_duration_seconds=50.0,
            )

    def test_validation_resolution(self):
        """Test that resolution must be at least 1."""
        with pytest.raises(ValueError, match="resolution must be at least 1"):
            MusicDatasetConfig(
                name="test",
                input_dirs=["dir"],
                output_path="out.h5",
                resolution=0,
            )

    def test_save_and_load(self):
        """Test saving and loading config from JSON."""
        config = MusicDatasetConfig(
            name="save_test",
            input_dirs=["dir1", "dir2"],
            output_path="output.h5",
            allowed_genres=["Rock"],
            resolution=48,
            max_time_steps=1500,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config.save(str(config_path))

            # Verify file exists
            assert config_path.exists()

            # Load and verify
            loaded = MusicDatasetConfig.load(str(config_path))
            assert loaded.name == config.name
            assert loaded.input_dirs == config.input_dirs
            assert loaded.allowed_genres == config.allowed_genres
            assert loaded.resolution == config.resolution
            assert loaded.max_time_steps == config.max_time_steps

    def test_default_factory(self):
        """Test the default() class method."""
        config = MusicDatasetConfig.default(
            name="default_test",
            input_dirs=["data/midi"],
            output_path="data/out.h5",
        )

        assert config.name == "default_test"
        assert config.resolution == 24
        assert config.max_time_steps == 1000
        assert config.verbose is True
