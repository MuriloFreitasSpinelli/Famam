"""
Pytest fixtures for src_v4 tests.

Author: Murilo de Freitas Spinelli
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_genres():
    """Sample genre list for testing."""
    return ["Rock", "Jazz", "Classical", "Electronic"]


@pytest.fixture
def sample_artists():
    """Sample artist list for testing."""
    return ["Artist A", "Artist B", "Artist C"]


@pytest.fixture
def mock_music():
    """Create a mock muspy.Music object for testing."""
    try:
        import muspy

        track1 = muspy.Track(
            program=0,
            is_drum=False,
            name="Piano",
            notes=[
                muspy.Note(time=0, pitch=60, duration=24, velocity=80),
                muspy.Note(time=24, pitch=62, duration=24, velocity=75),
                muspy.Note(time=48, pitch=64, duration=24, velocity=70),
            ]
        )

        track2 = muspy.Track(
            program=33,
            is_drum=False,
            name="Bass",
            notes=[
                muspy.Note(time=0, pitch=36, duration=48, velocity=90),
                muspy.Note(time=48, pitch=38, duration=48, velocity=85),
            ]
        )

        drum_track = muspy.Track(
            program=0,
            is_drum=True,
            name="Drums",
            notes=[
                muspy.Note(time=0, pitch=36, duration=12, velocity=100),
                muspy.Note(time=24, pitch=38, duration=12, velocity=90),
                muspy.Note(time=48, pitch=36, duration=12, velocity=100),
                muspy.Note(time=72, pitch=38, duration=12, velocity=90),
            ]
        )

        music = muspy.Music(
            resolution=24,
            tempos=[muspy.Tempo(time=0, qpm=120)],
            tracks=[track1, track2, drum_track],
        )

        return music
    except ImportError:
        pytest.skip("muspy not installed")


@pytest.fixture
def mock_single_track_music():
    """Create a mock muspy.Music with single track."""
    try:
        import muspy

        track = muspy.Track(
            program=25,
            is_drum=False,
            name="Guitar",
            notes=[
                muspy.Note(time=0, pitch=64, duration=24, velocity=80),
                muspy.Note(time=24, pitch=67, duration=24, velocity=75),
                muspy.Note(time=48, pitch=71, duration=48, velocity=70),
            ]
        )

        return muspy.Music(
            resolution=24,
            tempos=[muspy.Tempo(time=0, qpm=120)],
            tracks=[track],
        )
    except ImportError:
        pytest.skip("muspy not installed")


@pytest.fixture
def mock_empty_music():
    """Create a mock muspy.Music with no notes."""
    try:
        import muspy
        return muspy.Music(resolution=24, tracks=[])
    except ImportError:
        pytest.skip("muspy not installed")
