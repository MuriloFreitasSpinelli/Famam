"""Tests for preprocessing module."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import muspy

from ..configs import PreprocessingConfig
from ..preprocessing import (
    adjust_resolution,
    quantize_music,
    remove_empty_tracks,
    segment_music,
    transpose_music,
    generate_transpositions,
    vary_tempo,
    generate_tempo_variations,
    preprocess_music,
)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_create_default_config(self):
        """Test creating config with defaults."""
        config = PreprocessingConfig()

        assert config.target_resolution == 24
        assert config.quantize is True
        assert config.segment_length is None
        assert config.enable_transposition is False
        assert config.enable_tempo_variation is False

    def test_create_config_with_segmentation(self):
        """Test creating config with segmentation."""
        config = PreprocessingConfig(
            segment_length=1000,
            max_padding_ratio=0.5,
        )

        assert config.segment_length == 1000
        assert config.max_padding_ratio == 0.5

    def test_create_config_with_augmentation(self):
        """Test creating config with augmentation enabled."""
        config = PreprocessingConfig.with_augmentation()

        assert config.enable_transposition is True
        assert config.enable_tempo_variation is True

    def test_validation_invalid_resolution(self):
        """Test validation rejects invalid resolution."""
        with pytest.raises(ValueError, match="target_resolution must be at least 1"):
            PreprocessingConfig(target_resolution=0)

    def test_validation_invalid_padding_ratio(self):
        """Test validation rejects invalid padding ratio."""
        with pytest.raises(ValueError, match="max_padding_ratio must be between"):
            PreprocessingConfig(max_padding_ratio=1.5)

    def test_validation_invalid_tempo_range(self):
        """Test validation rejects invalid tempo range."""
        with pytest.raises(ValueError, match="tempo_variation_range"):
            PreprocessingConfig(tempo_variation_range=(1.1, 0.9))

    def test_save_and_load(self):
        """Test saving and loading config."""
        config = PreprocessingConfig(
            segment_length=500,
            enable_transposition=True,
            transposition_semitones=(-3, -2, -1, 1, 2, 3),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preproc.json"
            config.save(str(path))

            loaded = PreprocessingConfig.load(str(path))

            assert loaded.segment_length == 500
            assert loaded.enable_transposition is True
            assert loaded.transposition_semitones == (-3, -2, -1, 1, 2, 3)


class TestAdjustResolution:
    """Tests for adjust_resolution function."""

    def _create_music(self, resolution=24, notes=5):
        """Helper to create a Music object."""
        music = muspy.Music(resolution=resolution)
        track = muspy.Track(program=0, is_drum=False)
        for i in range(notes):
            track.notes.append(
                muspy.Note(time=i * resolution, pitch=60 + i, duration=resolution, velocity=64)
            )
        music.tracks.append(track)
        return music

    def test_same_resolution_no_change(self):
        """Test that same resolution returns unchanged music."""
        music = self._create_music(resolution=24)
        result = adjust_resolution(music, 24)

        assert result.resolution == 24
        assert len(result.tracks[0].notes) == 5

    def test_double_resolution(self):
        """Test doubling resolution scales note times."""
        music = self._create_music(resolution=24)
        result = adjust_resolution(music, 48)

        assert result.resolution == 48
        # Note times should be doubled
        assert result.tracks[0].notes[0].time == 0
        assert result.tracks[0].notes[1].time == 48  # was 24


class TestQuantizeMusic:
    """Tests for quantize_music function."""

    def test_quantize_to_grid(self):
        """Test quantizing notes to a grid."""
        music = muspy.Music(resolution=24)
        track = muspy.Track(program=0, is_drum=False)
        # Note slightly off-grid
        track.notes.append(muspy.Note(time=25, pitch=60, duration=23, velocity=64))
        music.tracks.append(track)

        result = quantize_music(music, grid=24)

        # Should snap to nearest grid point
        assert result.tracks[0].notes[0].time == 24
        assert result.tracks[0].notes[0].duration == 24

    def test_quantize_minimum_duration(self):
        """Test that duration is at least grid size."""
        music = muspy.Music(resolution=24)
        track = muspy.Track(program=0, is_drum=False)
        track.notes.append(muspy.Note(time=0, pitch=60, duration=1, velocity=64))
        music.tracks.append(track)

        result = quantize_music(music, grid=24)

        assert result.tracks[0].notes[0].duration == 24  # minimum is grid size


class TestRemoveEmptyTracks:
    """Tests for remove_empty_tracks function."""

    def test_removes_empty_tracks(self):
        """Test that empty tracks are removed."""
        music = muspy.Music(resolution=24)

        # Track with notes
        track1 = muspy.Track(program=0, is_drum=False)
        track1.notes.append(muspy.Note(time=0, pitch=60, duration=24, velocity=64))
        music.tracks.append(track1)

        # Empty track
        track2 = muspy.Track(program=1, is_drum=False)
        music.tracks.append(track2)

        result = remove_empty_tracks(music)

        assert len(result.tracks) == 1
        assert result.tracks[0].program == 0

    def test_keeps_non_empty_tracks(self):
        """Test that non-empty tracks are preserved."""
        music = muspy.Music(resolution=24)
        for i in range(3):
            track = muspy.Track(program=i, is_drum=False)
            track.notes.append(muspy.Note(time=0, pitch=60, duration=24, velocity=64))
            music.tracks.append(track)

        result = remove_empty_tracks(music)

        assert len(result.tracks) == 3


class TestSegmentMusic:
    """Tests for segment_music function."""

    def _create_long_music(self, length_ticks=2400):
        """Helper to create a music object with notes spanning length_ticks."""
        music = muspy.Music(resolution=24)
        music.tempos = [muspy.Tempo(time=0, qpm=120)]
        track = muspy.Track(program=0, is_drum=False)

        # Add notes throughout the duration
        for t in range(0, length_ticks, 24):
            track.notes.append(muspy.Note(time=t, pitch=60, duration=24, velocity=64))

        music.tracks.append(track)
        return music

    def test_segment_into_chunks(self):
        """Test segmenting music into fixed-length chunks."""
        music = self._create_long_music(length_ticks=2400)

        segments = segment_music(music, segment_length=1000)

        # 2400 ticks / 1000 = 2 full segments + 400 remainder
        # 400/1000 = 0.6 padding ratio < 0.7, so 3 segments total
        assert len(segments) == 3

    def test_segment_discards_high_padding(self):
        """Test that segments with >70% padding are discarded."""
        music = self._create_long_music(length_ticks=1100)

        segments = segment_music(music, segment_length=1000, max_padding_ratio=0.7)

        # 1100 ticks = 1 full segment + 100 remainder
        # 100/1000 = 0.9 padding ratio > 0.7, so discard remainder
        assert len(segments) == 1

    def test_segment_keeps_low_padding(self):
        """Test that segments with acceptable padding are kept."""
        music = self._create_long_music(length_ticks=1500)

        segments = segment_music(music, segment_length=1000, max_padding_ratio=0.7)

        # 1500 ticks = 1 full segment + 500 remainder
        # 500/1000 = 0.5 padding ratio < 0.7, so keep it
        assert len(segments) == 2

    def test_segment_shifts_note_times(self):
        """Test that note times are shifted to segment start."""
        music = muspy.Music(resolution=24)
        music.tempos = [muspy.Tempo(time=0, qpm=120)]
        track = muspy.Track(program=0, is_drum=False)
        track.notes.append(muspy.Note(time=1000, pitch=60, duration=24, velocity=64))
        track.notes.append(muspy.Note(time=1500, pitch=62, duration=24, velocity=64))
        music.tracks.append(track)

        segments = segment_music(music, segment_length=1000, max_padding_ratio=0.99)

        # First segment (0-1000) should be empty
        # Second segment (1000-2000) should have notes shifted
        assert len(segments) >= 2
        if len(segments[1].tracks) > 0:
            # Note at 1000 should now be at 0 in second segment
            assert segments[1].tracks[0].notes[0].time == 0


class TestTransposeMusic:
    """Tests for transpose_music function."""

    def _create_music(self):
        """Helper to create a Music object."""
        music = muspy.Music(resolution=24)
        track = muspy.Track(program=0, is_drum=False)
        track.notes.append(muspy.Note(time=0, pitch=60, duration=24, velocity=64))
        track.notes.append(muspy.Note(time=24, pitch=64, duration=24, velocity=64))
        music.tracks.append(track)
        return music

    def test_transpose_up(self):
        """Test transposing up by semitones."""
        music = self._create_music()
        result = transpose_music(music, semitones=5)

        assert result.tracks[0].notes[0].pitch == 65  # 60 + 5
        assert result.tracks[0].notes[1].pitch == 69  # 64 + 5

    def test_transpose_down(self):
        """Test transposing down by semitones."""
        music = self._create_music()
        result = transpose_music(music, semitones=-3)

        assert result.tracks[0].notes[0].pitch == 57  # 60 - 3
        assert result.tracks[0].notes[1].pitch == 61  # 64 - 3

    def test_transpose_clamps_to_valid_range(self):
        """Test that pitches are clamped to 0-127."""
        music = muspy.Music(resolution=24)
        track = muspy.Track(program=0, is_drum=False)
        track.notes.append(muspy.Note(time=0, pitch=5, duration=24, velocity=64))
        track.notes.append(muspy.Note(time=24, pitch=125, duration=24, velocity=64))
        music.tracks.append(track)

        result = transpose_music(music, semitones=-10)

        assert result.tracks[0].notes[0].pitch == 0  # Clamped from -5
        assert result.tracks[0].notes[1].pitch == 115

    def test_transpose_skips_drum_tracks(self):
        """Test that drum tracks are not transposed."""
        music = muspy.Music(resolution=24)
        drum_track = muspy.Track(program=0, is_drum=True)
        drum_track.notes.append(muspy.Note(time=0, pitch=36, duration=24, velocity=64))
        music.tracks.append(drum_track)

        result = transpose_music(music, semitones=5)

        # Drum pitch should be unchanged
        assert result.tracks[0].notes[0].pitch == 36


class TestGenerateTranspositions:
    """Tests for generate_transpositions function."""

    def _create_music(self):
        music = muspy.Music(resolution=24)
        track = muspy.Track(program=0, is_drum=False)
        track.notes.append(muspy.Note(time=0, pitch=60, duration=24, velocity=64))
        music.tracks.append(track)
        return music

    def test_generates_multiple_transpositions(self):
        """Test generating multiple transposed versions."""
        music = self._create_music()
        transpositions = generate_transpositions(music, semitones=(-2, -1, 1, 2))

        assert len(transpositions) == 4
        assert transpositions[0].tracks[0].notes[0].pitch == 58  # -2
        assert transpositions[1].tracks[0].notes[0].pitch == 59  # -1
        assert transpositions[2].tracks[0].notes[0].pitch == 61  # +1
        assert transpositions[3].tracks[0].notes[0].pitch == 62  # +2


class TestVaryTempo:
    """Tests for vary_tempo function."""

    def _create_music(self):
        music = muspy.Music(resolution=24)
        track = muspy.Track(program=0, is_drum=False)
        track.notes.append(muspy.Note(time=0, pitch=60, duration=48, velocity=64))
        track.notes.append(muspy.Note(time=48, pitch=62, duration=48, velocity=64))
        music.tracks.append(track)
        return music

    def test_tempo_slower(self):
        """Test slowing down tempo (factor < 1)."""
        music = self._create_music()
        result = vary_tempo(music, factor=0.5)

        # Times and durations should double (slower = stretched)
        assert result.tracks[0].notes[0].time == 0
        assert result.tracks[0].notes[0].duration == 96  # 48 / 0.5
        assert result.tracks[0].notes[1].time == 96  # 48 / 0.5

    def test_tempo_faster(self):
        """Test speeding up tempo (factor > 1)."""
        music = self._create_music()
        result = vary_tempo(music, factor=2.0)

        # Times and durations should halve (faster = compressed)
        assert result.tracks[0].notes[0].duration == 24  # 48 / 2
        assert result.tracks[0].notes[1].time == 24  # 48 / 2


class TestGenerateTempoVariations:
    """Tests for generate_tempo_variations function."""

    def _create_music(self):
        music = muspy.Music(resolution=24)
        track = muspy.Track(program=0, is_drum=False)
        track.notes.append(muspy.Note(time=0, pitch=60, duration=48, velocity=64))
        music.tracks.append(track)
        return music

    def test_generates_variations(self):
        """Test generating tempo variations."""
        music = self._create_music()
        variations = generate_tempo_variations(
            music, variation_range=(0.8, 1.2), num_variations=3
        )

        # Should generate 3 variations (excluding 1.0 if present)
        assert len(variations) >= 2

    def test_excludes_factor_one(self):
        """Test that factor=1.0 is excluded."""
        music = self._create_music()
        variations = generate_tempo_variations(
            music, variation_range=(0.9, 1.1), num_variations=3
        )

        # All variations should have different durations from original
        original_duration = music.tracks[0].notes[0].duration
        for var in variations:
            assert var.tracks[0].notes[0].duration != original_duration


class TestPreprocessMusic:
    """Integration tests for preprocess_music function."""

    def _create_music(self, length_ticks=1000):
        music = muspy.Music(resolution=24)
        music.tempos = [muspy.Tempo(time=0, qpm=120)]
        track = muspy.Track(program=0, is_drum=False)
        for t in range(0, length_ticks, 48):
            track.notes.append(muspy.Note(time=t, pitch=60, duration=24, velocity=64))
        music.tracks.append(track)
        return music

    def test_preprocess_no_options(self):
        """Test preprocessing with no special options."""
        music = self._create_music()
        config = PreprocessingConfig()

        results = preprocess_music(music, config)

        assert len(results) == 1
        assert results[0].resolution == 24

    def test_preprocess_with_segmentation(self):
        """Test preprocessing with segmentation."""
        music = self._create_music(length_ticks=2500)
        config = PreprocessingConfig(segment_length=1000)

        results = preprocess_music(music, config)

        # 2500 / 1000 = 2 full + 500 remainder (50% padding < 70%)
        assert len(results) == 3

    def test_preprocess_with_transposition(self):
        """Test preprocessing with transposition enabled."""
        music = self._create_music()
        config = PreprocessingConfig(
            enable_transposition=True,
            transposition_semitones=(-1, 1),  # 2 transpositions
        )

        results = preprocess_music(music, config)

        # 1 original + 2 transpositions = 3
        assert len(results) == 3

    def test_preprocess_with_all_augmentation(self):
        """Test preprocessing with all augmentation enabled."""
        music = self._create_music()
        config = PreprocessingConfig(
            enable_transposition=True,
            transposition_semitones=(-1, 1),  # 2 transpositions
            enable_tempo_variation=True,
            tempo_variation_steps=2,  # ~2 tempo variations (excluding 1.0)
        )

        results = preprocess_music(music, config)

        # This should produce: original + transpositions + tempo variations
        # + transposition×tempo combinations
        assert len(results) > 3

    def test_preprocess_empty_music_returns_empty(self):
        """Test that empty music returns empty list."""
        music = muspy.Music(resolution=24)
        music.tracks.append(muspy.Track(program=0, is_drum=False))
        config = PreprocessingConfig()

        results = preprocess_music(music, config)

        assert len(results) == 0
