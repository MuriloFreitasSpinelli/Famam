"""
Unit tests for src_v4.data_preprocessing module.

Tests:
    - MusicDatasetConfig: validation, serialization
    - EventEncoder: encoding/decoding tokens
    - REMIEncoder: bar/position encoding
    - MultiTrackEncoder: interleaved multi-track encoding

Author: Murilo de Freitas Spinelli
"""

import pytest
import numpy as np
from pathlib import Path


class TestMusicDatasetConfig:
    """Tests for MusicDatasetConfig."""

    def test_default_config(self):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        config = MusicDatasetConfig()

        assert config.name == "music_dataset"
        assert config.resolution == 24
        assert config.max_seq_length == 2048
        assert config.quantize is True
        assert config.encoder_type == "remi"

    def test_custom_config(self):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        config = MusicDatasetConfig(
            name="test_dataset",
            resolution=48,
            max_seq_length=1024,
            encoder_type="event",
        )

        assert config.name == "test_dataset"
        assert config.resolution == 48
        assert config.max_seq_length == 1024
        assert config.encoder_type == "event"

    def test_validation_min_tracks(self):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        with pytest.raises(ValueError, match="min_tracks"):
            MusicDatasetConfig(min_tracks=0)

    def test_validation_max_tracks(self):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        with pytest.raises(ValueError, match="max_tracks"):
            MusicDatasetConfig(min_tracks=5, max_tracks=3)

    def test_validation_resolution(self):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        with pytest.raises(ValueError, match="resolution"):
            MusicDatasetConfig(resolution=0)

    def test_validation_padding_ratio(self):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        with pytest.raises(ValueError, match="max_padding_ratio"):
            MusicDatasetConfig(max_padding_ratio=1.5)

    def test_ticks_per_bar(self):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        config = MusicDatasetConfig(resolution=24, time_signature=(4, 4))
        assert config.ticks_per_bar == 96  # 24 * 4

        config = MusicDatasetConfig(resolution=24, time_signature=(3, 4))
        assert config.ticks_per_bar == 72  # 24 * 3

    def test_save_load(self, temp_dir):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        filepath = temp_dir / "config.json"

        config1 = MusicDatasetConfig(
            name="test",
            resolution=48,
            allowed_genres=["Rock", "Jazz"],
            enable_transposition=True,
        )
        config1.save(str(filepath))

        assert filepath.exists()

        config2 = MusicDatasetConfig.load(str(filepath))

        assert config2.name == config1.name
        assert config2.resolution == config1.resolution
        assert config2.allowed_genres == config1.allowed_genres
        assert config2.enable_transposition == config1.enable_transposition

    def test_repr(self):
        from src_v4.data_preprocessing.music_dataset_config import MusicDatasetConfig

        config = MusicDatasetConfig(name="test")
        repr_str = repr(config)

        assert "MusicDatasetConfig" in repr_str
        assert "test" in repr_str


class TestEventEncoder:
    """Tests for EventEncoder."""

    def test_init(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5, num_instruments=129)

        assert encoder.vocab_size > 0
        assert encoder.vocabulary.num_genres == 5
        assert encoder.vocabulary.num_instruments == 129

    def test_special_tokens(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)

        assert encoder.pad_token_id == encoder.vocabulary.PAD_TOKEN
        assert encoder.bos_token_id == encoder.vocabulary.BOS_TOKEN
        assert encoder.eos_token_id == encoder.vocabulary.EOS_TOKEN

    def test_encode_note_on(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)

        token = encoder.vocabulary.encode_note_on(60)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "note_on"
        assert value == 60

    def test_encode_note_off(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)

        token = encoder.vocabulary.encode_note_off(60)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "note_off"
        assert value == 60

    def test_encode_time_shift(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)

        token = encoder.vocabulary.encode_time_shift(50)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "time_shift"
        assert value == 50

    def test_encode_time_shift_clamped(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        max_shift = encoder.vocabulary.MAX_TIME_SHIFT

        token = encoder.vocabulary.encode_time_shift(max_shift + 50)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "time_shift"
        assert value == max_shift

    def test_encode_genre(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)

        token = encoder.vocabulary.encode_genre(3)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "genre"
        assert value == 3

    def test_encode_instrument(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5, num_instruments=129)

        token = encoder.vocabulary.encode_instrument(33)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "instrument"
        assert value == 33

    def test_track_to_events(self, mock_single_track_music):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        track = mock_single_track_music.tracks[0]

        events = encoder.track_to_events(track)

        assert len(events) > 0
        event_types = [e[0] for e in events]
        assert "note_on" in event_types
        assert "note_off" in event_types

    def test_encode_track(self, mock_single_track_music):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5, num_instruments=129)
        track = mock_single_track_music.tracks[0]

        encoded = encoder.encode_track(
            track=track,
            genre_id=0,
            instrument_id=25,
            max_length=512,
        )

        assert encoded.token_ids.shape == (512,)
        assert encoded.attention_mask.shape == (512,)
        assert encoded.token_ids[0] == encoder.bos_token_id

    def test_decode_tokens(self, mock_single_track_music):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5, num_instruments=129)
        track = mock_single_track_music.tracks[0]

        encoded = encoder.encode_track(
            track=track,
            genre_id=0,
            instrument_id=25,
            max_length=512,
        )

        events = encoder.decode_tokens(encoded.token_ids, skip_special=True)

        assert len(events) > 0
        event_types = [e[0] for e in events]
        assert "note_on" in event_types

    def test_create_conditioning_tokens(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5, num_instruments=129)

        tokens = encoder.create_conditioning_tokens(genre_id=2, instrument_id=33)

        assert len(tokens) == 3
        assert tokens[0] == encoder.bos_token_id

    def test_create_labels(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        input_ids = np.array([1, 10, 20, 30, 2, 0, 0], dtype=np.int32)

        labels = encoder.create_labels(input_ids)

        assert len(labels) == len(input_ids)
        assert labels[0] == 10  # Shifted by 1
        assert labels[-1] == encoder.pad_token_id

    def test_get_state_from_state(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder1 = EventEncoder(num_genres=5, num_instruments=129, encode_velocity=True)
        state = encoder1.get_state()

        encoder2 = EventEncoder.from_state(state)

        assert encoder2.vocab_size == encoder1.vocab_size
        assert encoder2.encode_velocity == encoder1.encode_velocity

    def test_repr(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        repr_str = repr(encoder)

        assert "EventEncoder" in repr_str
        assert "num_genres=5" in repr_str


class TestREMIEncoder:
    """Tests for REMIEncoder."""

    def test_init(self):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5, resolution=24)

        assert encoder.vocab_size > 0
        assert encoder.vocabulary.num_genres == 5

    def test_special_tokens(self):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5)

        assert encoder.pad_token_id == encoder.vocabulary.PAD_TOKEN
        assert encoder.bos_token_id == encoder.vocabulary.BOS_TOKEN
        assert encoder.eos_token_id == encoder.vocabulary.EOS_TOKEN

    def test_encode_bar(self):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5)

        token = encoder.vocabulary.encode_bar()
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "bar"

    def test_encode_position(self):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5)

        token = encoder.vocabulary.encode_position(8)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "position"
        assert value == 8

    def test_encode_pitch(self):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5)

        token = encoder.vocabulary.encode_pitch(60)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "pitch"
        assert value == 60

    def test_encode_duration(self):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5)

        token = encoder.vocabulary.encode_duration(8)
        event_type, value = encoder.vocabulary.decode_token(token)

        assert event_type == "duration"
        assert value == 8

    def test_encode_track(self, mock_single_track_music):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5, num_instruments=129, resolution=24)
        track = mock_single_track_music.tracks[0]

        encoded = encoder.encode_track(
            track=track,
            genre_id=0,
            instrument_id=25,
            max_length=512,
        )

        assert encoded.token_ids.shape == (512,)
        assert encoded.attention_mask.shape == (512,)

    def test_get_state_from_state(self):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder1 = REMIEncoder(num_genres=5, resolution=48, positions_per_bar=16)
        state = encoder1.get_state()

        encoder2 = REMIEncoder.from_state(state)

        assert encoder2.vocab_size == encoder1.vocab_size
        assert encoder2.vocabulary.resolution == 48

    def test_repr(self):
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5)
        repr_str = repr(encoder)

        assert "REMIEncoder" in repr_str


class TestMultiTrackEncoder:
    """Tests for MultiTrackEncoder."""

    def test_init(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5, resolution=24)

        assert encoder.vocab_size > 0
        assert encoder.num_genres == 5
        assert encoder.resolution == 24

    def test_special_tokens(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5)

        assert encoder.pad_token_id == encoder.PAD_TOKEN
        assert encoder.bos_token_id == encoder.BOS_TOKEN
        assert encoder.eos_token_id == encoder.EOS_TOKEN

    def test_bar_token(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5)

        token = encoder.bar_token(0)
        event_type, value = encoder.decode_token(token)

        assert event_type == "bar"
        assert value == 0

    def test_position_token(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5)

        token = encoder.position_token(8)
        event_type, value = encoder.decode_token(token)

        assert event_type == "position"
        assert value == 8

    def test_instrument_token(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5)

        token = encoder.instrument_token(33)
        event_type, value = encoder.decode_token(token)

        assert event_type == "instrument"
        assert value == 33

    def test_pitch_token(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5)

        token = encoder.pitch_token(60)
        event_type, value = encoder.decode_token(token)

        assert event_type == "pitch"
        assert value == 60

    def test_encode_music(self, mock_music):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5, resolution=24)

        encoded = encoder.encode_music(
            music=mock_music,
            genre_id=0,
            max_length=1024,
        )

        assert encoded.token_ids.shape == (1024,)
        assert encoded.attention_mask.shape == (1024,)
        assert encoded.token_ids[0] == encoder.BOS_TOKEN

    def test_decode_to_music(self, mock_music):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5, resolution=24)

        encoded = encoder.encode_music(
            music=mock_music,
            genre_id=0,
            max_length=1024,
        )

        decoded_music = encoder.decode_to_music(
            encoded.token_ids,
            resolution=24,
            tempo=120.0,
        )

        assert len(decoded_music.tracks) > 0
        total_notes = sum(len(t.notes) for t in decoded_music.tracks)
        assert total_notes > 0

    def test_create_conditioning_tokens(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5)

        tokens = encoder.create_conditioning_tokens(genre_id=2)

        assert len(tokens) >= 1
        assert tokens[0] == encoder.BOS_TOKEN

    def test_get_state_from_state(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder1 = MultiTrackEncoder(
            num_genres=5,
            resolution=48,
            max_bars=32,
            positions_per_bar=32,
        )
        state = encoder1.get_state()

        encoder2 = MultiTrackEncoder.from_state(state)

        assert encoder2.vocab_size == encoder1.vocab_size
        assert encoder2.resolution == encoder1.resolution
        assert encoder2.max_bars == encoder1.max_bars

    def test_repr(self):
        from src_v4.data_preprocessing.encoders import MultiTrackEncoder

        encoder = MultiTrackEncoder(num_genres=5)
        repr_str = repr(encoder)

        assert "MultiTrackEncoder" in repr_str
        assert "num_genres=5" in repr_str


class TestBaseEncoder:
    """Tests for BaseEncoder abstract class."""

    def test_pad_sequence(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)

        tokens = [1, 10, 20, 30, 2]
        padded, mask = encoder.pad_sequence(tokens, max_length=10)

        assert len(padded) == 10
        assert len(mask) == 10
        assert padded[-1] == encoder.pad_token_id
        assert mask[4] == 1  # Last real token
        assert mask[5] == 0  # First padding

    def test_pad_sequence_truncate(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)

        tokens = list(range(100))
        padded, mask = encoder.pad_sequence(tokens, max_length=10)

        assert len(padded) == 10
        assert padded[-1] == encoder.eos_token_id

    def test_is_special_token(self):
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)

        assert encoder.is_special_token(encoder.pad_token_id)
        assert encoder.is_special_token(encoder.bos_token_id)
        assert encoder.is_special_token(encoder.eos_token_id)
        assert not encoder.is_special_token(100)
