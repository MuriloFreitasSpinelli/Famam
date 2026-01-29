"""
Unit tests for data_preprocessing module.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockNote:
    """Mock muspy.Note for testing."""
    time: int
    pitch: int
    duration: int
    velocity: int = 64


@dataclass
class MockTrack:
    """Mock muspy.Track for testing."""
    notes: List[MockNote]
    program: int = 0
    is_drum: bool = False
    name: str = ""


# =============================================================================
# Tests for EventVocabulary
# =============================================================================

class TestEventVocabulary:
    """Tests for EventVocabulary class."""

    def test_import(self):
        """Test that EventVocabulary can be imported."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5, num_instruments=129)
        assert vocab is not None

    def test_vocab_size(self):
        """Test vocabulary size calculation."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=10, num_instruments=129)
        # 128 note_on + 128 note_off + 100 time_shift + 32 velocity + 3 special + 10 genres + 129 instruments
        expected = 128 + 128 + 100 + 32 + 3 + 10 + 129
        assert vocab.vocab_size == expected

    def test_encode_note_on(self):
        """Test note-on encoding."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5)

        assert vocab.encode_note_on(0) == 0
        assert vocab.encode_note_on(60) == 60
        assert vocab.encode_note_on(127) == 127
        # Test clamping
        assert vocab.encode_note_on(200) == 127
        assert vocab.encode_note_on(-5) == 0

    def test_encode_note_off(self):
        """Test note-off encoding."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5)

        assert vocab.encode_note_off(0) == 128
        assert vocab.encode_note_off(60) == 188
        assert vocab.encode_note_off(127) == 255

    def test_encode_time_shift(self):
        """Test time-shift encoding."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5)

        assert vocab.encode_time_shift(1) == 256
        assert vocab.encode_time_shift(50) == 305
        assert vocab.encode_time_shift(100) == 355
        # Test clamping
        assert vocab.encode_time_shift(150) == 355
        assert vocab.encode_time_shift(0) == 256  # min is 1

    def test_encode_velocity(self):
        """Test velocity encoding."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5)

        token = vocab.encode_velocity(64)
        assert 356 <= token < 388

    def test_encode_genre(self):
        """Test genre encoding."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5)

        assert vocab.encode_genre(0) == vocab.GENRE_OFFSET
        assert vocab.encode_genre(4) == vocab.GENRE_OFFSET + 4

        with pytest.raises(ValueError):
            vocab.encode_genre(10)  # Out of range

    def test_encode_instrument(self):
        """Test instrument encoding."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5, num_instruments=129)

        assert vocab.encode_instrument(0) == vocab.INSTRUMENT_OFFSET
        assert vocab.encode_instrument(128) == vocab.INSTRUMENT_OFFSET + 128

        with pytest.raises(ValueError):
            vocab.encode_instrument(200)  # Out of range

    def test_decode_token(self):
        """Test token decoding."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5, num_instruments=129)

        # Note on
        event_type, value = vocab.decode_token(60)
        assert event_type == 'note_on'
        assert value == 60

        # Note off
        event_type, value = vocab.decode_token(188)
        assert event_type == 'note_off'
        assert value == 60

        # Time shift
        event_type, value = vocab.decode_token(306)
        assert event_type == 'time_shift'
        assert value == 51

        # Special tokens
        assert vocab.decode_token(vocab.PAD_TOKEN) == ('pad', 0)
        assert vocab.decode_token(vocab.BOS_TOKEN) == ('bos', 0)
        assert vocab.decode_token(vocab.EOS_TOKEN) == ('eos', 0)

    def test_is_special_token(self):
        """Test special token checking."""
        from src_v4.data_preprocessing.encoders.event_encoder import EventVocabulary
        vocab = EventVocabulary(num_genres=5)

        assert vocab.is_special_token(vocab.PAD_TOKEN)
        assert vocab.is_special_token(vocab.BOS_TOKEN)
        assert vocab.is_special_token(vocab.EOS_TOKEN)
        assert not vocab.is_special_token(60)  # note_on


# =============================================================================
# Tests for EventEncoder
# =============================================================================

class TestEventEncoder:
    """Tests for EventEncoder class."""

    def test_import(self):
        """Test that EventEncoder can be imported."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)
        assert encoder is not None

    def test_vocab_size(self):
        """Test vocab_size property."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=10, num_instruments=129)
        assert encoder.vocab_size > 0
        assert encoder.vocab_size == encoder.vocabulary.vocab_size

    def test_special_tokens(self):
        """Test special_tokens property."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        special = encoder.special_tokens
        assert 'pad' in special
        assert 'bos' in special
        assert 'eos' in special

    def test_pad_bos_eos_tokens(self):
        """Test pad, bos, eos token properties."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        assert encoder.pad_token_id == encoder.vocabulary.PAD_TOKEN
        assert encoder.bos_token_id == encoder.vocabulary.BOS_TOKEN
        assert encoder.eos_token_id == encoder.vocabulary.EOS_TOKEN

    def test_track_to_events(self):
        """Test converting track to events."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        track = MockTrack(notes=[
            MockNote(time=0, pitch=60, duration=24, velocity=64),
            MockNote(time=24, pitch=62, duration=24, velocity=64),
        ])

        events = encoder.track_to_events(track)

        # Should have note_on, note_off, time_shift events
        event_types = [e[0] for e in events]
        assert 'note_on' in event_types
        assert 'note_off' in event_types
        assert 'time_shift' in event_types

    def test_track_to_events_empty(self):
        """Test track_to_events with empty track."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        track = MockTrack(notes=[])
        events = encoder.track_to_events(track)
        assert events == []

    def test_events_to_tokens(self):
        """Test converting events to tokens."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        events = [
            ('note_on', 60),
            ('time_shift', 24),
            ('note_off', 60),
        ]

        tokens = encoder.events_to_tokens(events, genre_id=0, instrument_id=0)

        # Should start with BOS, genre, instrument
        assert tokens[0] == encoder.bos_token_id
        # Should end with EOS
        assert tokens[-1] == encoder.eos_token_id

    def test_encode_track(self):
        """Test full encode_track method."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        track = MockTrack(notes=[
            MockNote(time=0, pitch=60, duration=24, velocity=64),
        ])

        result = encoder.encode_track(
            track=track,
            genre_id=0,
            instrument_id=0,
            max_length=100,
        )

        assert result.token_ids is not None
        assert result.attention_mask is not None
        assert len(result.token_ids) == 100
        assert len(result.attention_mask) == 100

    def test_decode_tokens(self):
        """Test decoding tokens back to events."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        # Create some tokens
        tokens = np.array([
            encoder.bos_token_id,
            encoder.vocabulary.encode_genre(0),
            encoder.vocabulary.encode_instrument(0),
            encoder.vocabulary.encode_note_on(60),
            encoder.vocabulary.encode_time_shift(24),
            encoder.vocabulary.encode_note_off(60),
            encoder.eos_token_id,
        ])

        events = encoder.decode_tokens(tokens, skip_special=True)

        event_types = [e[0] for e in events]
        assert 'note_on' in event_types
        assert 'note_off' in event_types
        assert 'time_shift' in event_types

    def test_create_conditioning_tokens(self):
        """Test creating conditioning tokens."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        tokens = encoder.create_conditioning_tokens(genre_id=2, instrument_id=10)

        assert len(tokens) == 3
        assert tokens[0] == encoder.bos_token_id

    def test_pad_sequence(self):
        """Test padding sequences."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        tokens = [1, 2, 3, 4, 5]
        padded, mask = encoder.pad_sequence(tokens, max_length=10)

        assert len(padded) == 10
        assert len(mask) == 10
        assert sum(mask) == 5  # 5 real tokens

    def test_pad_sequence_truncate(self):
        """Test truncating sequences."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5)

        tokens = list(range(100))
        padded, mask = encoder.pad_sequence(tokens, max_length=10)

        assert len(padded) == 10
        assert padded[-1] == encoder.eos_token_id

    def test_get_state(self):
        """Test get_state for serialization."""
        from src_v4.data_preprocessing.encoders import EventEncoder
        encoder = EventEncoder(num_genres=5, num_instruments=129, encode_velocity=True)

        state = encoder.get_state()

        assert state['encoder_type'] == 'event'
        assert state['num_genres'] == 5
        assert state['num_instruments'] == 129
        assert state['encode_velocity'] == True

    def test_from_state(self):
        """Test from_state reconstruction."""
        from src_v4.data_preprocessing.encoders import EventEncoder

        state = {
            'encoder_type': 'event',
            'num_genres': 10,
            'num_instruments': 129,
            'encode_velocity': False,
        }

        encoder = EventEncoder.from_state(state)

        assert encoder.vocabulary.num_genres == 10
        assert encoder.vocabulary.num_instruments == 129
        assert encoder.encode_velocity == False


# =============================================================================
# Tests for REMIEncoder
# =============================================================================

class TestREMIEncoder:
    """Tests for REMIEncoder class."""

    def test_import(self):
        """Test that REMIEncoder can be imported."""
        from src_v4.data_preprocessing.encoders import REMIEncoder
        encoder = REMIEncoder(num_genres=5)
        assert encoder is not None

    def test_vocab_size(self):
        """Test vocab_size property."""
        from src_v4.data_preprocessing.encoders import REMIEncoder
        encoder = REMIEncoder(num_genres=10, num_instruments=129)
        assert encoder.vocab_size > 0

    def test_special_tokens(self):
        """Test special_tokens property."""
        from src_v4.data_preprocessing.encoders import REMIEncoder
        encoder = REMIEncoder(num_genres=5)

        special = encoder.special_tokens
        assert 'pad' in special
        assert 'bos' in special
        assert 'eos' in special
        assert 'bar' in special

    def test_track_to_remi_events(self):
        """Test converting track to REMI events."""
        from src_v4.data_preprocessing.encoders import REMIEncoder
        encoder = REMIEncoder(num_genres=5, resolution=24)

        track = MockTrack(notes=[
            MockNote(time=0, pitch=60, duration=24, velocity=64),
            MockNote(time=96, pitch=62, duration=24, velocity=64),  # Next bar
        ])

        events = encoder.track_to_remi_events(track)

        event_types = [e[0] for e in events]
        assert 'bar' in event_types
        assert 'position' in event_types
        assert 'pitch' in event_types
        assert 'duration' in event_types
        assert 'velocity' in event_types

    def test_encode_track(self):
        """Test full encode_track method."""
        from src_v4.data_preprocessing.encoders import REMIEncoder
        encoder = REMIEncoder(num_genres=5)

        track = MockTrack(notes=[
            MockNote(time=0, pitch=60, duration=24, velocity=64),
        ])

        result = encoder.encode_track(
            track=track,
            genre_id=0,
            instrument_id=0,
            max_length=100,
        )

        assert result.token_ids is not None
        assert len(result.token_ids) == 100

    def test_decode_tokens(self):
        """Test decoding tokens."""
        from src_v4.data_preprocessing.encoders import REMIEncoder
        encoder = REMIEncoder(num_genres=5)

        tokens = np.array([
            encoder.bos_token_id,
            encoder.vocabulary.encode_genre(0),
            encoder.vocabulary.encode_instrument(0),
            encoder.vocabulary.encode_bar(),
            encoder.vocabulary.encode_position(0),
            encoder.vocabulary.encode_velocity(64),
            encoder.vocabulary.encode_pitch(60),
            encoder.vocabulary.encode_duration(8),
            encoder.eos_token_id,
        ])

        events = encoder.decode_tokens(tokens, skip_special=True)

        event_types = [e[0] for e in events]
        assert 'bar' in event_types
        assert 'pitch' in event_types
        assert 'duration' in event_types

    def test_get_state(self):
        """Test get_state for serialization."""
        from src_v4.data_preprocessing.encoders import REMIEncoder
        encoder = REMIEncoder(num_genres=5, resolution=48, positions_per_bar=16)

        state = encoder.get_state()

        assert state['encoder_type'] == 'remi'
        assert state['num_genres'] == 5
        assert state['resolution'] == 48
        assert state['positions_per_bar'] == 16

    def test_from_state(self):
        """Test from_state reconstruction."""
        from src_v4.data_preprocessing.encoders import REMIEncoder

        state = {
            'encoder_type': 'remi',
            'num_genres': 10,
            'num_instruments': 129,
            'resolution': 48,
            'positions_per_bar': 16,
            'time_signature': [4, 4],
        }

        encoder = REMIEncoder.from_state(state)

        assert encoder.vocabulary.num_genres == 10
        assert encoder.resolution == 48
        assert encoder.positions_per_bar == 16


# =============================================================================
# Tests for Vocabulary
# =============================================================================

class TestVocabulary:
    """Tests for Vocabulary class."""

    def test_import(self):
        """Test that Vocabulary can be imported."""
        from src_v4.data_preprocessing import Vocabulary
        vocab = Vocabulary()
        assert vocab is not None

    def test_add_genre(self):
        """Test adding genres."""
        from src_v4.data_preprocessing import Vocabulary
        vocab = Vocabulary()

        vocab.add_genre("rock")
        vocab.add_genre("jazz")

        assert vocab.num_genres == 2
        assert vocab.get_genre_id("rock") == 0
        assert vocab.get_genre_id("jazz") == 1

    def test_get_genre_id_unknown(self):
        """Test getting unknown genre."""
        from src_v4.data_preprocessing import Vocabulary
        vocab = Vocabulary()

        assert vocab.get_genre_id("unknown") == -1

    def test_add_instrument(self):
        """Test adding instruments."""
        from src_v4.data_preprocessing import Vocabulary
        vocab = Vocabulary()

        vocab.add_instrument(0, "song1")  # Piano
        vocab.add_instrument(0, "song2")
        vocab.add_instrument(25, "song1")  # Guitar

        assert 0 in vocab.instrument_to_songs
        assert len(vocab.instrument_to_songs[0]) == 2

    def test_get_instrument_id(self):
        """Test getting instrument ID."""
        from src_v4.data_preprocessing import Vocabulary
        vocab = Vocabulary()

        piano_id = vocab.get_instrument_id("Acoustic Grand Piano")
        assert piano_id == 0

        drums_id = vocab.get_instrument_id("Drums")
        assert drums_id == 128


# =============================================================================
# Tests for MusicDatasetConfig
# =============================================================================

class TestMusicDatasetConfig:
    """Tests for MusicDatasetConfig class."""

    def test_import(self):
        """Test that MusicDatasetConfig can be imported."""
        from src_v4.data_preprocessing import MusicDatasetConfig
        config = MusicDatasetConfig()
        assert config is not None

    def test_default_values(self):
        """Test default configuration values."""
        from src_v4.data_preprocessing import MusicDatasetConfig
        config = MusicDatasetConfig()

        assert config.resolution == 24
        assert config.max_seq_length == 1024
        assert config.train_split == 0.8

    def test_save_load(self, tmp_path):
        """Test saving and loading config."""
        from src_v4.data_preprocessing import MusicDatasetConfig

        config = MusicDatasetConfig(
            resolution=48,
            max_seq_length=2048,
        )

        filepath = tmp_path / "config.json"
        config.save(str(filepath))

        loaded = MusicDatasetConfig.load(str(filepath))

        assert loaded.resolution == 48
        assert loaded.max_seq_length == 2048


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
