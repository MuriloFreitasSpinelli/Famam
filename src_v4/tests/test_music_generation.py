"""
Unit tests for music_generation module.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# =============================================================================
# Tests for GenerationConfig
# =============================================================================

class TestGenerationConfig:
    """Tests for GenerationConfig class."""

    def test_import(self):
        """Test that GenerationConfig can be imported."""
        from src_v4.music_generation import GenerationConfig
        config = GenerationConfig()
        assert config is not None

    def test_default_values(self):
        """Test default configuration values."""
        from src_v4.music_generation import GenerationConfig
        config = GenerationConfig()

        assert config.max_length == 1024
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.resolution == 24
        assert config.tempo == 120.0

    def test_custom_values(self):
        """Test custom configuration values."""
        from src_v4.music_generation import GenerationConfig
        config = GenerationConfig(
            max_length=512,
            temperature=0.8,
            top_k=100,
            top_p=0.95,
        )

        assert config.max_length == 512
        assert config.temperature == 0.8
        assert config.top_k == 100
        assert config.top_p == 0.95


# =============================================================================
# Tests for MusicGenerator
# =============================================================================

class TestMusicGenerator:
    """Tests for MusicGenerator class."""

    def test_import(self):
        """Test that MusicGenerator can be imported."""
        from src_v4.music_generation import MusicGenerator
        assert MusicGenerator is not None

    def test_create_generator(self):
        """Test creating a MusicGenerator."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        # Create mock model and encoder
        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        config = GenerationConfig(max_length=50)

        generator = MusicGenerator(model, encoder, config)

        assert generator is not None
        assert generator.model == model
        assert generator.encoder == encoder
        assert generator.config == config

    def test_generate_tokens(self):
        """Test token generation."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        config = GenerationConfig(max_length=30)

        generator = MusicGenerator(model, encoder, config)

        tokens = generator.generate_tokens(
            genre_id=0,
            instrument_id=0,
            max_length=30,
        )

        assert tokens is not None
        assert len(tokens) > 0
        assert len(tokens) <= 30

    def test_generate_events(self):
        """Test event generation."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        config = GenerationConfig(max_length=30)

        generator = MusicGenerator(model, encoder, config)

        events = generator.generate_events(
            genre_id=0,
            instrument_id=0,
        )

        assert events is not None
        assert isinstance(events, list)

    def test_generate_notes(self):
        """Test note generation."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        config = GenerationConfig(max_length=50)

        generator = MusicGenerator(model, encoder, config)

        notes = generator.generate_notes(
            genre_id=0,
            instrument_id=0,
        )

        assert notes is not None
        assert isinstance(notes, list)

    def test_generate_track(self):
        """Test track generation."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder
        import muspy

        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        config = GenerationConfig(max_length=50)

        generator = MusicGenerator(model, encoder, config)

        track = generator.generate_track(
            genre_id=0,
            instrument_id=0,
            program=0,
            is_drum=False,
        )

        assert track is not None
        assert isinstance(track, muspy.Track)
        assert track.program == 0
        assert track.is_drum == False

    def test_generate_music(self):
        """Test music object generation."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder
        import muspy

        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        config = GenerationConfig(max_length=50)

        generator = MusicGenerator(model, encoder, config)

        music = generator.generate_music(
            genre_id=0,
            instrument_ids=[0],
            include_drums=False,
        )

        assert music is not None
        assert isinstance(music, muspy.Music)
        assert len(music.tracks) >= 1

    def test_generate_midi(self, tmp_path):
        """Test MIDI file generation."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        config = GenerationConfig(max_length=50)

        generator = MusicGenerator(model, encoder, config)

        output_path = tmp_path / "test_output.mid"

        music = generator.generate_midi(
            output_path=output_path,
            genre_id=0,
            instrument_ids=[0],
            include_drums=False,
        )

        assert music is not None
        assert output_path.exists()

    def test_events_to_notes_event_encoding(self):
        """Test converting event-based events to notes."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        generator = MusicGenerator(model, encoder)

        events = [
            ('note_on', 60),
            ('time_shift', 24),
            ('note_off', 60),
            ('time_shift', 24),
            ('note_on', 62),
            ('time_shift', 24),
            ('note_off', 62),
        ]

        notes = generator._events_to_notes(events)

        assert len(notes) == 2
        assert notes[0].pitch == 60
        assert notes[0].time == 0
        assert notes[0].duration == 24
        assert notes[1].pitch == 62

    def test_events_to_notes_remi_encoding(self):
        """Test converting REMI events to notes."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import REMIEncoder

        encoder = REMIEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        generator = MusicGenerator(model, encoder)

        events = [
            ('bar', 0),
            ('position', 0),
            ('velocity', 64),
            ('pitch', 60),
            ('duration', 8),
            ('position', 8),
            ('velocity', 64),
            ('pitch', 62),
            ('duration', 8),
        ]

        notes = generator._events_to_notes(events)

        assert len(notes) == 2
        assert notes[0].pitch == 60
        assert notes[1].pitch == 62

    def test_tokens_to_pianoroll(self):
        """Test converting tokens to pianoroll."""
        from src_v4.music_generation import MusicGenerator, GenerationConfig
        from src_v4.model_training import TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )
        generator = MusicGenerator(model, encoder)

        # Create tokens for a simple note
        vocab = encoder.vocabulary
        tokens = np.array([
            vocab.BOS_TOKEN,
            vocab.encode_genre(0),
            vocab.encode_instrument(0),
            vocab.encode_note_on(60),
            vocab.encode_time_shift(24),
            vocab.encode_note_off(60),
            vocab.EOS_TOKEN,
        ])

        pianoroll = generator.tokens_to_pianoroll(tokens, num_time_steps=100)

        assert pianoroll.shape == (128, 100)
        assert pianoroll[60, 0:24].sum() > 0  # Note 60 should be active

    def test_from_bundle(self):
        """Test creating generator from bundle."""
        from src_v4.music_generation import MusicGenerator
        from src_v4.model_training import ModelBundle, TrainingConfig, TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        encoder = EventEncoder(num_genres=5)
        config = TrainingConfig(d_model=64, num_layers=2, num_heads=4)
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=config.max_seq_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
        )

        bundle = ModelBundle(model, encoder, config)

        generator = MusicGenerator.from_bundle(bundle)

        assert generator is not None
        assert generator.model == model
        assert generator.encoder == encoder


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
