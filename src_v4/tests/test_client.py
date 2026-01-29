"""
Unit tests for client module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


# =============================================================================
# Tests for PipelineConfig
# =============================================================================

class TestPipelineConfig:
    """Tests for PipelineConfig class."""

    def test_import(self):
        """Test that PipelineConfig can be imported."""
        from src_v4.client import PipelineConfig
        config = PipelineConfig()
        assert config is not None

    def test_default_values(self):
        """Test default configuration values."""
        from src_v4.client import PipelineConfig
        config = PipelineConfig()

        assert config.encoder_type == "event"
        assert config.model_type == "transformer"
        assert config.resolution == 24
        assert config.max_seq_length == 1024
        assert config.d_model == 256
        assert config.num_layers == 4
        assert config.batch_size == 32
        assert config.epochs == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        from src_v4.client import PipelineConfig
        config = PipelineConfig(
            encoder_type="remi",
            model_type="lstm",
            d_model=512,
            epochs=50,
        )

        assert config.encoder_type == "remi"
        assert config.model_type == "lstm"
        assert config.d_model == 512
        assert config.epochs == 50

    def test_save_load(self, tmp_path):
        """Test saving and loading config."""
        from src_v4.client import PipelineConfig

        config = PipelineConfig(
            model_name="test_model",
            encoder_type="remi",
            d_model=512,
            epochs=50,
        )

        filepath = tmp_path / "config.json"
        config.save(filepath)

        assert filepath.exists()

        loaded = PipelineConfig.load(filepath)

        assert loaded.model_name == "test_model"
        assert loaded.encoder_type == "remi"
        assert loaded.d_model == 512
        assert loaded.epochs == 50

    def test_to_dataset_config(self):
        """Test converting to MusicDatasetConfig."""
        from src_v4.client import PipelineConfig

        config = PipelineConfig(
            data_dir="my_data",
            resolution=48,
            max_seq_length=2048,
            encoder_type="remi",
        )

        dataset_config = config.to_dataset_config()

        assert dataset_config.midi_dir == "my_data"
        assert dataset_config.resolution == 48
        assert dataset_config.max_seq_length == 2048
        assert dataset_config.encoder_type == "remi"

    def test_to_training_config(self):
        """Test converting to TrainingConfig."""
        from src_v4.client import PipelineConfig

        config = PipelineConfig(
            model_name="my_model",
            model_type="transformer",
            d_model=512,
            num_layers=6,
            epochs=50,
        )

        training_config = config.to_training_config(vocab_size=1000)

        assert training_config.model_name == "my_model"
        assert training_config.model_type == "transformer"
        assert training_config.d_model == 512
        assert training_config.num_layers == 6
        assert training_config.epochs == 50
        assert training_config.vocab_size == 1000

    def test_to_generation_config(self):
        """Test converting to GenerationConfig."""
        from src_v4.client import PipelineConfig

        config = PipelineConfig(
            max_seq_length=512,
            temperature=0.8,
            top_k=100,
            top_p=0.95,
        )

        gen_config = config.to_generation_config()

        assert gen_config.max_length == 512
        assert gen_config.temperature == 0.8
        assert gen_config.top_k == 100
        assert gen_config.top_p == 0.95


# =============================================================================
# Tests for MusicPipeline
# =============================================================================

class TestMusicPipeline:
    """Tests for MusicPipeline class."""

    def test_import(self):
        """Test that MusicPipeline can be imported."""
        from src_v4.client import MusicPipeline
        assert MusicPipeline is not None

    def test_create_pipeline(self):
        """Test creating a MusicPipeline."""
        from src_v4.client import MusicPipeline, PipelineConfig

        config = PipelineConfig()
        pipeline = MusicPipeline(config)

        assert pipeline is not None
        assert pipeline.config == config
        assert pipeline.encoder is None
        assert pipeline.model is None

    def test_create_pipeline_default_config(self):
        """Test creating a pipeline with default config."""
        from src_v4.client import MusicPipeline

        pipeline = MusicPipeline()

        assert pipeline is not None
        assert pipeline.config is not None

    def test_create_encoder_event(self):
        """Test creating an event encoder."""
        from src_v4.client import MusicPipeline, PipelineConfig
        from src_v4.data_preprocessing.encoders import EventEncoder

        config = PipelineConfig(encoder_type="event")
        pipeline = MusicPipeline(config)

        encoder = pipeline.create_encoder(num_genres=5)

        assert encoder is not None
        assert isinstance(encoder, EventEncoder)
        assert encoder.vocabulary.num_genres == 5

    def test_create_encoder_remi(self):
        """Test creating a REMI encoder."""
        from src_v4.client import MusicPipeline, PipelineConfig
        from src_v4.data_preprocessing.encoders import REMIEncoder

        config = PipelineConfig(encoder_type="remi")
        pipeline = MusicPipeline(config)

        encoder = pipeline.create_encoder(num_genres=5)

        assert encoder is not None
        assert isinstance(encoder, REMIEncoder)

    def test_build_model_transformer(self):
        """Test building a transformer model."""
        from src_v4.client import MusicPipeline, PipelineConfig
        from src_v4.model_training import TransformerModel

        config = PipelineConfig(
            model_type="transformer",
            d_model=64,
            num_layers=2,
            num_heads=4,
        )
        pipeline = MusicPipeline(config)
        pipeline.create_encoder(num_genres=5)

        model = pipeline.build_model()

        assert model is not None
        assert isinstance(model, TransformerModel)

    def test_build_model_lstm(self):
        """Test building an LSTM model."""
        from src_v4.client import MusicPipeline, PipelineConfig
        from src_v4.model_training import LSTMModel

        config = PipelineConfig(
            model_type="lstm",
            d_model=64,
            lstm_units=128,
        )
        pipeline = MusicPipeline(config)
        pipeline.create_encoder(num_genres=5)

        model = pipeline.build_model()

        assert model is not None
        assert isinstance(model, LSTMModel)

    def test_build_model_without_encoder_raises(self):
        """Test that building model without encoder raises error."""
        from src_v4.client import MusicPipeline

        pipeline = MusicPipeline()

        with pytest.raises(ValueError, match="No encoder"):
            pipeline.build_model()

    def test_list_genres_empty(self):
        """Test listing genres when no vocabulary."""
        from src_v4.client import MusicPipeline

        pipeline = MusicPipeline()
        genres = pipeline.list_genres()

        assert genres == []

    def test_list_instruments_empty(self):
        """Test listing instruments when no vocabulary."""
        from src_v4.client import MusicPipeline

        pipeline = MusicPipeline()
        instruments = pipeline.list_instruments()

        assert instruments == []

    def test_summary(self):
        """Test pipeline summary."""
        from src_v4.client import MusicPipeline, PipelineConfig

        config = PipelineConfig(
            encoder_type="remi",
            model_type="transformer",
        )
        pipeline = MusicPipeline(config)

        summary = pipeline.summary()

        assert "Pipeline Summary" in summary
        assert "remi" in summary
        assert "transformer" in summary

    def test_save_config(self, tmp_path):
        """Test saving pipeline config."""
        from src_v4.client import MusicPipeline, PipelineConfig

        config = PipelineConfig(
            output_dir=str(tmp_path),
            model_name="test_model",
        )
        pipeline = MusicPipeline(config)

        filepath = tmp_path / "saved_config.json"
        saved_path = pipeline.save_config(filepath)

        assert Path(saved_path).exists()

    def test_from_config(self, tmp_path):
        """Test creating pipeline from config file."""
        from src_v4.client import MusicPipeline, PipelineConfig

        # Save a config
        config = PipelineConfig(
            model_name="loaded_model",
            d_model=512,
        )
        filepath = tmp_path / "config.json"
        config.save(filepath)

        # Load pipeline from config
        pipeline = MusicPipeline.from_config(filepath)

        assert pipeline.config.model_name == "loaded_model"
        assert pipeline.config.d_model == 512

    def test_ensure_generator_without_model_raises(self):
        """Test that _ensure_generator raises without model."""
        from src_v4.client import MusicPipeline

        pipeline = MusicPipeline()

        with pytest.raises(ValueError, match="No model"):
            pipeline._ensure_generator()

    def test_generate_without_model_raises(self):
        """Test that generate raises without model."""
        from src_v4.client import MusicPipeline

        pipeline = MusicPipeline()

        with pytest.raises(ValueError, match="No model"):
            pipeline.generate()


# =============================================================================
# Tests for CLI
# =============================================================================

class TestCLI:
    """Tests for CLI module."""

    def test_import_cli(self):
        """Test that CLI can be imported."""
        from src_v4.client.cli import main, create_parser
        assert main is not None
        assert create_parser is not None

    def test_create_parser(self):
        """Test creating argument parser."""
        from src_v4.client.cli import create_parser

        parser = create_parser()

        assert parser is not None

    def test_parser_config_create(self):
        """Test parsing config create command."""
        from src_v4.client.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['config', 'create', '-o', 'test.json'])

        assert args.command == 'config'
        assert args.config_action == 'create'
        assert args.output == 'test.json'

    def test_parser_dataset_build(self):
        """Test parsing dataset build command."""
        from src_v4.client.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['dataset', 'build', 'data/midi', '-o', 'dataset.h5'])

        assert args.command == 'dataset'
        assert args.dataset_action == 'build'
        assert args.midi_dir == 'data/midi'
        assert args.output == 'dataset.h5'

    def test_parser_train(self):
        """Test parsing train command."""
        from src_v4.client.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['train', 'dataset.h5', '-c', 'config.json', '--epochs', '50'])

        assert args.command == 'train'
        assert args.dataset == 'dataset.h5'
        assert args.config == 'config.json'
        assert args.epochs == 50

    def test_parser_generate(self):
        """Test parsing generate command."""
        from src_v4.client.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'generate', 'model.h5', '-o', 'output.mid',
            '--genre', '1', '--temperature', '0.8'
        ])

        assert args.command == 'generate'
        assert args.model == 'model.h5'
        assert args.output == 'output.mid'
        assert args.genre == 1
        assert args.temperature == 0.8

    def test_parser_interactive(self):
        """Test parsing interactive command."""
        from src_v4.client.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['interactive'])

        assert args.command == 'interactive'


# =============================================================================
# Tests for Terminal Interface
# =============================================================================

class TestTerminalInterface:
    """Tests for terminal interface."""

    def test_import_terminal(self):
        """Test that terminal module can be imported."""
        from src_v4.client.terminal import TerminalApp
        assert TerminalApp is not None

    def test_create_terminal_app(self):
        """Test creating terminal app."""
        from src_v4.client.terminal import TerminalApp

        app = TerminalApp()

        assert app is not None
        assert app.config is not None
        assert app.pipeline is not None
        assert app.running == True


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
