"""
Unit tests for model_training module.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock, patch


# =============================================================================
# Tests for TrainingConfig
# =============================================================================

class TestTrainingConfig:
    """Tests for TrainingConfig class."""

    def test_import(self):
        """Test that TrainingConfig can be imported."""
        from src_v4.model_training import TrainingConfig
        config = TrainingConfig()
        assert config is not None

    def test_default_values(self):
        """Test default configuration values."""
        from src_v4.model_training import TrainingConfig
        config = TrainingConfig()

        assert config.model_type == "transformer"
        assert config.d_model == 256
        assert config.num_layers == 4
        assert config.num_heads == 8
        assert config.batch_size == 32

    def test_transformer_config(self):
        """Test transformer-specific config."""
        from src_v4.model_training import TrainingConfig
        config = TrainingConfig(
            model_type="transformer",
            d_model=512,
            num_layers=6,
            num_heads=8,
            d_ff=2048,
        )

        assert config.model_type == "transformer"
        assert config.d_model == 512
        assert config.num_layers == 6

    def test_lstm_config(self):
        """Test LSTM-specific config."""
        from src_v4.model_training import TrainingConfig
        config = TrainingConfig(
            model_type="lstm",
            lstm_units=512,
            d_model=256,
        )

        assert config.model_type == "lstm"
        assert config.lstm_units == 512

    def test_preset_configs(self):
        """Test preset configuration functions."""
        from src_v4.model_training import (
            get_transformer_small,
            get_transformer_medium,
            get_transformer_large,
            get_lstm_small,
            get_lstm_medium,
        )

        small = get_transformer_small()
        assert small.model_type == "transformer"
        assert small.d_model < get_transformer_medium().d_model

        lstm = get_lstm_small()
        assert lstm.model_type == "lstm"

    def test_save_load(self, tmp_path):
        """Test saving and loading config."""
        from src_v4.model_training import TrainingConfig

        config = TrainingConfig(
            model_name="test_model",
            d_model=512,
            epochs=50,
        )

        filepath = tmp_path / "config.json"
        config.save(str(filepath))

        loaded = TrainingConfig.load(str(filepath))

        assert loaded.model_name == "test_model"
        assert loaded.d_model == 512
        assert loaded.epochs == 50


# =============================================================================
# Tests for BaseMusicModel
# =============================================================================

class TestBaseMusicModel:
    """Tests for BaseMusicModel abstract class."""

    def test_import(self):
        """Test that BaseMusicModel can be imported."""
        from src_v4.model_training import BaseMusicModel
        assert BaseMusicModel is not None

    def test_cannot_instantiate(self):
        """Test that BaseMusicModel cannot be instantiated directly."""
        from src_v4.model_training import BaseMusicModel

        with pytest.raises(TypeError):
            BaseMusicModel(vocab_size=100, max_seq_length=512, d_model=256)


# =============================================================================
# Tests for TransformerModel
# =============================================================================

class TestTransformerModel:
    """Tests for TransformerModel class."""

    def test_import(self):
        """Test that TransformerModel can be imported."""
        from src_v4.model_training import TransformerModel
        assert TransformerModel is not None

    def test_create_model(self):
        """Test creating a TransformerModel."""
        from src_v4.model_training import TransformerModel

        model = TransformerModel(
            vocab_size=500,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
            dropout_rate=0.1,
        )

        assert model is not None
        assert model.vocab_size == 500
        assert model.d_model == 64

    def test_build_from_config(self):
        """Test building model from config."""
        from src_v4.model_training import TrainingConfig, build_transformer_from_config

        config = TrainingConfig(
            model_type="transformer",
            vocab_size=500,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )

        model = build_transformer_from_config(config, vocab_size=500)
        assert model is not None

    def test_call(self):
        """Test forward pass."""
        from src_v4.model_training import TransformerModel

        model = TransformerModel(
            vocab_size=500,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )

        # Create dummy input
        batch_size = 2
        seq_length = 32
        input_ids = np.random.randint(0, 500, (batch_size, seq_length))
        attention_mask = np.ones((batch_size, seq_length))

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        output = model(inputs, training=False)

        assert output.shape == (batch_size, seq_length, 500)

    def test_generate(self):
        """Test generation method."""
        from src_v4.model_training import TransformerModel

        model = TransformerModel(
            vocab_size=500,
            max_seq_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
        )

        start_tokens = np.array([1, 2, 3])  # BOS, genre, instrument

        generated = model.generate(
            start_tokens=start_tokens,
            max_length=20,
            temperature=1.0,
            top_k=10,
            top_p=0.9,
            eos_token_id=4,
        )

        assert len(generated) >= len(start_tokens)
        assert len(generated) <= 20


# =============================================================================
# Tests for LSTMModel
# =============================================================================

class TestLSTMModel:
    """Tests for LSTMModel class."""

    def test_import(self):
        """Test that LSTMModel can be imported."""
        from src_v4.model_training import LSTMModel
        assert LSTMModel is not None

    def test_create_model(self):
        """Test creating an LSTMModel."""
        from src_v4.model_training import LSTMModel

        model = LSTMModel(
            vocab_size=500,
            max_seq_length=128,
            d_model=64,
            lstm_units=128,
            num_lstm_layers=2,
        )

        assert model is not None
        assert model.vocab_size == 500

    def test_build_from_config(self):
        """Test building model from config."""
        from src_v4.model_training import TrainingConfig, build_lstm_from_config

        config = TrainingConfig(
            model_type="lstm",
            vocab_size=500,
            max_seq_length=128,
            d_model=64,
            lstm_units=128,
        )

        model = build_lstm_from_config(config, vocab_size=500)
        assert model is not None

    def test_call(self):
        """Test forward pass."""
        from src_v4.model_training import LSTMModel

        model = LSTMModel(
            vocab_size=500,
            max_seq_length=128,
            d_model=64,
            lstm_units=128,
        )

        batch_size = 2
        seq_length = 32
        input_ids = np.random.randint(0, 500, (batch_size, seq_length))
        attention_mask = np.ones((batch_size, seq_length))

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        output = model(inputs, training=False)

        assert output.shape == (batch_size, seq_length, 500)

    def test_generate(self):
        """Test generation method."""
        from src_v4.model_training import LSTMModel

        model = LSTMModel(
            vocab_size=500,
            max_seq_length=128,
            d_model=64,
            lstm_units=128,
        )

        start_tokens = np.array([1, 2, 3])

        generated = model.generate(
            start_tokens=start_tokens,
            max_length=20,
            temperature=1.0,
            eos_token_id=4,
        )

        assert len(generated) >= len(start_tokens)


# =============================================================================
# Tests for Trainer
# =============================================================================

class TestTrainer:
    """Tests for Trainer class."""

    def test_import(self):
        """Test that Trainer can be imported."""
        from src_v4.model_training import Trainer
        assert Trainer is not None

    def test_create_trainer(self):
        """Test creating a Trainer."""
        from src_v4.model_training import Trainer, TrainingConfig
        from src_v4.data_preprocessing.encoders import EventEncoder

        config = TrainingConfig()
        encoder = EventEncoder(num_genres=5)

        trainer = Trainer(config, encoder)

        assert trainer is not None
        assert trainer.config == config
        assert trainer.encoder == encoder

    def test_build_model(self):
        """Test building model through trainer."""
        from src_v4.model_training import Trainer, TrainingConfig
        from src_v4.data_preprocessing.encoders import EventEncoder

        config = TrainingConfig(
            model_type="transformer",
            d_model=64,
            num_layers=2,
            num_heads=4,
        )
        encoder = EventEncoder(num_genres=5)

        trainer = Trainer(config, encoder)
        model = trainer.build_model()

        assert model is not None


# =============================================================================
# Tests for Loss and Metrics
# =============================================================================

class TestLossAndMetrics:
    """Tests for custom loss and metrics."""

    def test_masked_loss_import(self):
        """Test importing MaskedSparseCategoricalCrossentropy."""
        from src_v4.model_training import MaskedSparseCategoricalCrossentropy
        assert MaskedSparseCategoricalCrossentropy is not None

    def test_masked_loss(self):
        """Test masked loss computation."""
        from src_v4.model_training import MaskedSparseCategoricalCrossentropy

        loss_fn = MaskedSparseCategoricalCrossentropy(pad_token_id=0)

        # Create dummy data
        y_true = tf.constant([[1, 2, 3, 0, 0]], dtype=tf.int32)
        y_pred = tf.random.uniform((1, 5, 10))  # batch, seq, vocab

        loss = loss_fn(y_true, y_pred)

        assert loss is not None
        assert loss.numpy() >= 0

    def test_masked_accuracy_import(self):
        """Test importing MaskedAccuracy."""
        from src_v4.model_training import MaskedAccuracy
        assert MaskedAccuracy is not None

    def test_masked_accuracy(self):
        """Test masked accuracy computation."""
        from src_v4.model_training import MaskedAccuracy

        metric = MaskedAccuracy(pad_token_id=0)

        # Create dummy data where predictions are correct
        y_true = tf.constant([[1, 2, 3, 0, 0]], dtype=tf.int32)
        # One-hot style predictions
        y_pred = tf.one_hot(y_true, depth=10)

        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should be 1.0 since predictions match (excluding padding)
        assert result.numpy() == 1.0

    def test_transformer_lr_schedule(self):
        """Test TransformerLRSchedule."""
        from src_v4.model_training import TransformerLRSchedule

        schedule = TransformerLRSchedule(d_model=256, warmup_steps=4000)

        # Test at different steps
        lr_1 = schedule(1)
        lr_1000 = schedule(1000)
        lr_4000 = schedule(4000)
        lr_10000 = schedule(10000)

        # LR should increase during warmup, then decrease
        assert lr_1.numpy() < lr_4000.numpy()
        assert lr_10000.numpy() < lr_4000.numpy()


# =============================================================================
# Tests for ModelBundle
# =============================================================================

class TestModelBundle:
    """Tests for ModelBundle class."""

    def test_import(self):
        """Test that ModelBundle can be imported."""
        from src_v4.model_training import ModelBundle
        assert ModelBundle is not None

    def test_create_bundle(self):
        """Test creating a ModelBundle."""
        from src_v4.model_training import ModelBundle, TrainingConfig, TransformerModel
        from src_v4.data_preprocessing.encoders import EventEncoder

        # Create components
        encoder = EventEncoder(num_genres=5)
        config = TrainingConfig(
            model_type="transformer",
            d_model=64,
            num_layers=2,
            num_heads=4,
        )
        model = TransformerModel(
            vocab_size=encoder.vocab_size,
            max_seq_length=config.max_seq_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
        )

        bundle = ModelBundle(
            model=model,
            encoder=encoder,
            config=config,
            model_name="test_model",
        )

        assert bundle is not None
        assert bundle.model_name == "test_model"
        assert bundle.vocab_size == encoder.vocab_size

    def test_bundle_summary(self):
        """Test bundle summary method."""
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
        summary = bundle.summary()

        assert "Model Bundle Summary" in summary
        assert "transformer" in summary.lower()


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
