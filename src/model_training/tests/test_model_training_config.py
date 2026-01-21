"""Tests for ModelTrainingConfig."""

import json
import tempfile
from pathlib import Path

import pytest

from ..configs import ModelTrainingConfig


class TestModelTrainingConfigInit:
    """Tests for ModelTrainingConfig initialization and validation."""

    def test_basic_initialization(self):
        """Test basic config creation with required parameters."""
        config = ModelTrainingConfig(model_name="test_model")
        assert config.model_name == "test_model"
        assert config.num_pitches == 128
        assert config.max_time_steps == 1000
        assert config.optimizer == 'adam'

    def test_custom_parameters(self):
        """Test config with custom parameters."""
        config = ModelTrainingConfig(
            model_name="custom_model",
            lstm_units=[256, 128, 64],
            dense_units=[128, 64],
            dropout_rate=0.3,
            batch_size=64,
            epochs=50,
            learning_rate=0.0001,
        )
        assert config.lstm_units == [256, 128, 64]
        assert config.dense_units == [128, 64]
        assert config.dropout_rate == 0.3
        assert config.batch_size == 64
        assert config.epochs == 50
        assert config.learning_rate == 0.0001


class TestModelTrainingConfigValidation:
    """Tests for ModelTrainingConfig validation."""

    def test_invalid_optimizer(self):
        """Test that invalid optimizer raises ValueError."""
        with pytest.raises(ValueError, match="Invalid optimizer"):
            ModelTrainingConfig(model_name="test", optimizer="invalid_optimizer")

    def test_valid_optimizers(self):
        """Test all valid optimizers."""
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl']
        for opt in valid_optimizers:
            config = ModelTrainingConfig(model_name="test", optimizer=opt)
            assert config.optimizer == opt

    def test_invalid_loss_function(self):
        """Test that invalid loss function raises ValueError."""
        with pytest.raises(ValueError, match="Invalid loss_function"):
            ModelTrainingConfig(model_name="test", loss_function="invalid_loss")

    def test_valid_loss_functions(self):
        """Test all valid loss functions."""
        valid_losses = ['mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy',
                        'sparse_categorical_crossentropy', 'huber', 'log_cosh']
        for loss in valid_losses:
            config = ModelTrainingConfig(model_name="test", loss_function=loss)
            assert config.loss_function == loss

    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metric"):
            ModelTrainingConfig(model_name="test", metrics=['invalid_metric'])

    def test_valid_metrics(self):
        """Test valid metrics."""
        valid_metrics = ['accuracy', 'precision', 'recall', 'auc', 'mae', 'mse', 'rmse']
        for metric in valid_metrics:
            config = ModelTrainingConfig(model_name="test", metrics=[metric])
            assert metric in config.metrics

    def test_multiple_metrics(self):
        """Test config with multiple metrics."""
        config = ModelTrainingConfig(model_name="test", metrics=['mae', 'mse', 'accuracy'])
        assert config.metrics == ['mae', 'mse', 'accuracy']

    def test_invalid_lr_schedule(self):
        """Test that invalid learning rate schedule raises ValueError."""
        with pytest.raises(ValueError, match="Invalid lr_schedule"):
            ModelTrainingConfig(model_name="test", lr_schedule="invalid_schedule")

    def test_valid_lr_schedules(self):
        """Test all valid learning rate schedules."""
        valid_schedules = ['constant', 'exponential_decay', 'step_decay',
                          'cosine_decay', 'polynomial_decay', 'reduce_on_plateau']
        for schedule in valid_schedules:
            config = ModelTrainingConfig(model_name="test", lr_schedule=schedule)
            assert config.lr_schedule == schedule

    def test_dropout_rate_too_high(self):
        """Test that dropout rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            ModelTrainingConfig(model_name="test", dropout_rate=1.5)

    def test_dropout_rate_negative(self):
        """Test that negative dropout rate raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            ModelTrainingConfig(model_name="test", dropout_rate=-0.1)

    def test_recurrent_dropout_too_high(self):
        """Test that recurrent dropout > 1 raises ValueError."""
        with pytest.raises(ValueError, match="recurrent_dropout must be between 0 and 1"):
            ModelTrainingConfig(model_name="test", recurrent_dropout=1.5)

    def test_recurrent_dropout_negative(self):
        """Test that negative recurrent dropout raises ValueError."""
        with pytest.raises(ValueError, match="recurrent_dropout must be between 0 and 1"):
            ModelTrainingConfig(model_name="test", recurrent_dropout=-0.1)

    def test_dropout_rate_bounds(self):
        """Test dropout rate at valid bounds."""
        config_0 = ModelTrainingConfig(model_name="test", dropout_rate=0)
        config_1 = ModelTrainingConfig(model_name="test", dropout_rate=1)
        assert config_0.dropout_rate == 0
        assert config_1.dropout_rate == 1

    def test_batch_size_zero(self):
        """Test that zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            ModelTrainingConfig(model_name="test", batch_size=0)

    def test_batch_size_negative(self):
        """Test that negative batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            ModelTrainingConfig(model_name="test", batch_size=-1)

    def test_epochs_zero(self):
        """Test that zero epochs raises ValueError."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            ModelTrainingConfig(model_name="test", epochs=0)

    def test_epochs_negative(self):
        """Test that negative epochs raises ValueError."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            ModelTrainingConfig(model_name="test", epochs=-5)

    def test_learning_rate_zero(self):
        """Test that zero learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ModelTrainingConfig(model_name="test", learning_rate=0)

    def test_learning_rate_negative(self):
        """Test that negative learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ModelTrainingConfig(model_name="test", learning_rate=-0.001)

    def test_empty_lstm_units(self):
        """Test that empty lstm_units raises ValueError."""
        with pytest.raises(ValueError, match="lstm_units must have at least one layer"):
            ModelTrainingConfig(model_name="test", lstm_units=[])


class TestModelTrainingConfigMethods:
    """Tests for ModelTrainingConfig methods."""

    def test_get_input_shape(self):
        """Test get_input_shape returns correct tuple."""
        config = ModelTrainingConfig(
            model_name="test",
            num_pitches=128,
            max_time_steps=500
        )
        assert config.get_input_shape() == (128, 500)

    def test_get_optimizer_kwargs_adam(self):
        """Test optimizer kwargs for Adam."""
        config = ModelTrainingConfig(
            model_name="test",
            optimizer='adam',
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        kwargs = config.get_optimizer_kwargs()
        assert kwargs['learning_rate'] == 0.001
        assert kwargs['beta_1'] == 0.9
        assert kwargs['beta_2'] == 0.999
        assert kwargs['epsilon'] == 1e-7

    def test_get_optimizer_kwargs_sgd(self):
        """Test optimizer kwargs for SGD."""
        config = ModelTrainingConfig(
            model_name="test",
            optimizer='sgd',
            learning_rate=0.01,
            momentum=0.9,
            nesterov=True
        )
        kwargs = config.get_optimizer_kwargs()
        assert kwargs['learning_rate'] == 0.01
        assert kwargs['momentum'] == 0.9
        assert kwargs['nesterov'] is True

    def test_get_optimizer_kwargs_rmsprop(self):
        """Test optimizer kwargs for RMSprop."""
        config = ModelTrainingConfig(
            model_name="test",
            optimizer='rmsprop',
            learning_rate=0.001,
            rho=0.9,
            epsilon=1e-7
        )
        kwargs = config.get_optimizer_kwargs()
        assert kwargs['learning_rate'] == 0.001
        assert kwargs['rho'] == 0.9
        assert kwargs['epsilon'] == 1e-7

    def test_get_regularization_kwargs_no_regularization(self):
        """Test regularization kwargs with no regularization."""
        config = ModelTrainingConfig(model_name="test", l1_reg=0.0, l2_reg=0.0)
        kwargs = config.get_regularization_kwargs()
        assert 'kernel_regularizer' not in kwargs

    def test_get_regularization_kwargs_l1_only(self):
        """Test regularization kwargs with L1 only."""
        config = ModelTrainingConfig(model_name="test", l1_reg=0.01, l2_reg=0.0)
        kwargs = config.get_regularization_kwargs()
        assert 'kernel_regularizer' in kwargs

    def test_get_regularization_kwargs_l2_only(self):
        """Test regularization kwargs with L2 only."""
        config = ModelTrainingConfig(model_name="test", l1_reg=0.0, l2_reg=0.01)
        kwargs = config.get_regularization_kwargs()
        assert 'kernel_regularizer' in kwargs

    def test_get_regularization_kwargs_l1_l2(self):
        """Test regularization kwargs with both L1 and L2."""
        config = ModelTrainingConfig(model_name="test", l1_reg=0.01, l2_reg=0.01)
        kwargs = config.get_regularization_kwargs()
        assert 'kernel_regularizer' in kwargs

    def test_get_regularization_kwargs_kernel_constraint(self):
        """Test regularization kwargs with kernel constraint."""
        config = ModelTrainingConfig(model_name="test", kernel_constraint_max_value=3.0)
        kwargs = config.get_regularization_kwargs()
        assert 'kernel_constraint' in kwargs

    def test_get_regularization_kwargs_recurrent_constraint(self):
        """Test regularization kwargs with recurrent constraint."""
        config = ModelTrainingConfig(model_name="test", recurrent_constraint_max_value=3.0)
        kwargs = config.get_regularization_kwargs()
        assert 'recurrent_constraint' in kwargs

    def test_summary(self):
        """Test summary method returns formatted string."""
        config = ModelTrainingConfig(model_name="test_summary_model")
        summary = config.summary()
        assert "test_summary_model" in summary
        assert "Model Training Configuration Summary" in summary
        assert "LSTM Architecture" in summary
        assert "Training Hyperparameters" in summary


class TestModelTrainingConfigSaveLoad:
    """Tests for ModelTrainingConfig save/load functionality."""

    def test_save_and_load_roundtrip(self):
        """Test that save and load preserves all config values."""
        config = ModelTrainingConfig(
            model_name="roundtrip_test",
            lstm_units=[128, 64, 32],
            dense_units=[64, 32],
            dropout_rate=0.25,
            batch_size=64,
            epochs=100,
            optimizer='sgd',
            learning_rate=0.01,
            momentum=0.9,
            l1_reg=0.001,
            l2_reg=0.001,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_config.json"
            config.save(str(filepath))

            # Verify file was created
            assert filepath.exists()

            # Load and compare
            loaded_config = ModelTrainingConfig.load(str(filepath))

            assert loaded_config.model_name == config.model_name
            assert loaded_config.lstm_units == config.lstm_units
            assert loaded_config.dense_units == config.dense_units
            assert loaded_config.dropout_rate == config.dropout_rate
            assert loaded_config.batch_size == config.batch_size
            assert loaded_config.epochs == config.epochs
            assert loaded_config.optimizer == config.optimizer
            assert loaded_config.learning_rate == config.learning_rate
            assert loaded_config.momentum == config.momentum
            assert loaded_config.l1_reg == config.l1_reg
            assert loaded_config.l2_reg == config.l2_reg

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories if needed."""
        config = ModelTrainingConfig(model_name="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "path" / "config.json"
            config.save(str(filepath))
            assert filepath.exists()

    def test_load_preserves_types(self):
        """Test that loaded config has correct types."""
        config = ModelTrainingConfig(
            model_name="type_test",
            lstm_units=[128, 64],
            bidirectional=True,
            random_seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_config.json"
            config.save(str(filepath))
            loaded = ModelTrainingConfig.load(str(filepath))

            assert isinstance(loaded.lstm_units, list)
            assert isinstance(loaded.bidirectional, bool)
            assert isinstance(loaded.random_seed, int)

    def test_saved_json_is_valid(self):
        """Test that saved file is valid JSON."""
        config = ModelTrainingConfig(model_name="json_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_config.json"
            config.save(str(filepath))

            with open(filepath, 'r') as f:
                data = json.load(f)

            assert data['model_name'] == "json_test"
            assert 'lstm_units' in data
            assert 'optimizer' in data


class TestModelTrainingConfigClassVariables:
    """Tests for class-level constants."""

    def test_valid_optimizers_set(self):
        """Test VALID_OPTIMIZERS is a non-empty set."""
        assert len(ModelTrainingConfig.VALID_OPTIMIZERS) > 0
        assert 'adam' in ModelTrainingConfig.VALID_OPTIMIZERS

    def test_valid_loss_functions_set(self):
        """Test VALID_LOSS_FUNCTIONS is a non-empty set."""
        assert len(ModelTrainingConfig.VALID_LOSS_FUNCTIONS) > 0
        assert 'mse' in ModelTrainingConfig.VALID_LOSS_FUNCTIONS

    def test_valid_metrics_set(self):
        """Test VALID_METRICS is a non-empty set."""
        assert len(ModelTrainingConfig.VALID_METRICS) > 0
        assert 'mae' in ModelTrainingConfig.VALID_METRICS

    def test_valid_lr_schedules_set(self):
        """Test VALID_LR_SCHEDULES is a non-empty set."""
        assert len(ModelTrainingConfig.VALID_LR_SCHEDULES) > 0
        assert 'constant' in ModelTrainingConfig.VALID_LR_SCHEDULES
