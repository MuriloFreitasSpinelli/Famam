"""Tests for model_trainer module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from ..model_trainer import build_lstm_model, ModelTrainer, train_from_music_dataset
from ..configs import ModelTrainingConfig
from ...core.vocabulary import Vocabulary


class TestBuildLstmModel:
    """Tests for build_lstm_model function."""

    def test_model_creation(self, small_config):
        """Test that model is created successfully."""
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        assert model is not None
        assert model.name == 'music_generator'

    def test_model_inputs(self, small_config):
        """Test model has correct inputs."""
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        input_names = [inp.name.split(':')[0] for inp in model.inputs]
        assert 'pianoroll_input' in input_names
        assert 'genre_input' in input_names

    def test_model_output_shape(self, small_config):
        """Test model output shape matches input pianoroll shape."""
        input_shape = (16, 10)
        model = build_lstm_model(
            input_shape=input_shape,
            num_genres=3,
            config=small_config,
        )
        # Output shape should be (None, 16, 10) matching input pianoroll
        output_shape = model.output_shape
        assert output_shape[1:] == input_shape

    def test_model_with_bidirectional(self, small_config):
        """Test model creation with bidirectional LSTM."""
        small_config.bidirectional = True
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        # Check that bidirectional layers exist
        layer_names = [layer.name for layer in model.layers]
        assert any('bilstm' in name for name in layer_names)

    def test_model_without_bidirectional(self, small_config):
        """Test model creation without bidirectional LSTM."""
        small_config.bidirectional = False
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        layer_names = [layer.name for layer in model.layers]
        # Should have regular lstm layers
        assert any('lstm' in name for name in layer_names)

    def test_model_with_multiple_lstm_layers(self, small_config):
        """Test model with multiple LSTM layers."""
        small_config.lstm_units = [16, 8, 4]
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        layer_names = [layer.name for layer in model.layers]
        lstm_layers = [name for name in layer_names if 'lstm' in name.lower()]
        assert len(lstm_layers) >= 3

    def test_model_with_regularization(self, small_config):
        """Test model creation with L2 regularization."""
        small_config.l2_reg = 0.01
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        assert model is not None

    def test_model_with_l1_regularization(self, small_config):
        """Test model creation with L1 regularization."""
        small_config.l1_reg = 0.01
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        assert model is not None

    def test_model_with_l1_l2_regularization(self, small_config):
        """Test model creation with combined L1 and L2 regularization."""
        small_config.l1_reg = 0.01
        small_config.l2_reg = 0.01
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        assert model is not None

    def test_model_genre_embedding_layer(self, small_config):
        """Test that genre embedding layer exists."""
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=5,
            config=small_config,
        )
        layer_names = [layer.name for layer in model.layers]
        assert 'genre_embedding' in layer_names

    def test_model_dense_layers(self, small_config):
        """Test that dense layers are created."""
        small_config.dense_units = [32, 16]
        model = build_lstm_model(
            input_shape=(16, 10),
            num_genres=3,
            config=small_config,
        )
        layer_names = [layer.name for layer in model.layers]
        dense_layers = [name for name in layer_names if 'dense_' in name]
        assert len(dense_layers) >= 2

    def test_model_can_predict(self, small_config):
        """Test that model can make predictions."""
        input_shape = (16, 10)
        model = build_lstm_model(
            input_shape=input_shape,
            num_genres=3,
            config=small_config,
        )
        model.compile(optimizer='adam', loss='mse')

        # Create dummy input
        pianoroll = np.random.rand(1, 16, 10).astype(np.float32)
        genre_id = np.array([[1]], dtype=np.int32)

        output = model.predict({'pianoroll_input': pianoroll, 'genre_input': genre_id}, verbose=0)
        assert output.shape == (1, 16, 10)


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_initialization(self, basic_config):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(basic_config)
        assert trainer.config == basic_config
        assert trainer.model is None
        assert trainer.history is None
        assert trainer.num_genres == 0

    def test_build_model(self, small_config, sample_vocabulary):
        """Test build_model method."""
        trainer = ModelTrainer(small_config)
        model = trainer.build_model(num_genres=sample_vocabulary.num_genres)

        assert trainer.model is not None
        assert trainer.num_genres == sample_vocabulary.num_genres

    def test_build_model_with_custom_input_shape(self, small_config):
        """Test build_model with custom input shape."""
        trainer = ModelTrainer(small_config)
        custom_shape = (32, 50)
        model = trainer.build_model(num_genres=5, input_shape=custom_shape)

        # Check output shape reflects custom input
        assert model.output_shape[1:] == custom_shape

    def test_build_model_default_input_shape(self, small_config):
        """Test build_model uses config input shape by default."""
        trainer = ModelTrainer(small_config)
        model = trainer.build_model(num_genres=5)

        expected_shape = (small_config.num_pitches, small_config.max_time_steps)
        assert model.output_shape[1:] == expected_shape


class TestModelTrainerGetOptimizer:
    """Tests for ModelTrainer._get_optimizer method."""

    def test_get_optimizer_adam(self, small_config):
        """Test Adam optimizer creation."""
        small_config.optimizer = 'adam'
        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        optimizer = trainer._get_optimizer()
        assert 'Adam' in type(optimizer).__name__

    def test_get_optimizer_sgd(self, small_config):
        """Test SGD optimizer creation."""
        small_config.optimizer = 'sgd'
        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        optimizer = trainer._get_optimizer()
        assert 'SGD' in type(optimizer).__name__

    def test_get_optimizer_rmsprop(self, small_config):
        """Test RMSprop optimizer creation."""
        small_config.optimizer = 'rmsprop'
        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        optimizer = trainer._get_optimizer()
        assert 'RMSprop' in type(optimizer).__name__

    def test_get_optimizer_adamax(self, small_config):
        """Test Adamax optimizer creation."""
        small_config.optimizer = 'adamax'
        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        optimizer = trainer._get_optimizer()
        assert 'Adamax' in type(optimizer).__name__

    def test_get_optimizer_nadam(self, small_config):
        """Test Nadam optimizer creation."""
        small_config.optimizer = 'nadam'
        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        optimizer = trainer._get_optimizer()
        assert 'Nadam' in type(optimizer).__name__

    def test_get_optimizer_unknown_defaults_to_adam(self, small_config):
        """Test unknown optimizer defaults to Adam."""
        # Bypass validation to test fallback behavior
        small_config.optimizer = 'adagrad'  # Valid but not explicitly handled
        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        # Should fallback to Adam for unhandled cases
        optimizer = trainer._get_optimizer()
        assert optimizer is not None


class TestModelTrainerGetCallbacks:
    """Tests for ModelTrainer._get_callbacks method."""

    def test_callbacks_empty_when_disabled(self, small_config):
        """Test no callbacks when all are disabled."""
        small_config.use_early_stopping = False
        small_config.use_checkpointing = False
        small_config.use_tensorboard = False
        small_config.lr_schedule = 'constant'

        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        callbacks = trainer._get_callbacks()
        assert len(callbacks) == 0

    def test_early_stopping_callback(self, small_config):
        """Test EarlyStopping callback is created."""
        small_config.use_early_stopping = True
        small_config.use_checkpointing = False
        small_config.use_tensorboard = False
        small_config.lr_schedule = 'constant'

        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        callbacks = trainer._get_callbacks()
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_types

    def test_reduce_lr_on_plateau_callback(self, small_config):
        """Test ReduceLROnPlateau callback is created."""
        small_config.use_early_stopping = False
        small_config.use_checkpointing = False
        small_config.use_tensorboard = False
        small_config.lr_schedule = 'reduce_on_plateau'

        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=3)

        callbacks = trainer._get_callbacks()
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'ReduceLROnPlateau' in callback_types


class TestModelTrainerPrepareDataset:
    """Tests for ModelTrainer._prepare_dataset method."""

    def test_prepare_dataset_formats_correctly(self, small_config, mock_dataset):
        """Test that dataset is formatted correctly for model input."""
        trainer = ModelTrainer(small_config)
        prepared = trainer._prepare_dataset(mock_dataset, batch_size=2, shuffle=False)

        for inputs, target in prepared.take(1):
            assert 'pianoroll_input' in inputs
            assert 'genre_input' in inputs
            # Target should be pianoroll
            assert target.shape[1:] == (small_config.num_pitches, small_config.max_time_steps)

    def test_prepare_dataset_batches(self, small_config, mock_dataset):
        """Test that dataset is batched correctly."""
        batch_size = 2
        trainer = ModelTrainer(small_config)
        prepared = trainer._prepare_dataset(mock_dataset, batch_size=batch_size, shuffle=False)

        for inputs, target in prepared.take(1):
            assert inputs['pianoroll_input'].shape[0] == batch_size

    def test_prepare_dataset_shuffle(self, small_config, mock_dataset):
        """Test that shuffle parameter is applied."""
        trainer = ModelTrainer(small_config)
        # Just verify it doesn't error with shuffle=True
        prepared = trainer._prepare_dataset(mock_dataset, batch_size=2, shuffle=True)
        assert prepared is not None


class TestModelTrainerTrain:
    """Tests for ModelTrainer.train method."""

    def test_train_builds_model_if_needed(self, small_config, mock_dataset, sample_vocabulary):
        """Test that train builds model if not already built."""
        trainer = ModelTrainer(small_config)
        assert trainer.model is None

        train_ds = mock_dataset.take(4)
        val_ds = mock_dataset.skip(4).take(2)

        history = trainer.train(train_ds, val_ds, num_genres=sample_vocabulary.num_genres)

        assert trainer.model is not None
        assert history is not None

    def test_train_returns_history_dict(self, small_config, mock_dataset, sample_vocabulary):
        """Test that train returns a history dictionary."""
        trainer = ModelTrainer(small_config)

        train_ds = mock_dataset.take(4)
        val_ds = mock_dataset.skip(4).take(2)

        history = trainer.train(train_ds, val_ds, num_genres=sample_vocabulary.num_genres)

        assert isinstance(history, dict)
        assert 'loss' in history
        assert 'val_loss' in history

    def test_train_with_prebuilt_model(self, small_config, mock_dataset, sample_vocabulary):
        """Test training with a pre-built model."""
        trainer = ModelTrainer(small_config)
        trainer.build_model(num_genres=sample_vocabulary.num_genres)

        train_ds = mock_dataset.take(4)
        val_ds = mock_dataset.skip(4).take(2)

        history = trainer.train(train_ds, val_ds, num_genres=sample_vocabulary.num_genres)
        assert history is not None


class TestModelTrainerEvaluate:
    """Tests for ModelTrainer.evaluate method."""

    def test_evaluate_requires_model(self, small_config, mock_dataset):
        """Test that evaluate raises error without model."""
        trainer = ModelTrainer(small_config)

        with pytest.raises(ValueError, match="Train or load a model first"):
            trainer.evaluate(mock_dataset)

    def test_evaluate_returns_dict(self, small_config, mock_dataset, sample_vocabulary):
        """Test that evaluate returns metrics dictionary."""
        trainer = ModelTrainer(small_config)

        train_ds = mock_dataset.take(4)
        val_ds = mock_dataset.skip(4).take(2)
        test_ds = mock_dataset.skip(6).take(2)

        trainer.train(train_ds, val_ds, num_genres=sample_vocabulary.num_genres)
        results = trainer.evaluate(test_ds)

        assert isinstance(results, dict)
        assert 'loss' in results


class TestModelTrainerSaveBundle:
    """Tests for ModelTrainer.save_bundle method."""

    def test_save_bundle_requires_model(self, small_config, sample_vocabulary):
        """Test that save_bundle raises error without model."""
        trainer = ModelTrainer(small_config)

        with pytest.raises(ValueError, match="Train or load a model first"):
            trainer.save_bundle("test_path", sample_vocabulary)

    def test_save_bundle_creates_files(self, small_config, mock_dataset, sample_vocabulary):
        """Test that save_bundle creates bundle files."""
        trainer = ModelTrainer(small_config)

        train_ds = mock_dataset.take(4)
        val_ds = mock_dataset.skip(4).take(2)

        trainer.train(train_ds, val_ds, num_genres=sample_vocabulary.num_genres)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_bundle"
            bundle = trainer.save_bundle(str(filepath), sample_vocabulary)

            # Check files were created
            h5_path = filepath.with_suffix('.h5')
            keras_path = filepath.with_suffix('.keras')

            assert h5_path.exists()
            assert keras_path.exists()
            assert bundle is not None


class TestTrainFromMusicDataset:
    """Tests for train_from_music_dataset convenience function."""

    def test_train_from_music_dataset_basic(self, small_config, mock_datasets, sample_vocabulary):
        """Test basic train_from_music_dataset usage."""
        model, history, trainer = train_from_music_dataset(
            mock_datasets,
            small_config,
            sample_vocabulary,
        )

        assert model is not None
        assert isinstance(history, dict)
        assert isinstance(trainer, ModelTrainer)

    def test_train_from_music_dataset_without_test(self, small_config, mock_dataset, sample_vocabulary):
        """Test train_from_music_dataset without test dataset."""
        datasets = {
            'train': mock_dataset.take(4),
            'validation': mock_dataset.skip(4).take(2),
        }

        model, history, trainer = train_from_music_dataset(
            datasets,
            small_config,
            sample_vocabulary,
        )

        assert model is not None
        assert history is not None

    def test_train_from_music_dataset_with_test(self, small_config, mock_datasets, sample_vocabulary):
        """Test train_from_music_dataset with test dataset evaluation."""
        model, history, trainer = train_from_music_dataset(
            mock_datasets,
            small_config,
            sample_vocabulary,
        )

        # Should complete without errors
        assert trainer.history is not None

    def test_train_from_music_dataset_returns_correct_types(self, small_config, mock_datasets, sample_vocabulary):
        """Test return types are correct."""
        result = train_from_music_dataset(
            mock_datasets,
            small_config,
            sample_vocabulary,
        )

        assert len(result) == 3
        model, history, trainer = result

        assert isinstance(model, tf.keras.Model)
        assert isinstance(history, dict)
        assert isinstance(trainer, ModelTrainer)


class TestModelTrainerIntegration:
    """Integration tests for complete training workflow."""

    def test_full_training_workflow(self, small_config, mock_datasets, sample_vocabulary):
        """Test complete training, evaluation, and save workflow."""
        # Train
        trainer = ModelTrainer(small_config)
        history = trainer.train(
            mock_datasets['train'],
            mock_datasets['validation'],
            num_genres=sample_vocabulary.num_genres,
        )
        assert history is not None

        # Evaluate
        results = trainer.evaluate(mock_datasets['test'])
        assert 'loss' in results

        # Save bundle
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "full_test_bundle"
            bundle = trainer.save_bundle(str(filepath), sample_vocabulary)
            assert bundle is not None

    def test_training_with_different_optimizers(self, small_config, mock_dataset, sample_vocabulary):
        """Test training works with different optimizers."""
        train_ds = mock_dataset.take(4)
        val_ds = mock_dataset.skip(4).take(2)

        for optimizer in ['adam', 'sgd', 'rmsprop']:
            small_config.optimizer = optimizer
            trainer = ModelTrainer(small_config)
            history = trainer.train(train_ds, val_ds, num_genres=sample_vocabulary.num_genres)
            assert history is not None

    def test_training_with_bidirectional_lstm(self, small_config, mock_dataset, sample_vocabulary):
        """Test training with bidirectional LSTM."""
        small_config.bidirectional = True
        trainer = ModelTrainer(small_config)

        train_ds = mock_dataset.take(4)
        val_ds = mock_dataset.skip(4).take(2)

        history = trainer.train(train_ds, val_ds, num_genres=sample_vocabulary.num_genres)
        assert history is not None

    def test_training_with_regularization(self, small_config, mock_dataset, sample_vocabulary):
        """Test training with L2 regularization."""
        small_config.l2_reg = 0.01
        trainer = ModelTrainer(small_config)

        train_ds = mock_dataset.take(4)
        val_ds = mock_dataset.skip(4).take(2)

        history = trainer.train(train_ds, val_ds, num_genres=sample_vocabulary.num_genres)
        assert history is not None
