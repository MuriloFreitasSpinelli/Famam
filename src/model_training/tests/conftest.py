"""Shared pytest fixtures for model_training tests."""

import pytest
import numpy as np
import tensorflow as tf

from ..configs import ModelTrainingConfig
from ...core.vocabulary import Vocabulary


@pytest.fixture
def basic_config():
    """Create a basic ModelTrainingConfig for testing."""
    return ModelTrainingConfig(
        model_name="test_model",
        num_pitches=128,
        max_time_steps=100,  # Smaller for faster tests
        lstm_units=[32, 16],
        dense_units=[16],
        dropout_rate=0.1,
        batch_size=4,
        epochs=2,
        optimizer='adam',
        learning_rate=0.001,
        loss_function='mse',
        metrics=['mae'],
        use_early_stopping=False,
        use_checkpointing=False,
        use_tensorboard=False,
        save_history=False,
        save_final_model=False,
    )


@pytest.fixture
def small_config():
    """Create a minimal config for fast model building tests."""
    return ModelTrainingConfig(
        model_name="small_test_model",
        num_pitches=16,
        max_time_steps=10,
        lstm_units=[8],
        dense_units=[8],
        dropout_rate=0.0,
        batch_size=2,
        epochs=1,
        use_early_stopping=False,
        use_checkpointing=False,
        use_tensorboard=False,
        save_history=False,
        save_final_model=False,
    )


@pytest.fixture
def sample_vocabulary():
    """Create a sample vocabulary for testing."""
    vocab = Vocabulary()
    vocab.add_genre("rock")
    vocab.add_genre("jazz")
    vocab.add_genre("classical")
    vocab.add_artist("artist_1")
    vocab.add_artist("artist_2")
    return vocab


@pytest.fixture
def mock_dataset(small_config):
    """Create a mock tf.data.Dataset for testing."""
    num_samples = 8
    num_pitches = small_config.num_pitches
    max_time_steps = small_config.max_time_steps

    def generator():
        for i in range(num_samples):
            yield {
                'pianoroll': np.random.rand(num_pitches, max_time_steps).astype(np.float32),
                'genre_id': np.int32(i % 3),
                'instrument_id': np.int32(i % 5),
            }

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature={
            'pianoroll': tf.TensorSpec(shape=(num_pitches, max_time_steps), dtype=tf.float32),
            'genre_id': tf.TensorSpec(shape=(), dtype=tf.int32),
            'instrument_id': tf.TensorSpec(shape=(), dtype=tf.int32),
        }
    )
    return dataset


@pytest.fixture
def mock_datasets(mock_dataset):
    """Create train/validation/test dataset splits."""
    return {
        'train': mock_dataset.take(4),
        'validation': mock_dataset.skip(4).take(2),
        'test': mock_dataset.skip(6).take(2),
    }
