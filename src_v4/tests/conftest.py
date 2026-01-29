"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List


# =============================================================================
# Mock Classes
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
# Fixtures
# =============================================================================

@pytest.fixture
def mock_note():
    """Create a mock note."""
    return MockNote(time=0, pitch=60, duration=24, velocity=64)


@pytest.fixture
def mock_track():
    """Create a mock track with notes."""
    notes = [
        MockNote(time=0, pitch=60, duration=24, velocity=64),
        MockNote(time=24, pitch=62, duration=24, velocity=64),
        MockNote(time=48, pitch=64, duration=24, velocity=64),
    ]
    return MockTrack(notes=notes)


@pytest.fixture
def mock_empty_track():
    """Create an empty mock track."""
    return MockTrack(notes=[])


@pytest.fixture
def event_encoder():
    """Create an EventEncoder for testing."""
    from src_v4.data_preprocessing.encoders import EventEncoder
    return EventEncoder(num_genres=5, num_instruments=129)


@pytest.fixture
def remi_encoder():
    """Create a REMIEncoder for testing."""
    from src_v4.data_preprocessing.encoders import REMIEncoder
    return REMIEncoder(num_genres=5, num_instruments=129)


@pytest.fixture
def small_transformer_config():
    """Create a small transformer config for testing."""
    from src_v4.model_training import TrainingConfig
    return TrainingConfig(
        model_type="transformer",
        vocab_size=500,
        max_seq_length=128,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        dropout_rate=0.1,
        batch_size=4,
        epochs=1,
    )


@pytest.fixture
def small_lstm_config():
    """Create a small LSTM config for testing."""
    from src_v4.model_training import TrainingConfig
    return TrainingConfig(
        model_type="lstm",
        vocab_size=500,
        max_seq_length=128,
        d_model=64,
        lstm_units=64,
        batch_size=4,
        epochs=1,
    )


@pytest.fixture
def small_transformer(event_encoder, small_transformer_config):
    """Create a small transformer model for testing."""
    from src_v4.model_training import TransformerModel
    return TransformerModel(
        vocab_size=event_encoder.vocab_size,
        max_seq_length=small_transformer_config.max_seq_length,
        d_model=small_transformer_config.d_model,
        num_layers=small_transformer_config.num_layers,
        num_heads=small_transformer_config.num_heads,
        d_ff=small_transformer_config.d_ff,
        dropout_rate=small_transformer_config.dropout_rate,
    )


@pytest.fixture
def small_lstm(event_encoder, small_lstm_config):
    """Create a small LSTM model for testing."""
    from src_v4.model_training import LSTMModel
    return LSTMModel(
        vocab_size=event_encoder.vocab_size,
        max_seq_length=small_lstm_config.max_seq_length,
        d_model=small_lstm_config.d_model,
        lstm_units=small_lstm_config.lstm_units,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pipeline_config():
    """Create a pipeline config for testing."""
    from src_v4.client import PipelineConfig
    return PipelineConfig(
        encoder_type="event",
        model_type="transformer",
        d_model=64,
        num_layers=2,
        num_heads=4,
        batch_size=4,
        epochs=1,
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create a pipeline for testing."""
    from src_v4.client import MusicPipeline
    return MusicPipeline(pipeline_config)


# =============================================================================
# Helper Functions
# =============================================================================

def create_dummy_tokens(encoder, length=50):
    """Create dummy token sequence for testing."""
    vocab = encoder.vocabulary
    tokens = [
        vocab.BOS_TOKEN if hasattr(vocab, 'BOS_TOKEN') else encoder.bos_token_id,
    ]

    # Add some random events
    for _ in range(length - 2):
        # Add time shift and note on/off
        if hasattr(vocab, 'encode_time_shift'):
            tokens.append(vocab.encode_time_shift(np.random.randint(1, 25)))
        if hasattr(vocab, 'encode_note_on'):
            tokens.append(vocab.encode_note_on(np.random.randint(48, 72)))

    tokens.append(vocab.EOS_TOKEN if hasattr(vocab, 'EOS_TOKEN') else encoder.eos_token_id)

    return np.array(tokens[:length], dtype=np.int32)


def create_dummy_dataset(encoder, num_samples=10, seq_length=64):
    """Create a dummy TensorFlow dataset for testing."""
    import tensorflow as tf

    def generator():
        for _ in range(num_samples):
            input_ids = np.random.randint(0, encoder.vocab_size, seq_length)
            attention_mask = np.ones(seq_length, dtype=np.int32)
            labels = np.roll(input_ids, -1)
            labels[-1] = encoder.pad_token_id

            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

    return tf.data.Dataset.from_generator(
        generator,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(seq_length,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(seq_length,), dtype=tf.int32),
            'labels': tf.TensorSpec(shape=(seq_length,), dtype=tf.int32),
        }
    )
