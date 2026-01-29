"""
src_v4 - Refactored music generation codebase.

Event-based encoding for autoregressive Transformer and LSTM models.

Quick start:
    from src_v4 import MusicPipeline

    # Create pipeline
    pipeline = MusicPipeline()

    # Build dataset from MIDI files
    pipeline.build_dataset("path/to/midi/files")

    # Train model
    pipeline.train(epochs=50)
    pipeline.save_model("models/my_model")

    # Generate music
    pipeline.generate_midi("output.mid", genre_id=0)
"""

from . import data_preprocessing
from . import model_training
from . import music_generation
from . import client

# Convenience imports
from .client import MusicPipeline, PipelineConfig
from .model_training import ModelBundle, load_model_bundle
from .music_generation import MusicGenerator, GenerationConfig