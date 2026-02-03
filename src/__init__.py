"""
FAMAM - Music Generation with Transformers and LSTMs.

Multitrack music generation using event-based encoding for
autoregressive Transformer and LSTM models.

Quick start:
    from src import MusicPipeline

    # Create pipeline
    pipeline = MusicPipeline()

    # Build dataset from MIDI files
    pipeline.build_dataset("path/to/midi/files")

    # Train model
    pipeline.train(epochs=50)
    pipeline.save_model("models/my_model")

    # Generate music
    pipeline.generate_midi("output.mid", genre_id=0)

Authors: Murilo de Freitas Spinelli, Radu Cristea, Ryan, Timofey
Repository: https://github.com/MuriloFreitasSpinelli/Famam
"""

__version__ = "3.3.0"
__authors__ = ["Murilo de Freitas Spinelli", "Radu Cristea", "Ryan", "Timofey"]
__email__ = "murilodefreitasspinelli@gmail.com"

from . import data
from . import models
from . import generation
from . import cli
from . import training

# Convenience imports
from .cli import MusicPipeline, PipelineConfig
from .models import ModelBundle, load_model_bundle
from .generation import MusicGenerator, GenerationConfig