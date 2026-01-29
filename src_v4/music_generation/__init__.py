"""
Music generation module.

Provides autoregressive generation using trained Transformer and LSTM models
with various sampling strategies and MIDI export.

Includes:
    - Single-track generation (MusicGenerator)
    - Multi-track generation where all instruments know each other (MultiTrackGenerator)
    - Rule-based drum pattern generation (DrumPatternGenerator)
"""

from .generator import (
    MusicGenerator,
    GenerationConfig,
)
from .multitrack_generator import (
    MultiTrackGenerator,
    MultiTrackConfig,
)
from .drum_generator import (
    DrumPatternGenerator,
    DrumPattern,
    DrumNote,
    generate_alternative_rock_drums,
    ROCK_PATTERNS,
    ALTERNATIVE_PATTERNS,
)

__all__ = [
    'MusicGenerator',
    'GenerationConfig',
    'MultiTrackGenerator',
    'MultiTrackConfig',
    'DrumPatternGenerator',
    'DrumPattern',
    'DrumNote',
    'generate_alternative_rock_drums',
    'ROCK_PATTERNS',
    'ALTERNATIVE_PATTERNS',
]
