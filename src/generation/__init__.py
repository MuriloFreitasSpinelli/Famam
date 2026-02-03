"""
Music generation module.

Provides autoregressive generation using trained Transformer and LSTM models
with various sampling strategies and MIDI export.

Generators:
    - MonoGenerator: Single-track generation (tracks generated independently)
    - PolyGenerator: Multi-track generation (all tracks aware of each other)

Base classes:
    - BaseGenerator: Abstract base class for generators
    - GeneratorConfig: Base configuration dataclass
"""

from .base_generator import (
    BaseGenerator,
    GeneratorConfig,
)
from .mono_generator import (
    MonoGenerator,
    MonoGeneratorConfig,
    # Backwards compatibility
    MusicGenerator,
    GenerationConfig,
)
from .poly_generator import (
    PolyGenerator,
    PolyGeneratorConfig,
    # Backwards compatibility
    MultiTrackGenerator,
    MultiTrackConfig,
)

__all__ = [
    # Base
    'BaseGenerator',
    'GeneratorConfig',
    # Mono (single-track)
    'MonoGenerator',
    'MonoGeneratorConfig',
    # Poly (multi-track)
    'PolyGenerator',
    'PolyGeneratorConfig',
    # Backwards compatibility aliases
    'MusicGenerator',
    'GenerationConfig',
    'MultiTrackGenerator',
    'MultiTrackConfig',
]
