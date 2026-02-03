"""
Client module - CLI interfaces for the music generation pipeline.

Two specialized CLIs:

1. Experiment CLI - For dataset and training workflows:
    from src.client import experiment_main
    experiment_main()

2. Generation CLI - For music generation:
    from src.client import generation_main
    generation_main()

Or run from command line:
    python -m src.client.experiment_cli
    python -m src.client.generation_cli
"""

from .pipeline import MusicPipeline, PipelineConfig
from .experiment_cli import main as experiment_main, ExperimentCLI
from .generation_cli import main as generation_main, GenerationCLI

__all__ = [
    'MusicPipeline',
    'PipelineConfig',
    'experiment_main',
    'ExperimentCLI',
    'generation_main',
    'GenerationCLI',
]
