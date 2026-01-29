"""
Client module - High-level API for music generation pipeline.

Provides unified interface for:
    - Dataset building
    - Model training
    - Music generation

Quick start:
    from src_v4.client import MusicPipeline, PipelineConfig

    # Option 1: Use defaults
    pipeline = MusicPipeline()

    # Option 2: Custom config
    config = PipelineConfig(
        encoder_type="remi",
        model_type="transformer",
        d_model=256,
        num_layers=4,
    )
    pipeline = MusicPipeline(config)

    # Build, train, generate
    pipeline.build_dataset("data/midi")
    pipeline.train(epochs=50)
    pipeline.save_model("models/my_model")
    pipeline.generate_midi("output.mid", genre_id=0)

For interactive menu:
    from src_v4.client import menu_main
    menu_main()
"""


from .pipeline import MusicPipeline, PipelineConfig
from .cli import main as cli_main, InteractiveShell
from .menu_cli import main as menu_main, MenuCLI

__all__ = [
    'MusicPipeline',
    'PipelineConfig',
    'cli_main',
    'InteractiveShell',
    'menu_main',
    'MenuCLI',
]
