import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing Music objects.

    Controls resolution, quantization, segmentation, and augmentation.
    """

    # === Resolution ===
    target_resolution: int = 24  # Ticks per quarter note

    # === Quantization ===
    quantize: bool = True
    quantize_grid: int = 1  # Quantize to this many ticks (1 = full resolution)

    # === Track Cleanup ===
    remove_empty_tracks: bool = True

    # === Segmentation ===
    # Split music into fixed-length segments
    segment_length: Optional[int] = None  # Time steps; None = no segmentation
    max_padding_ratio: float = 0.3  # Discard if padding > 30% (require 70% content)

    # === Augmentation: Transposition ===
    # Shifts all pitches by constant interval, exploiting relative pitch structure
    # Increases effective corpus size by up to 12x while enforcing key invariance
    enable_transposition: bool = False
    transposition_semitones: Tuple[int, ...] = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)

    # === Augmentation: Tempo Variation ===
    # Modifies durations by factor α, improves temporal robustness
    enable_tempo_variation: bool = False
    tempo_variation_range: Tuple[float, float] = (0.9, 1.1)
    tempo_variation_steps: int = 3  # Number of variations to generate

    def __post_init__(self):
        """Validate configuration."""
        if self.target_resolution < 1:
            raise ValueError("target_resolution must be at least 1")

        if self.quantize_grid < 1:
            raise ValueError("quantize_grid must be at least 1")

        if self.segment_length is not None and self.segment_length < 1:
            raise ValueError("segment_length must be at least 1 if specified")

        if not 0.0 <= self.max_padding_ratio <= 1.0:
            raise ValueError("max_padding_ratio must be between 0.0 and 1.0")

        if self.tempo_variation_range[0] > self.tempo_variation_range[1]:
            raise ValueError("tempo_variation_range[0] must be <= tempo_variation_range[1]")

        if self.tempo_variation_range[0] <= 0:
            raise ValueError("tempo_variation_range values must be positive")

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PreprocessingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Convert lists back to tuples
        if "transposition_semitones" in data:
            data["transposition_semitones"] = tuple(data["transposition_semitones"])
        if "tempo_variation_range" in data:
            data["tempo_variation_range"] = tuple(data["tempo_variation_range"])

        return cls(**data)

    @classmethod
    def default(cls) -> "PreprocessingConfig":
        """Create configuration with sensible defaults."""
        return cls()

    @classmethod
    def with_augmentation(cls) -> "PreprocessingConfig":
        """Create configuration with all augmentations enabled."""
        return cls(
            enable_transposition=True,
            enable_tempo_variation=True,
        )

    @classmethod
    def with_segmentation(cls, segment_length: int) -> "PreprocessingConfig":
        """Create configuration with segmentation enabled."""
        return cls(segment_length=segment_length)
