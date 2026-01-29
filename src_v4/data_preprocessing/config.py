"""
Unified configuration for music dataset creation and preprocessing.

Combines dataset configuration (paths, filtering) with preprocessing
configuration (resolution, segmentation, augmentation) into a single class.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Tuple, Literal


@dataclass
class MusicDatasetConfig:
    """
    Unified configuration for creating and preprocessing a MusicDataset.

    Combines:
        - Dataset metadata and paths
        - MIDI filtering criteria
        - Preprocessing settings (resolution, quantization)
        - Segmentation settings
        - Augmentation settings
        - Encoding settings
    """

    # === Dataset Metadata ===
    name: str = "music_dataset"

    # === Input/Output Paths ===
    input_dirs: List[str] = field(default_factory=list)
    output_path: str = ""
    genre_tsv_path: Optional[str] = None

    # === Genre Filtering ===
    allowed_genres: Optional[List[str]] = None

    # === Track Filtering ===
    min_tracks: int = 1
    max_tracks: int = 16
    min_notes_per_track: int = 1

    # === Duration Filtering ===
    min_duration_seconds: Optional[float] = None
    max_duration_seconds: Optional[float] = None

    # === Resolution & Timing ===
    resolution: int = 24  # Ticks per quarter note
    time_signature: Tuple[int, int] = (4, 4)  # (numerator, denominator)

    # === Quantization ===
    quantize: bool = True
    quantize_grid: int = 1  # Quantize to this many ticks (1 = full resolution)

    # === Track Cleanup ===
    remove_empty_tracks: bool = True

    # === Segmentation ===
    segment_length: Optional[int] = None  # Time steps; None = no segmentation
    max_padding_ratio: float = 0.3  # Discard if padding > this ratio

    # === Augmentation: Transposition ===
    enable_transposition: bool = False
    transposition_semitones: Tuple[int, ...] = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)

    # === Augmentation: Tempo Variation ===
    enable_tempo_variation: bool = False
    tempo_variation_range: Tuple[float, float] = (0.9, 1.1)
    tempo_variation_steps: int = 3

    # === Encoding Settings ===
    encoder_type: Literal["event", "remi"] = "remi"
    max_seq_length: int = 2048  # Maximum token sequence length
    encode_velocity: bool = True
    positions_per_bar: int = 32  # For REMI encoder

    # === Processing Options ===
    max_samples: Optional[int] = None  # Limit samples (for testing)
    random_seed: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration."""
        # Track filtering
        if self.min_tracks < 1:
            raise ValueError("min_tracks must be at least 1")
        if self.max_tracks < self.min_tracks:
            raise ValueError("max_tracks must be >= min_tracks")
        if self.min_notes_per_track < 0:
            raise ValueError("min_notes_per_track must be non-negative")

        # Duration filtering
        if self.min_duration_seconds is not None and self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be non-negative")
        if self.max_duration_seconds is not None and self.min_duration_seconds is not None:
            if self.max_duration_seconds < self.min_duration_seconds:
                raise ValueError("max_duration_seconds must be >= min_duration_seconds")

        # Resolution & timing
        if self.resolution < 1:
            raise ValueError("resolution must be at least 1")
        if self.quantize_grid < 1:
            raise ValueError("quantize_grid must be at least 1")

        # Segmentation
        if self.segment_length is not None and self.segment_length < 1:
            raise ValueError("segment_length must be at least 1 if specified")
        if not 0.0 <= self.max_padding_ratio <= 1.0:
            raise ValueError("max_padding_ratio must be between 0.0 and 1.0")

        # Tempo variation
        if self.tempo_variation_range[0] > self.tempo_variation_range[1]:
            raise ValueError("tempo_variation_range[0] must be <= tempo_variation_range[1]")
        if self.tempo_variation_range[0] <= 0:
            raise ValueError("tempo_variation_range values must be positive")

        # Encoding
        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be at least 1")
        if self.positions_per_bar < 1:
            raise ValueError("positions_per_bar must be at least 1")

    @property
    def ticks_per_bar(self) -> int:
        """Calculate ticks per bar based on resolution and time signature."""
        beats_per_bar = self.time_signature[0] * (4 / self.time_signature[1])
        return int(beats_per_bar * self.resolution)

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to JSON file."""
        save_path = Path(path) if path else Path(self.output_path).with_suffix(".config.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        # Convert tuples to lists for JSON serialization
        data["time_signature"] = list(data["time_signature"])
        data["transposition_semitones"] = list(data["transposition_semitones"])
        data["tempo_variation_range"] = list(data["tempo_variation_range"])

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MusicDatasetConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Convert lists back to tuples
        if "time_signature" in data:
            data["time_signature"] = tuple(data["time_signature"])
        if "transposition_semitones" in data:
            data["transposition_semitones"] = tuple(data["transposition_semitones"])
        if "tempo_variation_range" in data:
            data["tempo_variation_range"] = tuple(data["tempo_variation_range"])

        return cls(**data)

    @classmethod
    def default(cls, name: str, input_dirs: List[str], output_path: str) -> "MusicDatasetConfig":
        """Create configuration with default values."""
        return cls(name=name, input_dirs=input_dirs, output_path=output_path)

    @classmethod
    def with_augmentation(
        cls,
        name: str,
        input_dirs: List[str],
        output_path: str,
    ) -> "MusicDatasetConfig":
        """Create configuration with all augmentations enabled."""
        return cls(
            name=name,
            input_dirs=input_dirs,
            output_path=output_path,
            enable_transposition=True,
            enable_tempo_variation=True,
        )

    @classmethod
    def with_segmentation(
        cls,
        name: str,
        input_dirs: List[str],
        output_path: str,
        segment_length: int,
    ) -> "MusicDatasetConfig":
        """Create configuration with segmentation enabled."""
        return cls(
            name=name,
            input_dirs=input_dirs,
            output_path=output_path,
            segment_length=segment_length,
        )

    def __repr__(self) -> str:
        return (
            f"MusicDatasetConfig(name='{self.name}', "
            f"encoder_type='{self.encoder_type}', "
            f"resolution={self.resolution}, "
            f"segment_length={self.segment_length}, "
            f"max_seq_length={self.max_seq_length})"
        )
