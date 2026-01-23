import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class MusicDatasetConfig:
    """
    Configuration for creating a MusicDataset from MIDI files.

    Controls MIDI loading, filtering, and pianoroll generation settings.
    """

    # === Dataset Metadata ===
    name: str

    # === Input/Output Paths ===
    input_dirs: List[str]
    output_path: str
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

    # === Pianoroll Settings ===
    resolution: int = 24  # Ticks per quarter note
    max_time_steps: int = 512  # Fixed time dimension for pianorolls (matches segment_length)

    # === Processing Options ===
    max_samples: Optional[int] = None  # Limit samples (for testing)
    random_seed: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.min_tracks < 1:
            raise ValueError("min_tracks must be at least 1")

        if self.max_tracks < self.min_tracks:
            raise ValueError("max_tracks must be >= min_tracks")

        if self.min_notes_per_track < 0:
            raise ValueError("min_notes_per_track must be non-negative")

        if self.min_duration_seconds is not None and self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be non-negative")

        if self.max_duration_seconds is not None and self.min_duration_seconds is not None:
            if self.max_duration_seconds < self.min_duration_seconds:
                raise ValueError("max_duration_seconds must be >= min_duration_seconds")

        if self.resolution < 1:
            raise ValueError("resolution must be at least 1")

        if self.max_time_steps < 1:
            raise ValueError("max_time_steps must be at least 1")

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to JSON file."""
        save_path = Path(path) if path else Path(self.output_path).with_suffix(".config.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MusicDatasetConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls(**json.load(f))

    @classmethod
    def default(cls, name: str, input_dirs: List[str], output_path: str) -> "MusicDatasetConfig":
        """Create configuration with default values."""
        return cls(name=name, input_dirs=input_dirs, output_path=output_path)
