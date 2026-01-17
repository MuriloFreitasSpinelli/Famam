import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class EnhancedDatasetConfig:
    """Configuration for creating an EnhancedMusicDataset from MIDI files."""
    
    # Dataset metadata
    dataset_name: str
    
    # Input/output paths
    input_dirs: List[str]  # Directories to search for MIDI files
    output_path: str  # Path to save the .h5 dataset file
    genre_tsv_path: Optional[str] = None  # Path to genre.tsv file
    
    # Filtering options
    allowed_instruments: Optional[List[str]] = None  # Filter by instrument names
    excluded_instruments: Optional[List[str]] = None  # Exclude specific instruments
    min_tracks: int = 1  # Minimum number of tracks
    max_tracks: int = 16  # Maximum number of tracks
    min_notes: int = 10  # Minimum total notes in the piece
    min_duration: Optional[float] = None  # Minimum duration in seconds
    max_duration: Optional[float] = None  # Maximum duration in seconds
    
    # Preprocessing options
    resolution: int = 24  # Ticks per quarter note
    quantize: bool = True  # Quantize note timings
    remove_empty_tracks: bool = True
    
    # Processing options
    max_samples: Optional[int] = None  # Limit number of samples (for testing)
    random_seed: Optional[int] = None
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_tracks < 1:
            raise ValueError("min_tracks must be at least 1")
        
        if self.max_tracks < self.min_tracks:
            raise ValueError("max_tracks must be >= min_tracks")
        
        if self.min_notes < 0:
            raise ValueError("min_notes must be non-negative")
        
        if self.min_duration is not None and self.min_duration < 0:
            raise ValueError("min_duration must be non-negative")
        
        if self.max_duration is not None and self.min_duration is not None:
            if self.max_duration < self.min_duration:
                raise ValueError("max_duration must be >= min_duration")
        
        if self.allowed_instruments and self.excluded_instruments:
            overlap = set(self.allowed_instruments) & set(self.excluded_instruments)
            if overlap:
                raise ValueError(
                    f"Instruments cannot be in both allowed and excluded lists: {overlap}"
                )
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EnhancedDatasetConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)