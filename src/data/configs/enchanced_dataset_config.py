import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import ClassVar, Optional, List


@dataclass
class EnhancedDatasetConfig:
    """Configuration for creating an EnhancedMusicDataset from MIDI files."""
    
    # Dataset metadata
    dataset_name: str
    
    # Input/output paths
    input_dirs: List[str]  # List of directories to search for MIDI files
    output_path: str  # Path to save the .h5 dataset file
    
    # Filtering options
    allowed_instruments: Optional[List[str]] = None  # Filter by instrument names (MIDI program names)
    excluded_instruments: Optional[List[str]] = None  # Exclude specific instruments
    min_tracks: int = 1  # Minimum number of tracks
    max_tracks: int = 128  # Maximum number of tracks
    min_notes: int = 10  # Minimum total notes in the piece
    min_duration: Optional[float] = None  # Minimum duration in seconds
    max_duration: Optional[float] = None  # Maximum duration in seconds
    
    # Metadata extraction
    extract_genre_from_path: bool = True  # Extract genre from TSV file
    extract_artist_from_path: bool = True  # Extract artist from folder structure
    genre_tsv_path: Optional[str] = None  # Path to genre.tsv file for genre lookup
    artist_folder_level: int = -2  # Which folder level contains artist (negative = from end, -2 = parent folder)
    
    # Preprocessing options
    resolution: int = 24  # Ticks per quarter note
    quantize: bool = True  # Quantize note timings
    remove_empty_tracks: bool = True
    
    # Vocabulary building
    build_vocabulary: bool = True
    
    # Processing options
    max_samples: Optional[int] = None  # Limit number of samples (for testing)
    random_seed: Optional[int] = None
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate splits
        if self.min_tracks < 1:
            raise ValueError("min_tracks must be at least 1")
        
        if self.max_tracks < self.min_tracks:
            raise ValueError("max_tracks must be >= min_tracks")
        
        if self.min_notes < 0:
            raise ValueError("min_notes must be non-negative")
        
        # Validate duration constraints
        if self.min_duration is not None and self.min_duration < 0:
            raise ValueError("min_duration must be non-negative")
        
        if self.max_duration is not None and self.min_duration is not None:
            if self.max_duration < self.min_duration:
                raise ValueError("max_duration must be >= min_duration")
        
        # Validate instrument filters
        if self.allowed_instruments and self.excluded_instruments:
            overlap = set(self.allowed_instruments) & set(self.excluded_instruments)
            if overlap:
                raise ValueError(
                    f"Instruments cannot be in both allowed and excluded lists: {overlap}"
                )
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            path: Path to save the JSON file
        """
        config_dict = asdict(self)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EnhancedDatasetConfig':
        """Load configuration from JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            EnhancedDatasetConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def default(cls, dataset_name: str, input_dirs: List[str], output_path: str) -> 'EnhancedDatasetConfig':
        """Create a configuration with default values.
        
        Args:
            dataset_name: Name for the dataset
            input_dirs: List of directories to search for MIDI files
            output_path: Path to save the .h5 dataset file
            
        Returns:
            EnhancedDatasetConfig with default settings
        """
        return cls(
            dataset_name=dataset_name,
            input_dirs=input_dirs,
            output_path=output_path
        )