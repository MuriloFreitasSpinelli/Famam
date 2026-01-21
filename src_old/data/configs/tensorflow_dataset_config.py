import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import ClassVar, Optional, List


@dataclass
class TensorflowDatasetConfig:
    """
    Configuration for tensor processing with train/val/test splits.

    Supports multiple music representations from muspy:
    - pianoroll: Binary piano roll (128 pitches × time_steps) - best for polyphonic
    - pitch: Pitch sequence (variable length) - best for monophonic melodies
    - event: Event-based (note on/off with timing) - compact, preserves dynamics
    - note: Note representation (pitch, time, duration, velocity) - structured

    Multi-track modes:
    - single: Merge all tracks into one (default, original behavior)
    - multitrack: Keep tracks separate (max_tracks, 128, time_steps for pianoroll)
    - per_instrument: Group by instrument type
    """

    # === Valid Options ===
    VALID_REPRESENTATIONS: ClassVar[set[str]] = {
        'pianoroll',      # (128, time) or (tracks, 128, time) - polyphonic
        'pitch',          # (time,) pitch values - monophonic melody
        'event',          # (events, 3) - event_type, value, time
        'note',           # (notes, 4) - pitch, time, duration, velocity
    }
    VALID_TENSOR_TYPES: ClassVar[set[str]] = {
        'music-only',          # Just music data
        'music-genre',         # Music + genre_id
        'music-artist',        # Music + artist_id
        'music-instrument',    # Music + instrument_ids
        'music-genre-artist',  # Music + genre + artist
        'full',                # Music + genre + artist + instruments
    }
    VALID_TRACK_MODES: ClassVar[set[str]] = {
        'single',         # Merge all tracks (original behavior)
        'multitrack',     # Keep tracks separate
        'per_instrument', # Group by instrument program
    }
    DEFAULT_MAX_TIME_STEPS: ClassVar[int] = 1440  # ~30 seconds at 120 BPM, resolution 24
    DEFAULT_MAX_TRACKS: ClassVar[int] = 8

    # === Required Fields ===
    tensor_name: str
    tensor_type: str
    representation_type: str
    train_split: float
    val_split: float
    test_split: float
    output_dir: str

    # === Sequence/Time Options ===
    max_time_steps: int = 1440  # Sequence length (reduce for memory, e.g., 512, 1024)

    # === Multi-Track Options ===
    track_mode: str = 'single'  # 'single', 'multitrack', 'per_instrument'
    max_tracks: int = 8         # Max tracks to keep (pad/truncate)

    # === Conditioning Options (convenience flags) ===
    use_genre: bool = True
    use_artist: bool = False
    use_instruments: bool = False

    # === Pianoroll-specific Options ===
    pianoroll_encode_velocity: bool = False  # Include velocity in pianoroll

    # === Event/Note-specific Options ===
    max_events: int = 2048      # Max events for event representation
    max_notes: int = 1024       # Max notes for note representation

    # === Instrument Handling ===
    instrument_groups: Optional[List[str]] = None  # Group instruments (e.g., ['piano', 'guitar', 'bass', 'drums'])

    # === Random State ===
    random_state: Optional[int] = 42
     
    def __post_init__(self):
        """Validate configuration parameters."""
        # Normalize representation type (handle legacy 'piano-roll')
        if self.representation_type == 'piano-roll':
            self.representation_type = 'pianoroll'

        # Validate representation type
        if self.representation_type not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"Invalid representation_type '{self.representation_type}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_REPRESENTATIONS))}"
            )

        # Validate tensor type
        if self.tensor_type not in self.VALID_TENSOR_TYPES:
            raise ValueError(
                f"Invalid tensor_type '{self.tensor_type}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_TENSOR_TYPES))}"
            )

        # Validate track mode
        if self.track_mode not in self.VALID_TRACK_MODES:
            raise ValueError(
                f"Invalid track_mode '{self.track_mode}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_TRACK_MODES))}"
            )

        # Validate that splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError(
                f"Splits must sum to 1.0, got {total_split} "
                f"(train={self.train_split}, val={self.val_split}, test={self.test_split})"
            )

        # Validate max_tracks
        if self.max_tracks < 1:
            raise ValueError(f"max_tracks must be >= 1, got {self.max_tracks}")

        # Auto-set tensor_type based on convenience flags if using defaults
        if self.tensor_type == 'music-only':
            self._auto_set_tensor_type()

    def _auto_set_tensor_type(self):
        """Auto-determine tensor_type from convenience flags."""
        has_genre = self.use_genre
        has_artist = self.use_artist
        has_instruments = self.use_instruments

        if has_genre and has_artist and has_instruments:
            self.tensor_type = 'full'
        elif has_genre and has_artist:
            self.tensor_type = 'music-genre-artist'
        elif has_genre and has_instruments:
            self.tensor_type = 'full'  # genre + instruments = full
        elif has_genre:
            self.tensor_type = 'music-genre'
        elif has_artist:
            self.tensor_type = 'music-artist'
        elif has_instruments:
            self.tensor_type = 'music-instrument'
        # else: keep 'music-only'

    @property
    def is_multitrack(self) -> bool:
        """Check if using multi-track mode."""
        return self.track_mode in ('multitrack', 'per_instrument')

    @property
    def music_shape(self) -> tuple:
        """
        Get expected music tensor shape based on configuration.

        Returns shape WITHOUT batch dimension.
        """
        if self.representation_type == 'pianoroll':
            if self.is_multitrack:
                return (self.max_tracks, 128, self.max_time_steps)
            else:
                return (128, self.max_time_steps)
        elif self.representation_type == 'pitch':
            return (self.max_time_steps,)
        elif self.representation_type == 'event':
            return (self.max_events, 3)  # event_type, value, time
        elif self.representation_type == 'note':
            return (self.max_notes, 4)   # pitch, time, duration, velocity
        else:
            raise ValueError(f"Unknown representation: {self.representation_type}")

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "TensorflowDatasetConfig:",
            f"  Name: {self.tensor_name}",
            f"  Representation: {self.representation_type}",
            f"  Track Mode: {self.track_mode}",
            f"  Max Tracks: {self.max_tracks}" if self.is_multitrack else "",
            f"  Max Time Steps: {self.max_time_steps}",
            f"  Expected Shape: {self.music_shape}",
            f"  Conditioning: genre={self.use_genre}, artist={self.use_artist}, instruments={self.use_instruments}",
            f"  Splits: train={self.train_split}, val={self.val_split}, test={self.test_split}",
        ]
        return "\n".join(l for l in lines if l)

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
    def load(cls, path: str) -> 'TensorflowDatasetConfig':
        """Load configuration from JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            TensorflowDatasetConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Backward compatibility: add defaults for new fields
        defaults = {
            'max_time_steps': cls.DEFAULT_MAX_TIME_STEPS,
            'track_mode': 'single',
            'max_tracks': cls.DEFAULT_MAX_TRACKS,
            'use_genre': True,
            'use_artist': False,
            'use_instruments': False,
            'pianoroll_encode_velocity': False,
            'max_events': 2048,
            'max_notes': 1024,
            'instrument_groups': None,
            'random_state': 42,
        }

        for key, default_value in defaults.items():
            if key not in config_dict:
                config_dict[key] = default_value

        # Handle legacy representation type
        if config_dict.get('representation_type') == 'piano-roll':
            config_dict['representation_type'] = 'pianoroll'

        return cls(**config_dict)

    @classmethod
    def default_multitrack(
        cls,
        tensor_name: str,
        output_dir: str = './models',
        max_tracks: int = 8,
        max_time_steps: int = 1024,
    ) -> 'TensorflowDatasetConfig':
        """Create a default multi-track configuration."""
        return cls(
            tensor_name=tensor_name,
            tensor_type='music-genre',
            representation_type='pianoroll',
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            output_dir=output_dir,
            track_mode='multitrack',
            max_tracks=max_tracks,
            max_time_steps=max_time_steps,
            use_genre=True,
            use_instruments=True,
        )

    @classmethod
    def default_single_track(
        cls,
        tensor_name: str,
        output_dir: str = './models',
        max_time_steps: int = 1024,
    ) -> 'TensorflowDatasetConfig':
        """Create a default single-track configuration (original behavior)."""
        return cls(
            tensor_name=tensor_name,
            tensor_type='music-genre',
            representation_type='pianoroll',
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            output_dir=output_dir,
            track_mode='single',
            max_time_steps=max_time_steps,
            use_genre=True,
        )