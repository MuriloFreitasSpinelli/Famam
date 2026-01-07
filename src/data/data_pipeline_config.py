from dataclasses import dataclass
from typing import Optional, List, Tuple
import json


@dataclass
class DataPipelineConfig:
    """Configuration for the data pipeline, including GigaMIDI filters."""

    # GigaMIDI filter settings
    bpm_range: Optional[Tuple[float, float]] = None
    genres: Optional[List[str]] = None
    num_tracks_range: Optional[Tuple[int, int]] = None
    loop_instruments: Optional[List[str]] = None
    artists: Optional[List[str]] = None
    max_samples: Optional[int] = None

    # Output settings
    output_path: str = "data/processed/dataset.h5"

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        config_dict = {
            'bpm_range': self.bpm_range,
            'genres': self.genres,
            'num_tracks_range': self.num_tracks_range,
            'loop_instruments': self.loop_instruments,
            'artists': self.artists,
            'max_samples': self.max_samples,
            'output_path': self.output_path
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'DataPipelineConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Convert lists back to tuples where needed
        if config_dict.get('bpm_range'):
            config_dict['bpm_range'] = tuple(config_dict['bpm_range'])
        if config_dict.get('num_tracks_range'):
            config_dict['num_tracks_range'] = tuple(config_dict['num_tracks_range'])
        return cls(**config_dict)
