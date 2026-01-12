import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import ClassVar


@dataclass
class DataTensorsConfig:
    """Configuration for tensor processing with train/val/test splits."""
    
    VALID_REPRESENTATIONS: ClassVar[set[str]] = {'pitch', 'piano-roll', 'event', 'note'}
    VALID_TENSOR_TYPES: ClassVar[set[str]] = {'music-only', 'music-genre', 'music-instrument', 'full'}

    tensor_name: str
    tensor_type: str
    representation_type: str
    train_split: float
    val_split: float
    test_split: float
    output_dir: str
     
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate representation type
        if self.representation_type not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"Invalid representation_type '{self.representation_type}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_REPRESENTATIONS))}"
            )
        if self.tensor_type not in self.VALID_TENSOR_TYPES:
            raise ValueError(
                f"Invalid tensor_type '{self.tensor_type}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_TENSOR_TYPES))}"
            )
        
        # Validate that splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError(
                f"Splits must sum to 1.0, got {total_split} "
                f"(train={self.train_split}, val={self.val_split}, test={self.test_split})"
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
    def load(cls, path: str) -> 'DataTensorsConfig':
        """Load configuration from JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            TensorsConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)