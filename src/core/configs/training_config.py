"""
Training configuration for music generation models.

Unified config supporting both Transformer and LSTM architectures.

Author: Murilo de Freitas Spinelli
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import ClassVar, Optional, List, Literal, Tuple
import os


@dataclass
class TrainingConfig:
    """
    Unified training configuration for music generation models.

    Supports both Transformer and LSTM architectures with shared
    training parameters and architecture-specific settings.
    """

    VALID_OPTIMIZERS: ClassVar[set] = {'adam', 'adamw', 'sgd', 'rmsprop'}
    VALID_MODEL_TYPES: ClassVar[set] = {'transformer', 'lstm'}

    model_name: str = "music_model"
    model_type: Literal["transformer", "lstm"] = "transformer"

    max_seq_length: int = 2048
    d_model: int = 512
    dropout_rate: float = 0.1

    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    use_relative_attention: bool = True
    max_relative_position: int = 512

    lstm_units: Tuple[int, ...] = (512, 512)
    bidirectional: bool = False
    recurrent_dropout: float = 0.0

    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    label_smoothing: float = 0.1

    optimizer: str = 'adam'
    weight_decay: float = 0.01
    beta_1: float = 0.9
    beta_2: float = 0.98
    epsilon: float = 1e-9

    use_lr_schedule: bool = True
    lr_schedule_type: Literal["transformer", "cosine", "constant"] = "transformer"

    use_gradient_clipping: bool = True
    gradient_clip_value: float = 1.0

    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    early_stopping_monitor: str = 'val_loss'

    use_checkpointing: bool = True
    checkpoint_monitor: str = 'val_loss'
    save_best_only: bool = True

    use_tensorboard: bool = True
    tensorboard_log_dir: str = './logs'

    output_dir: str = './models'
    save_history: bool = True
    save_final_model: bool = True

    random_seed: Optional[int] = 42

    shuffle: bool = True
    shuffle_buffer_size: int = 10000

    distribution_strategy: str = 'none'
    batch_size_per_replica: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        self.output_dir = os.path.expandvars(os.path.expanduser(self.output_dir))
        self.tensorboard_log_dir = os.path.expandvars(os.path.expanduser(self.tensorboard_log_dir))

        if self.model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type '{self.model_type}'. Must be one of: {self.VALID_MODEL_TYPES}")

        if self.optimizer.lower() not in self.VALID_OPTIMIZERS:
            raise ValueError(f"Invalid optimizer '{self.optimizer}'. Must be one of: {self.VALID_OPTIMIZERS}")

        if self.model_type == "transformer":
            if self.d_model % self.num_heads != 0:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")

        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"label_smoothing must be between 0 and 1, got {self.label_smoothing}")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head (Transformer only)."""
        return self.d_model // self.num_heads

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        # Convert tuples to lists for JSON
        config_dict['lstm_units'] = list(config_dict['lstm_units'])

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Convert lists back to tuples
        if 'lstm_units' in config_dict:
            config_dict['lstm_units'] = tuple(config_dict['lstm_units'])

        return cls(**config_dict)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            f"Training Configuration: {self.model_name}",
            "=" * 60,
            "",
            f"Model Type: {self.model_type.upper()}",
            f"Max Sequence Length: {self.max_seq_length}",
            f"Embedding Dimension: {self.d_model}",
            "",
        ]

        if self.model_type == "transformer":
            lines.extend([
                "Transformer Architecture:",
                f"  Layers: {self.num_layers}",
                f"  Attention Heads: {self.num_heads}",
                f"  Head Dimension: {self.head_dim}",
                f"  FFN Dimension: {self.d_ff}",
                f"  Relative Attention: {self.use_relative_attention}",
                "",
            ])
        else:
            lines.extend([
                "LSTM Architecture:",
                f"  Units: {self.lstm_units}",
                f"  Bidirectional: {self.bidirectional}",
                f"  Recurrent Dropout: {self.recurrent_dropout}",
                "",
            ])

        lines.extend([
            "Training:",
            f"  Optimizer: {self.optimizer}",
            f"  Learning Rate: {self.learning_rate}",
            f"  Warmup Steps: {self.warmup_steps}",
            f"  Batch Size: {self.batch_size}",
            f"  Epochs: {self.epochs}",
            f"  Dropout: {self.dropout_rate}",
            f"  Label Smoothing: {self.label_smoothing}",
            "",
            f"Output Directory: {self.output_dir}",
            "=" * 60,
        ])

        return "\n".join(lines)


# Preset configurations
def get_transformer_small(model_name: str = "transformer_small") -> TrainingConfig:
    """Small transformer for testing (~4GB VRAM)."""
    return TrainingConfig(
        model_name=model_name,
        model_type="transformer",
        max_seq_length=512,
        num_layers=4,
        d_model=256,
        num_heads=4,
        d_ff=1024,
        batch_size=8,
    )


def get_transformer_medium(model_name: str = "transformer_medium") -> TrainingConfig:
    """Medium transformer (~8GB VRAM)."""
    return TrainingConfig(
        model_name=model_name,
        model_type="transformer",
        max_seq_length=1024,
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        batch_size=4,
    )


def get_transformer_large(model_name: str = "transformer_large") -> TrainingConfig:
    """Large transformer (~16GB+ VRAM)."""
    return TrainingConfig(
        model_name=model_name,
        model_type="transformer",
        max_seq_length=2048,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=3072,
        batch_size=2,
    )


def get_lstm_small(model_name: str = "lstm_small") -> TrainingConfig:
    """Small LSTM for testing."""
    return TrainingConfig(
        model_name=model_name,
        model_type="lstm",
        max_seq_length=512,
        d_model=256,
        lstm_units=(256, 256),
        batch_size=16,
    )


def get_lstm_medium(model_name: str = "lstm_medium") -> TrainingConfig:
    """Medium LSTM."""
    return TrainingConfig(
        model_name=model_name,
        model_type="lstm",
        max_seq_length=1024,
        d_model=512,
        lstm_units=(512, 512),
        batch_size=8,
    )
