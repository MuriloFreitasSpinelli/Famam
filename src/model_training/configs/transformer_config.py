"""
Configuration for Transformer-based autoregressive music generation.

Uses event representation for sequence modeling with causal self-attention.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import ClassVar, Optional, List


@dataclass
class TransformerTrainingConfig:
    """
    Configuration for Transformer-based music generation model training.

    The model generates music as a sequence of event tokens autoregressively.

    Input: [BOS] [GENRE] [INSTRUMENT] [event_1] ... [event_N]
    Output: Logits over vocabulary for next token prediction
    """

    VALID_OPTIMIZERS: ClassVar[set[str]] = {
        'adam', 'adamw', 'sgd', 'rmsprop'
    }

    # ============ Model Identification ============
    model_name: str
    model_type: str = "transformer"  # Identifies this as a transformer model

    # ============ Architecture ============
    max_seq_length: int = 2048  # Maximum sequence length
    num_layers: int = 6  # Number of transformer blocks
    d_model: int = 512  # Model dimension / embedding size
    num_heads: int = 8  # Number of attention heads
    d_ff: int = 2048  # Feed-forward hidden dimension
    dropout_rate: float = 0.1  # Dropout rate

    # Embedding dimensions for conditioning (if separate from d_model)
    use_separate_conditioning_embeddings: bool = False
    conditioning_embedding_dim: int = 64

    # Position embedding
    use_learned_positional_embedding: bool = True  # vs sinusoidal

    # ============ Relative Attention (Music Transformer) ============
    # Use relative positional attention instead of absolute (highly recommended for music)
    use_relative_attention: bool = True
    # Maximum relative distance to consider (usually seq_length // 2)
    max_relative_position: int = 512

    # ============ Training Hyperparameters ============
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    warmup_steps: int = 4000  # Linear warmup steps

    # Label smoothing for classification
    label_smoothing: float = 0.1

    # Optimizer
    optimizer: str = 'adam'
    weight_decay: float = 0.01  # For AdamW

    # Adam parameters
    beta_1: float = 0.9
    beta_2: float = 0.98
    epsilon: float = 1e-9

    # ============ Learning Rate Schedule ============
    # Transformer schedule: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    use_transformer_lr_schedule: bool = True
    lr_decay_rate: float = 0.96  # For exponential decay (if not using transformer schedule)

    # ============ Regularization ============
    use_gradient_clipping: bool = True
    gradient_clip_value: float = 1.0

    # ============ Early Stopping ============
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    early_stopping_monitor: str = 'val_loss'

    # ============ Checkpointing ============
    use_checkpointing: bool = True
    checkpoint_monitor: str = 'val_loss'
    save_best_only: bool = True

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = './logs'

    # ============ Output ============
    output_dir: str = './models'
    save_history: bool = True
    save_final_model: bool = True

    # Random seed
    random_seed: Optional[int] = 42

    # ============ Dataset ============
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    encode_velocity: bool = False  # Whether to include velocity tokens

    # ============ Distribution ============
    distribution_strategy: str = 'none'
    batch_size_per_replica: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        import os
        self.output_dir = os.path.expandvars(os.path.expanduser(self.output_dir))
        self.tensorboard_log_dir = os.path.expandvars(os.path.expanduser(self.tensorboard_log_dir))

        # Validate architecture
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )

        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")

        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        # Validate optimizer
        if self.optimizer.lower() not in self.VALID_OPTIMIZERS:
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_OPTIMIZERS))}"
            )

        # Validate dropout
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")

        # Validate label smoothing
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"label_smoothing must be between 0 and 1, got {self.label_smoothing}")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.num_heads

    def get_input_shape(self) -> tuple:
        """Get the expected input shape for sequences."""
        return (self.max_seq_length,)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Transformer configuration saved to: {output_path}")

    @classmethod
    def load(cls, path: str) -> 'TransformerTrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        summary_lines = [
            "=" * 70,
            "Transformer Training Configuration Summary",
            "=" * 70,
            "",
            "Model:",
            f"  Name: {self.model_name}",
            f"  Type: {self.model_type}",
            f"  Max Sequence Length: {self.max_seq_length}",
            "",
            "Architecture:",
            f"  Layers: {self.num_layers}",
            f"  Model Dimension: {self.d_model}",
            f"  Attention Heads: {self.num_heads}",
            f"  Head Dimension: {self.head_dim}",
            f"  Feed-Forward Dimension: {self.d_ff}",
            f"  Dropout: {self.dropout_rate}",
            "",
            "Attention:",
            f"  Relative Attention: {self.use_relative_attention} (Music Transformer)",
            f"  Max Relative Position: {self.max_relative_position}",
            "",
            "Training:",
            f"  Optimizer: {self.optimizer}",
            f"  Learning Rate: {self.learning_rate}",
            f"  Warmup Steps: {self.warmup_steps}",
            f"  Batch Size: {self.batch_size}",
            f"  Epochs: {self.epochs}",
            f"  Label Smoothing: {self.label_smoothing}",
            "",
            "Regularization:",
            f"  Gradient Clipping: {self.use_gradient_clipping} (value={self.gradient_clip_value})",
            f"  Weight Decay: {self.weight_decay}",
            "",
            "Callbacks:",
            f"  Early Stopping: {self.use_early_stopping} (patience={self.early_stopping_patience})",
            f"  Checkpointing: {self.use_checkpointing}",
            f"  TensorBoard: {self.use_tensorboard}",
            "",
            "Output:",
            f"  Directory: {self.output_dir}",
            "=" * 70,
        ]
        return "\n".join(summary_lines)


# Preset configurations for quick experimentation
def get_small_config(model_name: str = "music_transformer_small") -> TransformerTrainingConfig:
    """Small configuration for initial testing (low memory, ~4GB VRAM)."""
    return TransformerTrainingConfig(
        model_name=model_name,
        max_seq_length=512,
        num_layers=4,
        d_model=256,
        num_heads=4,
        d_ff=1024,
        batch_size=4,
        epochs=50,
    )


def get_medium_config(model_name: str = "music_transformer_medium") -> TransformerTrainingConfig:
    """Medium configuration for reasonable quality (~8GB VRAM)."""
    return TransformerTrainingConfig(
        model_name=model_name,
        max_seq_length=1024,
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        batch_size=4,
        epochs=100,
    )


def get_large_config(model_name: str = "music_transformer_large") -> TransformerTrainingConfig:
    """Large configuration for high quality (requires ~16GB+ VRAM)."""
    return TransformerTrainingConfig(
        model_name=model_name,
        max_seq_length=1024,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=3072,
        batch_size=2,
        epochs=200,
    )
