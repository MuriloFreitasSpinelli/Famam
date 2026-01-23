import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import ClassVar, Optional, Dict, Any, List


VALID_DISTRIBUTION_STRATEGIES = {'none', 'mirrored', 'multi_worker_mirrored', 'tpu'}
VALID_EXECUTION_MODES = {'local', 'slurm'}
VALID_CROSS_DEVICE_OPS = {'nccl', 'hierarchical_copy', 'reduction_to_one_device'}


@dataclass
class ModelTrainingConfig:
    """
    Configuration for LSTM-based music generation model training.

    Input tensor structure (from MusicDataset):
        - pianoroll: float32 (128, max_time_steps)
        - instrument_id: int32 scalar (0-128, where 128=drums)
        - genre_id: int32 scalar
    """

    VALID_OPTIMIZERS: ClassVar[set[str]] = {
        'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl'
    }
    VALID_LOSS_FUNCTIONS: ClassVar[set[str]] = {
        'mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy',
        'sparse_categorical_crossentropy', 'huber', 'log_cosh'
    }
    VALID_METRICS: ClassVar[set[str]] = {
        'accuracy', 'precision', 'recall', 'auc', 'mae', 'mse', 'rmse'
    }
    VALID_LR_SCHEDULES: ClassVar[set[str]] = {
        'constant', 'exponential_decay', 'step_decay', 'cosine_decay',
        'polynomial_decay', 'reduce_on_plateau'
    }

    # ============ Model Identification ============
    model_name: str

    # ============ Input Shape (from MusicDataset) ============
    num_pitches: int = 128  # MIDI pitch range
    max_time_steps: int = 1000  # Must match MusicDataset.max_time_steps
    num_instruments: int = 129  # 0-127 programs + 128 for drums

    # ============ LSTM Architecture ============
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])
    dense_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1
    bidirectional: bool = False
    return_sequences: bool = True

    # Embedding dimensions (for instrument/genre conditioning)
    instrument_embedding_dim: int = 16
    genre_embedding_dim: int = 16

    # ============ Training Hyperparameters ============
    batch_size: int = 32
    epochs: int = 100
    shuffle: bool = True
    shuffle_buffer_size: int = 10000  # Buffer size for tf.data.Dataset.shuffle()
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    loss_function: str = 'mse'
    metrics: List[str] = field(default_factory=lambda: ['mae'])

    # ============ Optimizer-specific Parameters ============
    # Adam optimizer
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7

    # SGD optimizer
    momentum: float = 0.0
    nesterov: bool = False

    # RMSprop optimizer
    rho: float = 0.9

    # ============ Learning Rate Schedule ============
    lr_schedule: str = 'constant'

    # Exponential decay
    lr_decay_rate: float = 0.96
    lr_decay_steps: int = 10000

    # Step decay
    lr_drop_factor: float = 0.5
    lr_drop_epochs: int = 10

    # Cosine decay
    lr_alpha: float = 0.0

    # Polynomial decay
    lr_power: float = 1.0
    lr_end_learning_rate: float = 0.0001

    # Reduce on plateau
    lr_patience: int = 5
    lr_min_delta: float = 0.0001
    lr_cooldown: int = 0
    lr_min_lr: float = 1e-7

    # ============ Regularization ============
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    kernel_constraint_max_value: Optional[float] = None
    recurrent_constraint_max_value: Optional[float] = None
    use_gradient_clipping: bool = False
    gradient_clip_value: float = 1.0

    # ============ Performance ============
    mixed_precision: bool = False

    # ============ Early Stopping ============
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    early_stopping_monitor: str = 'val_loss'
    early_stopping_mode: str = 'min'
    restore_best_weights: bool = True

    # ============ Model Checkpointing ============
    use_checkpointing: bool = True
    checkpoint_monitor: str = 'val_loss'
    checkpoint_mode: str = 'min'
    save_best_only: bool = True
    save_weights_only: bool = False

    # TensorBoard logging
    use_tensorboard: bool = True
    tensorboard_log_dir: str = './logs'
    tensorboard_histogram_freq: int = 1
    tensorboard_write_graph: bool = True
    tensorboard_update_freq: str = 'epoch'

    # ============ Output Settings ============
    output_dir: str = './models'
    save_history: bool = True
    save_final_model: bool = True

    # Random seed for reproducibility
    random_seed: Optional[int] = 42

    # ============ Distribution Settings ============
    distribution_strategy: str = 'none'  # 'none', 'mirrored', 'multi_worker_mirrored', 'tpu'
    batch_size_per_replica: Optional[int] = None  # If set, overrides batch_size for distributed
    mirrored_devices: Optional[List[str]] = None  # Specific devices for MirroredStrategy
    cross_device_ops: str = 'nccl'  # 'nccl', 'hierarchical_copy', 'reduction_to_one_device'

    # ============ Execution Mode ============
    execution_mode: str = 'local'  # 'local', 'slurm'

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate optimizer
        if self.optimizer.lower() not in self.VALID_OPTIMIZERS:
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_OPTIMIZERS))}"
            )

        # Validate loss function
        if self.loss_function.lower() not in self.VALID_LOSS_FUNCTIONS:
            raise ValueError(
                f"Invalid loss_function '{self.loss_function}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_LOSS_FUNCTIONS))}"
            )

        # Validate metrics
        for metric in self.metrics:
            if metric.lower() not in self.VALID_METRICS:
                raise ValueError(
                    f"Invalid metric '{metric}'. "
                    f"Must be one of: {', '.join(sorted(self.VALID_METRICS))}"
                )

        # Validate learning rate schedule
        if self.lr_schedule.lower() not in self.VALID_LR_SCHEDULES:
            raise ValueError(
                f"Invalid lr_schedule '{self.lr_schedule}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_LR_SCHEDULES))}"
            )

        # Validate dropout rates
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")

        if not 0 <= self.recurrent_dropout <= 1:
            raise ValueError(f"recurrent_dropout must be between 0 and 1, got {self.recurrent_dropout}")

        # Validate batch size and epochs
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        # Validate LSTM units
        if not self.lstm_units:
            raise ValueError("lstm_units must have at least one layer")

        # Validate distribution strategy
        if self.distribution_strategy not in VALID_DISTRIBUTION_STRATEGIES:
            raise ValueError(
                f"Invalid distribution_strategy '{self.distribution_strategy}'. "
                f"Must be one of: {', '.join(sorted(VALID_DISTRIBUTION_STRATEGIES))}"
            )

        # Validate execution mode
        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(
                f"Invalid execution_mode '{self.execution_mode}'. "
                f"Must be one of: {', '.join(sorted(VALID_EXECUTION_MODES))}"
            )

        # Validate cross device ops
        if self.cross_device_ops not in VALID_CROSS_DEVICE_OPS:
            raise ValueError(
                f"Invalid cross_device_ops '{self.cross_device_ops}'. "
                f"Must be one of: {', '.join(sorted(VALID_CROSS_DEVICE_OPS))}"
            )

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Get optimizer-specific keyword arguments."""
        kwargs = {'learning_rate': self.learning_rate}

        if self.optimizer.lower() == 'adam':
            kwargs.update({
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'epsilon': self.epsilon,
            })
        elif self.optimizer.lower() == 'sgd':
            kwargs.update({
                'momentum': self.momentum,
                'nesterov': self.nesterov,
            })
        elif self.optimizer.lower() == 'rmsprop':
            kwargs.update({
                'rho': self.rho,
                'epsilon': self.epsilon,
            })

        return kwargs

    def get_regularization_kwargs(self) -> Dict[str, Any]:
        """Get regularization keyword arguments for LSTM layers."""
        from tensorflow.keras import regularizers  # type: ignore

        kwargs = {}

        # L1/L2 regularization
        if self.l1_reg > 0 and self.l2_reg > 0:
            kwargs['kernel_regularizer'] = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        elif self.l1_reg > 0:
            kwargs['kernel_regularizer'] = regularizers.l1(self.l1_reg)
        elif self.l2_reg > 0:
            kwargs['kernel_regularizer'] = regularizers.l2(self.l2_reg)

        # Max norm constraints
        if self.kernel_constraint_max_value is not None:
            from tensorflow.keras.constraints import max_norm  # type: ignore
            kwargs['kernel_constraint'] = max_norm(self.kernel_constraint_max_value)

        if self.recurrent_constraint_max_value is not None:
            from tensorflow.keras.constraints import max_norm  # type: ignore
            kwargs['recurrent_constraint'] = max_norm(self.recurrent_constraint_max_value)

        return kwargs

    def get_input_shape(self) -> tuple:
        """Get the expected input shape for the pianoroll tensor."""
        return (self.num_pitches, self.max_time_steps)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Training configuration saved to: {output_path}")

    @classmethod
    def load(cls, path: str) -> 'ModelTrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        summary_lines = [
            "=" * 70,
            "Model Training Configuration Summary",
            "=" * 70,
            "",
            "Model:",
            f"  Name: {self.model_name}",
            f"  Input Shape: ({self.num_pitches}, {self.max_time_steps})",
            "",
            "LSTM Architecture:",
            f"  LSTM Units: {self.lstm_units}",
            f"  Bidirectional: {self.bidirectional}",
            f"  Dense Units: {self.dense_units}",
            f"  Dropout Rate: {self.dropout_rate}",
            f"  Recurrent Dropout: {self.recurrent_dropout}",
            "",
            "Embeddings:",
            f"  Instrument Embedding Dim: {self.instrument_embedding_dim}",
            f"  Genre Embedding Dim: {self.genre_embedding_dim}",
            "",
            "Training Hyperparameters:",
            f"  Optimizer: {self.optimizer}",
            f"  Learning Rate: {self.learning_rate}",
            f"  Batch Size: {self.batch_size}",
            f"  Epochs: {self.epochs}",
            f"  Loss Function: {self.loss_function}",
            f"  Metrics: {', '.join(self.metrics)}",
            "",
            "Regularization:",
            f"  L1: {self.l1_reg}, L2: {self.l2_reg}",
            f"  Gradient Clipping: {self.use_gradient_clipping}",
            "",
            "Training Techniques:",
            f"  Early Stopping: {self.use_early_stopping} (patience={self.early_stopping_patience})",
            f"  Checkpointing: {self.use_checkpointing}",
            f"  Learning Rate Schedule: {self.lr_schedule}",
            f"  Mixed Precision: {self.mixed_precision}",
            "",
            "Output:",
            f"  Output Directory: {self.output_dir}",
            f"  Random Seed: {self.random_seed}",
            "",
            "Distribution:",
            f"  Strategy: {self.distribution_strategy}",
            f"  Execution Mode: {self.execution_mode}",
            f"  Cross-Device Ops: {self.cross_device_ops}",
            f"  Batch Size per Replica: {self.batch_size_per_replica or self.batch_size}",
            "=" * 70,
        ]

        return "\n".join(summary_lines)
