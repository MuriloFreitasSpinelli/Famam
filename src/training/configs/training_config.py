import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import ClassVar, Optional, Dict, Any, List, Union


@dataclass
class TrainingConfig:
    """Configuration for LSTM model training with hyperparameter tuning support."""
    
    # Valid options for configuration validation
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
    
    # ============ Model Architecture ============
    model_name: str
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])  # Units per LSTM layer
    dense_units: List[int] = field(default_factory=lambda: [64, 32])  # Units per dense layer
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1
    bidirectional: bool = False
    return_sequences: bool = True  # For stacked LSTMs
    
    # ============ Training Hyperparameters ============
    batch_size: int = 32
    epochs: int = 100
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
    lr_alpha: float = 0.0  # Minimum learning rate
    
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
    kernel_constraint_max_value: Optional[float] = None  # Max norm constraint
    recurrent_constraint_max_value: Optional[float] = None
    
    # ============ Early Stopping ============
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    early_stopping_monitor: str = 'val_loss'
    early_stopping_mode: str = 'min'  # 'min' or 'max'
    restore_best_weights: bool = True
    
    # ============ Model Checkpointing ============
    use_checkpointing: bool = True
    checkpoint_monitor: str = 'val_loss'
    checkpoint_mode: str = 'min'  # 'min' or 'max'
    save_best_only: bool = True
    save_weights_only: bool = False
    
    # ============ Data Processing ============
    validation_split: float = 0.0  # Use if not using separate validation set
    shuffle: bool = True
    class_weight: Optional[Dict[int, float]] = None
    sample_weight: Optional[str] = None  # Path to sample weights
    
    # ============ Advanced Training Techniques ============
    use_gradient_clipping: bool = False
    gradient_clip_value: Optional[float] = None
    gradient_clip_norm: Optional[float] = None
    
    mixed_precision: bool = False  # For faster training on supported hardware
    
    # TensorBoard logging
    use_tensorboard: bool = True
    tensorboard_log_dir: str = './logs'
    tensorboard_histogram_freq: int = 1
    tensorboard_write_graph: bool = True
    tensorboard_update_freq: str = 'epoch'  # 'batch', 'epoch', or integer
    
    # ============ Cross-Validation Settings ============
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_shuffle: bool = True
    cv_random_state: Optional[int] = 42
    
    # ============ Hyperparameter Tuning (Sklearn-style) ============
    use_hyperparameter_tuning: bool = False
    tuning_method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    
    # Parameter grid for tuning (if None, will use single values from above)
    param_grid: Optional[Dict[str, List[Any]]] = None
    
    # Random search specific
    n_iter: int = 10  # Number of parameter settings sampled
    
    # Bayesian optimization specific
    bayesian_n_calls: int = 50
    bayesian_n_initial_points: int = 10
    
    # Tuning validation
    tuning_cv_folds: int = 3
    tuning_scoring: str = 'neg_mean_squared_error'
    tuning_n_jobs: int = -1  # Use all available cores
    tuning_verbose: int = 2
    
    # ============ Output Settings ============
    output_dir: str = './models'
    save_history: bool = True
    save_final_model: bool = True
    
    # Random seed for reproducibility
    random_seed: Optional[int] = 42
    
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
        
        # Validate tuning method
        if self.use_hyperparameter_tuning:
            valid_tuning_methods = {'grid_search', 'random_search', 'bayesian'}
            if self.tuning_method not in valid_tuning_methods:
                raise ValueError(
                    f"Invalid tuning_method '{self.tuning_method}'. "
                    f"Must be one of: {', '.join(sorted(valid_tuning_methods))}"
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
        
        # Create default param_grid if using tuning but no grid provided
        if self.use_hyperparameter_tuning and self.param_grid is None:
            self.param_grid = self._create_default_param_grid()
    
    def _create_default_param_grid(self) -> Dict[str, List[Any]]:
        """Create a default parameter grid for hyperparameter tuning."""
        return {
            'lstm_units': [[64], [128], [128, 64], [256, 128]],
            'dense_units': [[32], [64], [64, 32]],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'optimizer': ['adam', 'rmsprop'],
        }
    
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
        """Get regularization keyword arguments for layers."""
        from tensorflow.keras import regularizers # type: ignore
        
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
            from tensorflow.keras.constraints import max_norm # type: ignore
            kwargs['kernel_constraint'] = max_norm(self.kernel_constraint_max_value)
        
        if self.recurrent_constraint_max_value is not None:
            from tensorflow.keras.constraints import max_norm # type: ignore
            kwargs['recurrent_constraint'] = max_norm(self.recurrent_constraint_max_value)
        
        return kwargs
    
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
        
        print(f"Training configuration saved to: {output_path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            TrainingConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        summary_lines = [
            "=" * 70,
            "Training Configuration Summary",
            "=" * 70,
            "",
            "Model Architecture:",
            f"  Model Name: {self.model_name}",
            f"  LSTM Units: {self.lstm_units}",
            f"  Dense Units: {self.dense_units}",
            f"  Bidirectional: {self.bidirectional}",
            f"  Dropout Rate: {self.dropout_rate}",
            f"  Recurrent Dropout: {self.recurrent_dropout}",
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
        ]
        
        if self.use_hyperparameter_tuning:
            summary_lines.extend([
                "Hyperparameter Tuning:",
                f"  Method: {self.tuning_method}",
                f"  CV Folds: {self.tuning_cv_folds}",
                f"  Scoring: {self.tuning_scoring}",
                "",
            ])
        
        summary_lines.extend([
            "Output:",
            f"  Output Directory: {self.output_dir}",
            f"  Random Seed: {self.random_seed}",
            "=" * 70,
        ])
        
        return "\n".join(summary_lines)