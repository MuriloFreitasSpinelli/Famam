from .configs import ModelTrainingConfig
from .configs.slurm_config import SlurmConfig
from .configs.transformer_config import (
    TransformerTrainingConfig,
    get_small_config,
    get_medium_config,
    get_large_config,
)
from .model_trainer import (
    ModelTrainer,
    build_lstm_model,
    train_from_music_dataset,
)
from .transformer_model import (
    MusicTransformer,
    build_transformer_model,
    build_transformer_from_config,
    TransformerBlock,
    TokenAndPositionEmbedding,
)
from .transformer_trainer import (
    TransformerTrainer,
    TransformerLRSchedule,
    MaskedSparseCategoricalCrossentropy,
    MaskedAccuracy,
    train_transformer_from_music_dataset,
)
from .distribution_strategy import (
    DistributionStrategyFactory,
    setup_tf_config,
    setup_tf_config_from_slurm,
    get_strategy_info,
    calculate_global_batch_size,
)
from .slurm_generator import (
    SlurmScriptGenerator,
    generate_quick_script,
)

__all__ = [
    # LSTM Config
    'ModelTrainingConfig',
    'SlurmConfig',
    # Transformer Config
    'TransformerTrainingConfig',
    'get_small_config',
    'get_medium_config',
    'get_large_config',
    # LSTM Training
    'ModelTrainer',
    'build_lstm_model',
    'train_from_music_dataset',
    # Transformer Model
    'MusicTransformer',
    'build_transformer_model',
    'build_transformer_from_config',
    'TransformerBlock',
    'TokenAndPositionEmbedding',
    # Transformer Training
    'TransformerTrainer',
    'TransformerLRSchedule',
    'MaskedSparseCategoricalCrossentropy',
    'MaskedAccuracy',
    'train_transformer_from_music_dataset',
    # Distribution
    'DistributionStrategyFactory',
    'setup_tf_config',
    'setup_tf_config_from_slurm',
    'get_strategy_info',
    'calculate_global_batch_size',
    # SLURM
    'SlurmScriptGenerator',
    'generate_quick_script',
]
