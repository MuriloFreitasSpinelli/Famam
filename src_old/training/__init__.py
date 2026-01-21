"""
Training module for conditioned music generation.

Works directly with tf.data.Dataset from generate_tensorflow_dataset.py.
Supports various muspy representations (pitch, piano-roll, event, note)
and conditioning on genre/instruments.
"""

from .trainer import (
    Trainer,
    build_model,
    train_from_datasets,
    detect_dataset_info,
    ConditioningType,
    DatasetInfo,
)

from .saved_model import SavedModel, ModelMetadata

from .hyperparameter_tuning import (
    KerasRegressorWrapper,
    grid_search_hyperparameters,
    random_search_hyperparameters,
    bayesian_search_hyperparameters,
    run_hyperparameter_tuning,
    tune_from_dataset,
    dataset_to_numpy,
    BAYESIAN_AVAILABLE,
)

from .configs.training_config import TrainingConfig

__all__ = [
    # Trainer
    'Trainer',
    'build_model',
    'train_from_datasets',
    'detect_dataset_info',
    'ConditioningType',
    'DatasetInfo',
    'TrainingConfig',
    # SavedModel
    'SavedModel',
    'ModelMetadata',
    # Hyperparameter tuning
    'KerasRegressorWrapper',
    'grid_search_hyperparameters',
    'random_search_hyperparameters',
    'bayesian_search_hyperparameters',
    'run_hyperparameter_tuning',
    'tune_from_dataset',
    'dataset_to_numpy',
    'BAYESIAN_AVAILABLE',
]
