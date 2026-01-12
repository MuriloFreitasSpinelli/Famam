"""
Hyperparameter tuning utilities for LSTM models using scikit-learn integration.

This module provides wrappers to use sklearn's GridSearchCV, RandomizedSearchCV,
and Bayesian optimization with Keras/TensorFlow models.

Supports both simple sequential models and conditioned models for next-note prediction.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, List, Tuple, Union
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout # type: ignore
from tensorflow.keras import regularizers # type: ignore
from pathlib import Path
import sys

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# For Bayesian optimization (optional, requires scikit-optimize)
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper to make Keras models compatible with sklearn's hyperparameter search.
    
    This wrapper allows you to use GridSearchCV, RandomizedSearchCV, and other
    sklearn tools with Keras models.
    """
    
    def __init__(
        self,
        build_fn: Callable,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 0,
        validation_split: float = 0.1,
        callbacks: Optional[List] = None,
        **kwargs
    ):
        """
        Initialize the wrapper.
        
        Args:
            build_fn: Function that builds and returns a compiled Keras model
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            validation_split: Fraction of data to use for validation
            callbacks: List of Keras callbacks
            **kwargs: Additional parameters passed to build_fn
        """
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.callbacks = callbacks or []
        self.model_params = kwargs
        self.model_ = None
        self.history_ = None
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'build_fn': self.build_fn,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
            'validation_split': self.validation_split,
            'callbacks': self.callbacks,
        }
        params.update(self.model_params)
        return params
    
    def fit(self, X, y, **kwargs):
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments passed to model.fit()
        """
        # Build model with current parameters
        self.model_ = self.build_fn(**self.model_params)
        
        # Fit the model
        fit_args = {
            'x': X,
            'y': y,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'validation_split': self.validation_split,
            'callbacks': self.callbacks,
        }
        fit_args.update(kwargs)
        
        self.history_ = self.model_.fit(**fit_args)
        
        return self
    
    def predict(self, X, **kwargs):
        """Make predictions."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before making predictions")
        return self.model_.predict(X, **kwargs)
    
    def score(self, X, y, **kwargs):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        For compatibility with sklearn, this returns negative MSE.
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        
        y_pred = self.predict(X, **kwargs)
        # Return negative MSE (sklearn convention for maximization)
        mse = np.mean((y - y_pred) ** 2)
        return -mse


def create_lstm_model(
    input_shape: Tuple[int, ...],
    output_units: int,
    lstm_units: List[int] = None, # type: ignore
    dense_units: List[int] = None, # type: ignore
    dropout_rate: float = 0.2,
    recurrent_dropout: float = 0.1,
    bidirectional: bool = False,
    learning_rate: float = 0.001,
    optimizer: str = 'adam',
    loss: str = 'mse',
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
    **kwargs
) -> keras.Model:
    """
    Build and compile an LSTM model.
    
    This function is designed to work with KerasRegressorWrapper for hyperparameter tuning.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        output_units: Number of output units
        lstm_units: List of units for each LSTM layer
        dense_units: List of units for each dense layer
        dropout_rate: Dropout rate
        recurrent_dropout: Recurrent dropout rate
        bidirectional: Whether to use bidirectional LSTM
        learning_rate: Learning rate
        optimizer: Optimizer name
        loss: Loss function
        l1_reg: L1 regularization coefficient
        l2_reg: L2 regularization coefficient
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Compiled Keras model
    """
    
    # Default values
    if lstm_units is None:
        lstm_units = [128, 64]
    if dense_units is None:
        dense_units = [64, 32]
    
    # Build regularizer
    if l1_reg > 0 and l2_reg > 0:
        regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        regularizer = regularizers.l1(l1_reg)
    elif l2_reg > 0:
        regularizer = regularizers.l2(l2_reg)
    else:
        regularizer = None
    
    model = Sequential()
    
    # Add LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1) or (len(dense_units) > 0)
        
        lstm_layer = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizer,
        )
        
        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)
        
        if i == 0:
            model.add(lstm_layer)
            model.build(input_shape=(None,) + input_shape)
        else:
            model.add(lstm_layer)
        
        if dropout_rate > 0 and i < len(lstm_units) - 1:
            model.add(Dropout(dropout_rate))
    
    # Add dense layers
    for units in dense_units:
        model.add(Dense(units, activation='relu', kernel_regularizer=regularizer))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(output_units, activation='linear'))
    
    # Compile model
    opt = _get_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=['mae', 'mse'])
    
    return model


def _get_optimizer(optimizer_name: str, learning_rate: float):
    """Get optimizer instance by name."""
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'adagrad':
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'adadelta':
        return keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer_name == 'adamax':
        return keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        return keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def grid_search_hyperparameters(
    build_fn: Callable,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
    scoring: str = 'neg_mean_squared_error',
    n_jobs: int = -1,
    verbose: int = 2,
    **fit_params
) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        build_fn: Function that builds and returns a compiled Keras model
        param_grid: Dictionary with parameter names as keys and lists of values to try
        X_train: Training features
        y_train: Training targets
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level
        **fit_params: Additional parameters for model.fit()
    
    Returns:
        Tuple of (GridSearchCV object, best parameters dict)
    """
    # Create wrapper
    model = KerasRegressorWrapper(build_fn=build_fn)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )
    
    # Fit
    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train, **fit_params)
    
    print(f"\nBest Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    return grid_search, grid_search.best_params_


def random_search_hyperparameters(
    build_fn: Callable,
    param_distributions: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = 'neg_mean_squared_error',
    n_jobs: int = -1,
    verbose: int = 2,
    random_state: Optional[int] = None,
    **fit_params
) -> Tuple[RandomizedSearchCV, Dict[str, Any]]:
    """
    Perform randomized search for hyperparameter tuning.
    
    Args:
        build_fn: Function that builds and returns a compiled Keras model
        param_distributions: Dictionary with parameter names and distributions/lists
        X_train: Training features
        y_train: Training targets
        n_iter: Number of parameter settings to sample
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level
        random_state: Random seed
        **fit_params: Additional parameters for model.fit()
    
    Returns:
        Tuple of (RandomizedSearchCV object, best parameters dict)
    """
    # Create wrapper
    model = KerasRegressorWrapper(build_fn=build_fn)
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
    )
    
    # Fit
    print("Starting Randomized Search...")
    random_search.fit(X_train, y_train, **fit_params)
    
    print(f"\nBest Score: {random_search.best_score_:.4f}")
    print(f"Best Parameters: {random_search.best_params_}")
    
    return random_search, random_search.best_params_


def bayesian_search_hyperparameters(
    build_fn: Callable,
    search_spaces: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_calls: int = 50,
    n_initial_points: int = 10,
    cv: int = 3,
    scoring: str = 'neg_mean_squared_error',
    n_jobs: int = -1,
    verbose: int = 2,
    random_state: Optional[int] = None,
    **fit_params
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform Bayesian optimization for hyperparameter tuning.
    
    Requires: pip install scikit-optimize
    
    Args:
        build_fn: Function that builds and returns a compiled Keras model
        search_spaces: Dictionary with parameter names and search space definitions
                      Example: {'learning_rate': Real(1e-4, 1e-2, prior='log-uniform')}
        X_train: Training features
        y_train: Training targets
        n_calls: Number of calls to the objective function
        n_initial_points: Number of random initialization points
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level
        random_state: Random seed
        **fit_params: Additional parameters for model.fit()
    
    Returns:
        Tuple of (BayesSearchCV object, best parameters dict)
    """
    if not BAYESIAN_AVAILABLE:
        raise ImportError(
            "Bayesian optimization requires scikit-optimize. "
            "Install it with: pip install scikit-optimize"
        )
    
    # Create wrapper
    model = KerasRegressorWrapper(build_fn=build_fn)
    
    # Create BayesSearchCV
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=n_calls,
        n_points=n_initial_points,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
    )
    
    # Fit
    print("Starting Bayesian Optimization...")
    bayes_search.fit(X_train, y_train, **fit_params)
    
    print(f"\nBest Score: {bayes_search.best_score_:.4f}")  # type: ignore[attr-defined]
    print(f"Best Parameters: {bayes_search.best_params_}")  # type: ignore[attr-defined]

    return bayes_search, bayes_search.best_params_  # type: ignore[attr-defined]


def create_search_spaces_for_bayesian(param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Convert a parameter grid to Bayesian search spaces.
    
    Args:
        param_grid: Dictionary with parameter names and lists of values
    
    Returns:
        Dictionary with parameter names and search space objects
    """
    if not BAYESIAN_AVAILABLE:
        raise ImportError("This function requires scikit-optimize")
    
    search_spaces = {}
    
    for param_name, values in param_grid.items():
        if isinstance(values[0], (int, np.integer)):
            # Integer parameter
            search_spaces[param_name] = Integer(min(values), max(values))
        elif isinstance(values[0], (float, np.floating)):
            # Float parameter
            search_spaces[param_name] = Real(min(values), max(values))
        elif isinstance(values[0], str):
            # Categorical parameter
            search_spaces[param_name] = Categorical(values)
        elif isinstance(values[0], list):
            # List parameter (e.g., layer sizes)
            search_spaces[param_name] = Categorical(values)
        else:
            # Default to categorical
            search_spaces[param_name] = Categorical(values)
    
    return search_spaces


def run_hyperparameter_tuning(
    config,  # TrainingConfig object
    build_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    **fit_params
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run hyperparameter tuning based on TrainingConfig settings.
    
    Args:
        config: TrainingConfig object with tuning settings
        build_fn: Function that builds and returns a compiled Keras model
        X_train: Training features
        y_train: Training targets
        **fit_params: Additional parameters for model.fit()
    
    Returns:
        Tuple of (search object, best parameters dict)
    """
    if not config.use_hyperparameter_tuning:
        raise ValueError("Hyperparameter tuning is not enabled in config")
    
    if config.param_grid is None:
        raise ValueError("param_grid must be specified in config for hyperparameter tuning")
    
    method = config.tuning_method.lower()
    
    if method == 'grid_search':
        return grid_search_hyperparameters(
            build_fn=build_fn,
            param_grid=config.param_grid,
            X_train=X_train,
            y_train=y_train,
            cv=config.tuning_cv_folds,
            scoring=config.tuning_scoring,
            n_jobs=config.tuning_n_jobs,
            verbose=config.tuning_verbose,
            **fit_params
        )
    
    elif method == 'random_search':
        return random_search_hyperparameters(
            build_fn=build_fn,
            param_distributions=config.param_grid,
            X_train=X_train,
            y_train=y_train,
            n_iter=config.n_iter,
            cv=config.tuning_cv_folds,
            scoring=config.tuning_scoring,
            n_jobs=config.tuning_n_jobs,
            verbose=config.tuning_verbose,
            random_state=config.random_seed,
            **fit_params
        )
    
    elif method == 'bayesian':
        search_spaces = create_search_spaces_for_bayesian(config.param_grid)
        return bayesian_search_hyperparameters(
            build_fn=build_fn,
            search_spaces=search_spaces,
            X_train=X_train,
            y_train=y_train,
            n_calls=config.bayesian_n_calls,
            n_initial_points=config.bayesian_n_initial_points,
            cv=config.tuning_cv_folds,
            scoring=config.tuning_scoring,
            n_jobs=config.tuning_n_jobs,
            verbose=config.tuning_verbose,
            random_state=config.random_seed,
            **fit_params
        )
    
    else:
        raise ValueError(f"Unknown tuning method: {method}")


# =============================================================================
# Utilities for tf.data.Dataset-based Hyperparameter Tuning
# =============================================================================

def dataset_to_numpy(
    dataset: "tf.data.Dataset",
    max_samples: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Convert a tf.data.Dataset to numpy arrays for sklearn-based tuning.

    Args:
        dataset: tf.data.Dataset from generate_tensorflow_dataset
        max_samples: Maximum samples to load (None for all)

    Returns:
        Tuple of (X_dict, y) where X_dict has 'music', 'genre', 'instruments'
    """
    music_list = []
    genre_list = []
    instrument_list = []
    target_list = []

    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        if isinstance(sample, dict):
            music_list.append(sample['music'].numpy())
            target_list.append(sample['music'].numpy())  # For autoencoder-style

            if 'genre_id' in sample:
                genre_list.append(sample['genre_id'].numpy())
            else:
                genre_list.append(0)

            if 'instrument_ids' in sample:
                instrument_list.append(sample['instrument_ids'].numpy())
            else:
                instrument_list.append(np.zeros(16, dtype=np.int32))
        else:
            music_list.append(sample.numpy())
            target_list.append(sample.numpy())
            genre_list.append(0)
            instrument_list.append(np.zeros(16, dtype=np.int32))

    X = {
        'music': np.array(music_list),
        'genre': np.array(genre_list),
        'instruments': np.array(instrument_list),
    }
    y = np.array(target_list)

    return X, y


def tune_from_dataset(
    train_dataset: "tf.data.Dataset",
    config,  # TrainingConfig
    method: str = 'random_search',
    n_iter: int = 10,
    cv: int = 3,
    max_samples: int = 1000,
    verbose: int = 2,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run hyperparameter tuning using tf.data.Dataset.

    Note: For large datasets, this converts a subset to numpy for sklearn CV.
    For very large datasets, consider using Keras Tuner instead.

    Args:
        train_dataset: tf.data.Dataset from generate_tensorflow_dataset
        config: TrainingConfig with tuning settings
        method: 'grid_search', 'random_search', or 'bayesian'
        n_iter: Number of iterations for random/bayesian search
        cv: Number of cross-validation folds
        max_samples: Max samples to use (to limit memory)
        verbose: Verbosity level

    Returns:
        Tuple of (search object, best parameters)
    """
    print(f"Converting dataset to numpy (max {max_samples} samples)...")
    X, y = dataset_to_numpy(train_dataset, max_samples)

    input_shape = X['music'].shape[1:]
    output_shape = y.shape[1:]

    # Flatten output for regression
    y_flat = y.reshape(len(y), -1)

    # Create wrapper
    wrapper = KerasRegressorWrapper(
        build_fn=lambda **kwargs: _build_tuning_model(
            input_shape=input_shape,
            output_size=y_flat.shape[1],
            **kwargs
        ),
        epochs=min(config.epochs, 10),  # Use fewer epochs for tuning
        verbose=0,
    )

    # Parameter grid
    param_grid = config.param_grid or {
        'lstm_units': [[64], [128], [128, 64]],
        'dense_units': [[32], [64]],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.0001, 0.001, 0.01],
    }

    if method == 'grid_search':
        search = GridSearchCV(
            estimator=wrapper,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=verbose,
            return_train_score=True,
        )
    elif method == 'random_search':
        search = RandomizedSearchCV(
            estimator=wrapper,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=verbose,
            random_state=config.random_seed,
            return_train_score=True,
        )
    elif method == 'bayesian':
        if not BAYESIAN_AVAILABLE:
            raise ImportError("Bayesian optimization requires scikit-optimize")

        search_spaces = create_search_spaces_for_bayesian(param_grid)
        search = BayesSearchCV(
            estimator=wrapper,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=verbose,
            random_state=config.random_seed,
            return_train_score=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Use just the music data for simple tuning
    print(f"Starting {method} hyperparameter tuning...")
    search.fit(X['music'], y_flat)

    print(f"\nBest Score: {search.best_score_:.4f}") # type: ignore
    print(f"Best Parameters: {search.best_params_}") # type: ignore

    return search, search.best_params_ # type: ignore


def _build_tuning_model(
    input_shape: Tuple,
    output_size: int,
    lstm_units: List[int] = None, # type: ignore
    dense_units: List[int] = None, # type: ignore
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    **kwargs
) -> tf.keras.Model: # type: ignore
    """Build a simple LSTM model for hyperparameter tuning."""
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten # type: ignore

    lstm_units = lstm_units if isinstance(lstm_units, list) else [lstm_units or 128]
    dense_units = dense_units if isinstance(dense_units, list) else [dense_units or 64]

    inputs = Input(shape=input_shape)
    x = inputs

    # Handle shape for LSTM
    if len(input_shape) == 1:
        x = tf.keras.layers.Reshape((1, input_shape[0]))(x) # type: ignore

    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        x = LSTM(units, return_sequences=return_sequences, dropout=dropout_rate)(x)

    for units in dense_units:
        x = Dense(units, activation='relu')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    outputs = Dense(output_size, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs) # type: ignore
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), # type: ignore
        loss='mse',
        metrics=['mae']
    )

    return model