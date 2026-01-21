"""
Hyperparameter tuning utilities for LSTM models using scikit-learn integration.

Provides wrappers to use sklearn's GridSearchCV, RandomizedSearchCV,
and Bayesian optimization with Keras/TensorFlow LSTM models.

Works with MusicDataset tensors:
    - pianoroll: (128, max_time_steps)
    - genre_id: int32 scalar
"""

import numpy as np  # type: ignore
from typing import Dict, Any, Callable, Optional, List, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore

import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Input, LSTM, Bidirectional, Dense, Dropout, Embedding, Concatenate, Reshape
)
from tensorflow.keras import regularizers  # type: ignore

from .configs import ModelTuningConfig

# For Bayesian optimization (optional)
try:
    from skopt import BayesSearchCV  # type: ignore
    from skopt.space import Real, Integer, Categorical  # type: ignore
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper to make Keras models compatible with sklearn's hyperparameter search.

    Note: sklearn requires 2D arrays (samples x features), but Keras models may expect
    3D+ input. Use input_shape and output_shape to automatically reshape data.
    """

    def __init__(
        self,
        build_fn: Callable,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 0,
        validation_split: float = 0.1,
        callbacks: Optional[List] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        **kwargs
    ):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.callbacks = callbacks
        self.input_shape = input_shape
        self.output_shape = output_shape
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
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
        }
        params.update(self.model_params)
        return params

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape flattened sklearn input back to Keras expected shape."""
        if self.input_shape is not None:
            return X.reshape((X.shape[0],) + self.input_shape)
        return X

    def _reshape_output(self, y: np.ndarray) -> np.ndarray:
        """Reshape flattened sklearn target back to Keras expected shape."""
        if self.output_shape is not None:
            return y.reshape((y.shape[0],) + self.output_shape)
        return y

    def _flatten_output(self, y: np.ndarray) -> np.ndarray:
        """Flatten Keras output back to sklearn 2D format."""
        if len(y.shape) > 2:
            return y.reshape((y.shape[0], -1))
        return y

    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.model_ = self.build_fn(**self.model_params)

        # Reshape from sklearn 2D to Keras expected shape
        X_reshaped = self._reshape_input(X)
        y_reshaped = self._reshape_output(y)

        fit_args = {
            'x': X_reshaped,
            'y': y_reshaped,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'validation_split': self.validation_split,
            'callbacks': self.callbacks or [],
        }
        fit_args.update(kwargs)

        self.history_ = self.model_.fit(**fit_args)
        return self

    def predict(self, X, **kwargs):
        """Make predictions."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before making predictions")
        X_reshaped = self._reshape_input(X)
        y_pred = self.model_.predict(X_reshaped, **kwargs)
        return self._flatten_output(y_pred)

    def score(self, X, y, **kwargs):
        """Return negative MSE (sklearn convention for maximization)."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before scoring")

        y_pred = self.predict(X, **kwargs)
        # y is already flat from sklearn, y_pred is flattened by predict
        mse = np.mean((y - y_pred) ** 2)
        return -mse


def create_lstm_model(
    input_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    lstm_units: Optional[List[int]] = None,
    dense_units: Optional[List[int]] = None,
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
    Build and compile an LSTM model for pianoroll prediction.

    Args:
        input_shape: (num_pitches, time_steps) e.g. (128, 1000)
        output_shape: Same as input_shape for autoencoder-style
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

    Returns:
        Compiled Keras model
    """
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

    # Input: pianoroll (128, time_steps)
    inputs = Input(shape=input_shape, name='pianoroll_input')

    # Transpose for LSTM: (time_steps, 128) so LSTM processes time dimension
    x = tf.keras.layers.Permute((2, 1))(inputs)

    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)

        lstm_layer = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizer,
        )

        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)

        x = lstm_layer(x)

        if dropout_rate > 0 and i < len(lstm_units) - 1:
            x = Dropout(dropout_rate)(x)

    # Dense layers
    for units in dense_units:
        x = Dense(units, activation='relu', kernel_regularizer=regularizer)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Output layer - flatten to match pianoroll output
    output_size = output_shape[0] * output_shape[1]
    x = Dense(output_size, activation='sigmoid')(x)
    outputs = Reshape(output_shape)(x)

    model = Model(inputs, outputs)

    # Compile
    opt = _get_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    return model


def create_genre_conditioned_lstm_model(
    input_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    num_genres: int,
    genre_embedding_dim: int = 16,
    lstm_units: Optional[List[int]] = None,
    dense_units: Optional[List[int]] = None,
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
    Build LSTM model with genre conditioning.

    Args:
        input_shape: (num_pitches, time_steps) e.g. (128, 1000)
        output_shape: Same as input_shape
        num_genres: Number of genres in vocabulary
        genre_embedding_dim: Dimension of genre embedding
        lstm_units: List of units for each LSTM layer
        dense_units: List of units for each dense layer
        ... (same as create_lstm_model)

    Returns:
        Compiled Keras model with pianoroll and genre inputs
    """
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

    # Inputs
    pianoroll_input = Input(shape=input_shape, name='pianoroll_input')
    genre_input = Input(shape=(1,), dtype='int32', name='genre_input')

    # Genre embedding
    genre_embedded = Embedding(
        input_dim=num_genres,
        output_dim=genre_embedding_dim,
        name='genre_embedding'
    )(genre_input)
    genre_embedded = tf.keras.layers.Flatten()(genre_embedded)

    # Transpose pianoroll for LSTM: (time_steps, 128)
    x = tf.keras.layers.Permute((2, 1))(pianoroll_input)
    time_steps = input_shape[1]

    # Tile genre embedding across time steps and concatenate
    genre_tiled = tf.keras.layers.RepeatVector(time_steps)(genre_embedded)
    x = Concatenate()([x, genre_tiled])

    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)

        lstm_layer = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizer,
        )

        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)

        x = lstm_layer(x)

        if dropout_rate > 0 and i < len(lstm_units) - 1:
            x = Dropout(dropout_rate)(x)

    # Dense layers
    for units in dense_units:
        x = Dense(units, activation='relu', kernel_regularizer=regularizer)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Output layer
    output_size = output_shape[0] * output_shape[1]
    x = Dense(output_size, activation='sigmoid')(x)
    outputs = Reshape(output_shape, name='pianoroll_output')(x)

    model = Model([pianoroll_input, genre_input], outputs)

    # Compile
    opt = _get_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    return model


def _get_optimizer(optimizer_name: str, learning_rate: float):
    """Get optimizer instance by name."""
    optimizer_name = optimizer_name.lower()

    optimizers = {
        'adam': keras.optimizers.Adam,
        'sgd': keras.optimizers.SGD,
        'rmsprop': keras.optimizers.RMSprop,
        'adagrad': keras.optimizers.Adagrad,
        'adadelta': keras.optimizers.Adadelta,
        'adamax': keras.optimizers.Adamax,
        'nadam': keras.optimizers.Nadam,
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizers[optimizer_name](learning_rate=learning_rate)


def grid_search_hyperparameters(
    build_fn: Callable,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
    scoring: str = 'neg_mean_squared_error',
    n_jobs: int = 1,
    verbose: int = 2,
    **fit_params
) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Perform grid search for hyperparameter tuning.

    Args:
        build_fn: Function that builds and returns a compiled Keras model
        param_grid: Dictionary with parameter names and lists of values
        X_train: Training features (pianoroll arrays)
        y_train: Training targets
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbosity level

    Returns:
        Tuple of (GridSearchCV object, best parameters dict)
    """
    model = KerasRegressorWrapper(build_fn=build_fn)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

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
    n_jobs: int = 1,
    verbose: int = 2,
    random_state: Optional[int] = None,
    **fit_params
) -> Tuple[RandomizedSearchCV, Dict[str, Any]]:
    """
    Perform randomized search for hyperparameter tuning.

    Args:
        build_fn: Function that builds and returns a compiled Keras model
        param_distributions: Dictionary with parameter distributions
        X_train: Training features
        y_train: Training targets
        n_iter: Number of parameter settings to sample
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        random_state: Random seed

    Returns:
        Tuple of (RandomizedSearchCV object, best parameters dict)
    """
    model = KerasRegressorWrapper(build_fn=build_fn)

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
    n_jobs: int = 1,
    verbose: int = 2,
    random_state: Optional[int] = None,
    **fit_params
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform Bayesian optimization for hyperparameter tuning.

    Requires: pip install scikit-optimize

    Args:
        build_fn: Function that builds and returns a compiled Keras model
        search_spaces: Dictionary with search space definitions
        X_train: Training features
        y_train: Training targets
        n_calls: Number of calls to the objective function
        n_initial_points: Number of random initialization points
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        random_state: Random seed

    Returns:
        Tuple of (BayesSearchCV object, best parameters dict)
    """
    if not BAYESIAN_AVAILABLE:
        raise ImportError(
            "Bayesian optimization requires scikit-optimize. "
            "Install with: pip install scikit-optimize"
        )

    model = KerasRegressorWrapper(build_fn=build_fn)

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

    print("Starting Bayesian Optimization...")
    bayes_search.fit(X_train, y_train, **fit_params)

    print(f"\nBest Score: {bayes_search.best_score_:.4f}")  # type: ignore
    print(f"Best Parameters: {bayes_search.best_params_}")  # type: ignore

    return bayes_search, bayes_search.best_params_  # type: ignore


def create_search_spaces_for_bayesian(param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Convert a parameter grid to Bayesian search spaces.

    Args:
        param_grid: Dictionary with parameter names and lists of values

    Returns:
        Dictionary with search space objects
    """
    if not BAYESIAN_AVAILABLE:
        raise ImportError("This function requires scikit-optimize")

    search_spaces = {}

    for param_name, values in param_grid.items():
        if isinstance(values[0], (int, np.integer)):
            search_spaces[param_name] = Integer(min(values), max(values))
        elif isinstance(values[0], (float, np.floating)):
            search_spaces[param_name] = Real(min(values), max(values))
        elif isinstance(values[0], (str, list)):
            search_spaces[param_name] = Categorical(values)
        else:
            search_spaces[param_name] = Categorical(values)

    return search_spaces


def run_hyperparameter_tuning(
    config: ModelTuningConfig,
    build_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    **fit_params
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run hyperparameter tuning based on ModelTuningConfig settings.

    Args:
        config: ModelTuningConfig object with tuning settings
        build_fn: Function that builds and returns a compiled Keras model
        X_train: Training features (pianoroll arrays)
        y_train: Training targets

    Returns:
        Tuple of (search object, best parameters dict)
    """
    if not config.use_hyperparameter_tuning:
        raise ValueError("Hyperparameter tuning is not enabled in config")

    if config.param_grid is None:
        raise ValueError("param_grid must be specified in config")

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
            random_state=config.cv_random_state,
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
            random_state=config.cv_random_state,
            **fit_params
        )

    else:
        raise ValueError(f"Unknown tuning method: {method}")


# =============================================================================
# Utilities for MusicDataset-based Hyperparameter Tuning
# =============================================================================

def dataset_to_numpy(
    dataset: "tf.data.Dataset",
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a MusicDataset's tf.data.Dataset to numpy arrays for sklearn tuning.

    Args:
        dataset: tf.data.Dataset from MusicDataset.to_tensorflow_dataset()
        max_samples: Maximum samples to load (None for all)

    Returns:
        Tuple of (pianorolls, genre_ids, pianorolls) - X, genre, y
    """
    pianoroll_list = []
    genre_list = []

    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        pianoroll_list.append(sample['pianoroll'].numpy())
        genre_list.append(sample['genre_id'].numpy())

    pianorolls = np.array(pianoroll_list)
    genre_ids = np.array(genre_list)

    # For autoencoder-style training, target is same as input
    return pianorolls, genre_ids, pianorolls


def tune_from_music_dataset(
    train_dataset: "tf.data.Dataset",
    config: ModelTuningConfig,
    num_genres: int,
    input_shape: Tuple[int, int] = (128, 1000),
    max_samples: int = 1000,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run hyperparameter tuning using MusicDataset.

    Note: Converts a subset to numpy for sklearn cross-validation.

    Args:
        train_dataset: tf.data.Dataset from MusicDataset.to_tensorflow_dataset()
        config: ModelTuningConfig with tuning settings
        num_genres: Number of genres in vocabulary
        input_shape: (num_pitches, max_time_steps)
        max_samples: Max samples to use (limits memory)

    Returns:
        Tuple of (search object, best parameters)
    """
    print(f"Converting dataset to numpy (max {max_samples} samples)...")
    pianorolls, genre_ids, targets = dataset_to_numpy(train_dataset, max_samples)

    print(f"  Loaded {len(pianorolls)} samples")
    print(f"  Pianoroll shape: {pianorolls.shape}")
    print(f"  Unique genres: {len(np.unique(genre_ids))}")

    # Flatten for sklearn
    X_flat = pianorolls.reshape(len(pianorolls), -1)
    y_flat = targets.reshape(len(targets), -1)

    # Create build function
    def build_fn(
        lstm_units=None,
        dense_units=None,
        dropout_rate=0.2,
        learning_rate=0.001,
        **kwargs
    ):
        return create_lstm_model(
            input_shape=input_shape,
            output_shape=input_shape,
            lstm_units=lstm_units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            **kwargs
        )

    # Create wrapper with reduced epochs for tuning
    # Pass input_shape and output_shape so the wrapper can reshape sklearn's 2D data
    wrapper = KerasRegressorWrapper(
        build_fn=build_fn,
        epochs=10,  # Fewer epochs for tuning
        verbose=0,
        input_shape=input_shape,
        output_shape=input_shape,  # autoencoder-style: output same as input
    )

    # Parameter grid
    param_grid = config.param_grid or {
        'lstm_units': [[64], [128], [128, 64]],
        'dense_units': [[32], [64]],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.0001, 0.001, 0.01],
    }

    method = config.tuning_method.lower()

    if method == 'grid_search':
        search = GridSearchCV(
            estimator=wrapper,
            param_grid=param_grid,
            cv=config.tuning_cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=config.tuning_verbose,
            return_train_score=True,
        )
    elif method == 'random_search':
        search = RandomizedSearchCV(
            estimator=wrapper,
            param_distributions=param_grid,
            n_iter=config.n_iter,
            cv=config.tuning_cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=config.tuning_verbose,
            random_state=config.cv_random_state,
            return_train_score=True,
        )
    elif method == 'bayesian':
        if not BAYESIAN_AVAILABLE:
            raise ImportError("Bayesian optimization requires scikit-optimize")

        search_spaces = create_search_spaces_for_bayesian(param_grid)
        search = BayesSearchCV(
            estimator=wrapper,
            search_spaces=search_spaces,
            n_iter=config.bayesian_n_calls,
            cv=config.tuning_cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=config.tuning_verbose,
            random_state=config.cv_random_state,
            return_train_score=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Starting {method} hyperparameter tuning...")
    search.fit(X_flat, y_flat)

    print(f"\nBest Score: {search.best_score_:.4f}")  # type: ignore
    print(f"Best Parameters: {search.best_params_}")  # type: ignore

    return search, search.best_params_  # type: ignore
