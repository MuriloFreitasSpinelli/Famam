"""
Hyperparameter tuning utilities for LSTM models using scikit-learn integration.

Provides wrappers to use sklearn's GridSearchCV, RandomizedSearchCV,
and Bayesian optimization with Keras/TensorFlow LSTM models.

Works with MusicDataset tensors:
    - pianoroll: (128, max_time_steps)
    - instrument_id: int32 scalar (0-127, or 128 for drums)
    - genre_id: int32 scalar
    - drum_pianoroll: (128, max_time_steps) for alignment learning
"""

import gc
import numpy as np  # type: ignore
from typing import Dict, Any, Callable, Optional, List, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore

import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Input, LSTM, Bidirectional, Dense, Dropout, Embedding, Concatenate, Reshape,
    TimeDistributed, Permute
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

    For multi-input models, use multi_input_config to specify how to split the flattened
    X array back into multiple inputs. Each entry maps input name to (shape, dtype).
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
        multi_input_config: Optional[Dict[str, Tuple[Tuple[int, ...], str]]] = None,
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
        self.multi_input_config = multi_input_config
        self.model_params = dict(kwargs)  # Make a copy to avoid mutation issues
        self.model_ = None
        self.history_ = None

    def set_params(self, **params):
        """Set parameters for this estimator."""
        # Known fixed parameters
        fixed_params = {
            'build_fn', 'epochs', 'batch_size', 'verbose', 'validation_split',
            'callbacks', 'input_shape', 'output_shape', 'multi_input_config'
        }
        for key, value in params.items():
            if key in fixed_params:
                setattr(self, key, value)
            else:
                # Store hyperparameters in model_params dict
                self.model_params[key] = value
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
            'multi_input_config': self.multi_input_config,
        }
        params.update(self.model_params)
        return params

    def _convert_params_for_build(self, params: Dict) -> Dict:
        """
        Convert parameter values for build_fn compatibility.

        Handles:
        - Tuples -> lists (from older tuple-based Categorical)
        - String-encoded lists -> actual lists (from string-based Categorical)
        """
        converted = {}
        for key, value in params.items():
            if isinstance(value, tuple):
                converted[key] = list(value)
            elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                # String-encoded list from Bayesian search
                import ast
                try:
                    converted[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    converted[key] = value
            else:
                converted[key] = value
        return converted

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape flattened sklearn input back to Keras expected shape."""
        if self.input_shape is not None:
            return X.reshape((X.shape[0],) + self.input_shape)
        return X

    def _reshape_multi_input(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Split flattened X into multiple inputs for multi-input Keras models.

        Returns a list of arrays in the order specified by multi_input_config.
        """
        if self.multi_input_config is None:
            raise ValueError("multi_input_config must be set for multi-input models")

        n_samples = X.shape[0]
        inputs = []
        offset = 0

        for input_name, (shape, dtype) in self.multi_input_config.items():
            # Calculate the flat size for this input
            flat_size = int(np.prod(shape))
            # Extract the slice for this input
            flat_input = X[:, offset:offset + flat_size]
            # Reshape to the expected shape
            reshaped = flat_input.reshape((n_samples,) + shape).astype(dtype)
            inputs.append(reshaped)
            offset += flat_size

        return inputs

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
        # Clear GPU memory from previous model to prevent OOM during hyperparameter search
        if self.model_ is not None:
            del self.model_
            self.model_ = None
        tf.keras.backend.clear_session()
        gc.collect()

        # Convert params for build_fn (handles tuples and string-encoded lists from skopt)
        build_params = self._convert_params_for_build(self.model_params)
        self.model_ = self.build_fn(**build_params)

        # Reshape from sklearn 2D to Keras expected shape
        if self.multi_input_config is not None:
            X_reshaped = self._reshape_multi_input(X)
        else:
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
        if self.multi_input_config is not None:
            X_reshaped = self._reshape_multi_input(X)
        else:
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
    # Ensure lstm_units and dense_units are lists (handles int/tuple from hyperparameter search)
    lstm_units = _ensure_list(lstm_units, [128, 64])
    dense_units = _ensure_list(dense_units, [64, 32])

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

    # LSTM layers - all return sequences for seq-to-seq
    for i, units in enumerate(lstm_units):
        lstm_layer = LSTM(
            units,
            return_sequences=True,  # Always return sequences
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizer,
        )

        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)

        x = lstm_layer(x)

        if dropout_rate > 0 and i < len(lstm_units) - 1:
            x = Dropout(dropout_rate)(x)

    # Dense layers (TimeDistributed to process each timestep)
    for units in dense_units:
        x = TimeDistributed(Dense(units, activation='relu', kernel_regularizer=regularizer))(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Output layer - predict pitches per timestep
    x = TimeDistributed(Dense(output_shape[0], activation='sigmoid'))(x)
    # Permute back to (pitches, time_steps)
    outputs = Permute((2, 1))(x)

    model = Model(inputs, outputs)

    # Compile
    opt = _get_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    return model


def _ensure_list(value, default):
    """Ensure value is a list, converting from int/tuple if needed."""
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return default


def create_genre_conditioned_lstm_model(
    input_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    num_genres: int,
    genre_embedding_dim: int = 16,
    num_instruments: int = 129,
    instrument_embedding_dim: int = 16,
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
    Build LSTM model with genre and instrument conditioning.

    Args:
        input_shape: (num_pitches, time_steps) e.g. (128, 1000)
        output_shape: Same as input_shape
        num_genres: Number of genres in vocabulary
        genre_embedding_dim: Dimension of genre embedding
        num_instruments: Number of instruments (default 129: 0-127 + drums)
        instrument_embedding_dim: Dimension of instrument embedding
        lstm_units: List of units for each LSTM layer
        dense_units: List of units for each dense layer
        ... (same as create_lstm_model)

    Returns:
        Compiled Keras model with pianoroll, genre, instrument, and drum inputs
    """
    # Ensure lstm_units and dense_units are lists (handles int/tuple from hyperparameter search)
    lstm_units = _ensure_list(lstm_units, [128, 64])
    dense_units = _ensure_list(dense_units, [64, 32])

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
    instrument_input = Input(shape=(1,), dtype='int32', name='instrument_input')
    drum_input = Input(shape=input_shape, name='drum_input')

    # Genre embedding
    genre_embedded = Embedding(
        input_dim=num_genres + 1,
        output_dim=genre_embedding_dim,
        name='genre_embedding'
    )(genre_input)
    genre_embedded = tf.keras.layers.Flatten()(genre_embedded)

    # Instrument embedding
    instrument_embedded = Embedding(
        input_dim=num_instruments + 1,
        output_dim=instrument_embedding_dim,
        name='instrument_embedding'
    )(instrument_input)
    instrument_embedded = tf.keras.layers.Flatten()(instrument_embedded)

    # Combine conditioning
    conditioning = Concatenate(name='conditioning')([genre_embedded, instrument_embedded])

    # Transpose pianoroll for LSTM: (time_steps, 128)
    x = tf.keras.layers.Permute((2, 1))(pianoroll_input)
    drum_x = tf.keras.layers.Permute((2, 1))(drum_input)
    time_steps = input_shape[1]

    # Concatenate with drum track for alignment
    x = Concatenate()([x, drum_x])

    # Tile conditioning across time steps and concatenate
    conditioning_tiled = tf.keras.layers.RepeatVector(time_steps)(conditioning)
    x = Concatenate()([x, conditioning_tiled])

    # LSTM layers - all return sequences for seq-to-seq
    for i, units in enumerate(lstm_units):
        lstm_layer = LSTM(
            units,
            return_sequences=True,  # Always return sequences
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizer,
        )

        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)

        x = lstm_layer(x)

        if dropout_rate > 0 and i < len(lstm_units) - 1:
            x = Dropout(dropout_rate)(x)

    # Dense layers (TimeDistributed to process each timestep)
    for units in dense_units:
        x = TimeDistributed(Dense(units, activation='relu', kernel_regularizer=regularizer))(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Output layer - predict pitches per timestep
    x = TimeDistributed(Dense(output_shape[0], activation='sigmoid'))(x)
    # Permute back to (pitches, time_steps)
    outputs = Permute((2, 1), name='pianoroll_output')(x)

    model = Model([pianoroll_input, genre_input, instrument_input, drum_input], outputs)

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

    For discrete options (small lists of specific values), uses Categorical.
    For list-valued parameters (like lstm_units=[[64], [128, 64]]), uses
    string encoding to avoid skopt's issues with tuple-valued Categorical spaces.

    Args:
        param_grid: Dictionary with parameter names and lists of values

    Returns:
        Dictionary with search space objects
    """
    if not BAYESIAN_AVAILABLE:
        raise ImportError("This function requires scikit-optimize")

    search_spaces = {}

    for param_name, values in param_grid.items():
        first_val = values[0]

        # Check if this is a list of lists (like lstm_units configurations)
        if isinstance(first_val, list):
            # Use string encoding: "[64]", "[128, 64]", etc.
            # This avoids skopt's issues with tuple-valued Categorical spaces
            str_values = [str(v) for v in values]
            search_spaces[param_name] = Categorical(str_values)

        # For numeric types with few discrete options, use Categorical (not Real/Integer ranges)
        # This preserves the exact values you specified instead of sampling continuously
        elif isinstance(first_val, (int, np.integer, float, np.floating)):
            # Use Categorical for discrete options (treats them as exact choices)
            search_spaces[param_name] = Categorical(values)

        elif isinstance(first_val, str):
            search_spaces[param_name] = Categorical(values)

        else:
            search_spaces[param_name] = Categorical(values)

    return search_spaces


def _decode_bayesian_param(value: Any) -> Any:
    """Decode a parameter value that may have been string-encoded for Bayesian search."""
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        # This looks like a string-encoded list, parse it
        import ast
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


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

def configure_gpu_memory_growth():
    """
    Configure TensorFlow to use GPU memory growth instead of allocating all memory upfront.
    This prevents OOM errors during hyperparameter tuning where multiple models are created.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"Could not set memory growth: {e}")


def dataset_to_numpy(
    dataset: "tf.data.Dataset",
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a MusicDataset's tf.data.Dataset to numpy arrays for sklearn tuning.

    Args:
        dataset: tf.data.Dataset from MusicDataset.to_tensorflow_dataset()
        max_samples: Maximum samples to load (None for all)

    Returns:
        Tuple of (pianorolls, genre_ids, instrument_ids, drum_pianorolls, pianorolls)
    """
    import sys

    pianoroll_list = []
    genre_list = []
    instrument_list = []
    drum_list = []

    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            print(f"  Processing sample {i + 1}/{max_samples if max_samples else '?'}...")
            sys.stdout.flush()

        pianoroll_list.append(sample['pianoroll'].numpy())
        genre_list.append(sample['genre_id'].numpy())
        instrument_list.append(sample['instrument_id'].numpy())

        # Get drum pianoroll if available
        if 'drum_pianoroll' in sample:
            drum_list.append(sample['drum_pianoroll'].numpy())
        else:
            drum_list.append(np.zeros_like(sample['pianoroll'].numpy()))

    pianorolls = np.array(pianoroll_list)
    genre_ids = np.array(genre_list)
    instrument_ids = np.array(instrument_list)
    drum_pianorolls = np.array(drum_list)

    # For autoencoder-style training, target is same as input
    return pianorolls, genre_ids, instrument_ids, drum_pianorolls, pianorolls


def tune_from_music_dataset(
    train_dataset: "tf.data.Dataset",
    config: ModelTuningConfig,
    num_genres: int,
    input_shape: Tuple[int, int] = (128, 1000),
    max_samples: int = 1000,
    num_instruments: int = 129,
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
        num_instruments: Number of instruments (default 129)

    Returns:
        Tuple of (search object, best parameters)
    """
    # Configure GPU memory growth to prevent OOM during tuning
    configure_gpu_memory_growth()

    import sys
    print(f"Converting dataset to numpy (max {max_samples} samples)...")
    sys.stdout.flush()
    pianorolls, genre_ids, instrument_ids, drum_pianorolls, targets = dataset_to_numpy(train_dataset, max_samples)

    print(f"  Loaded {len(pianorolls)} samples")
    print(f"  Pianoroll shape: {pianorolls.shape}")
    print(f"  Unique genres: {len(np.unique(genre_ids))}")
    print(f"  Unique instruments: {len(np.unique(instrument_ids))}")
    sys.stdout.flush()

    # Flatten all inputs for sklearn (multi-input model requires concatenation)
    n_samples = len(pianorolls)
    pianoroll_flat = pianorolls.reshape(n_samples, -1).astype(np.float32)
    genre_flat = genre_ids.reshape(n_samples, -1).astype(np.float32)
    instrument_flat = instrument_ids.reshape(n_samples, -1).astype(np.float32)
    drum_flat = drum_pianorolls.reshape(n_samples, -1).astype(np.float32)

    # Concatenate all inputs into single X array for sklearn
    X_flat = np.concatenate([pianoroll_flat, genre_flat, instrument_flat, drum_flat], axis=1)
    y_flat = targets.reshape(n_samples, -1)

    print(f"  Combined X shape: {X_flat.shape}")

    # Configure multi-input splitting for the wrapper
    # Order must match the model's input order: pianoroll, genre, instrument, drum
    multi_input_config = {
        'pianoroll_input': (input_shape, 'float32'),
        'genre_input': ((1,), 'int32'),
        'instrument_input': ((1,), 'int32'),
        'drum_input': (input_shape, 'float32'),
    }

    # Create build function
    def build_fn(
        lstm_units=None,
        dense_units=None,
        dropout_rate=0.2,
        learning_rate=0.001,
        genre_embedding_dim=16,
        instrument_embedding_dim=16,
        **kwargs
    ):
        return create_genre_conditioned_lstm_model(
            input_shape=input_shape,
            output_shape=input_shape,
            num_genres=num_genres,
            genre_embedding_dim=genre_embedding_dim,
            num_instruments=num_instruments,
            instrument_embedding_dim=instrument_embedding_dim,
            lstm_units=lstm_units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            **kwargs
        )

    # Create wrapper with reduced epochs for tuning
    # Use multi_input_config for the genre-conditioned model with 4 inputs
    wrapper = KerasRegressorWrapper(
        build_fn=build_fn,
        epochs=10,  # Fewer epochs for tuning
        verbose=0,
        output_shape=input_shape,  # autoencoder-style: output same as input pianoroll
        multi_input_config=multi_input_config,
    )

    # Parameter grid with instrument embedding options
    param_grid = config.param_grid or {
        'lstm_units': [[64], [128], [128, 64]],
        'dense_units': [[32], [64]],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.0001, 0.001, 0.01],
        'genre_embedding_dim': [8, 16, 32],
        'instrument_embedding_dim': [8, 16, 32],
    }

    method = config.tuning_method.lower()

    # Use n_jobs from config (1 is safe for GPU training, -1 uses all CPUs)
    n_jobs = config.tuning_n_jobs

    if method == 'grid_search':
        search = GridSearchCV(
            estimator=wrapper,
            param_grid=param_grid,
            cv=config.tuning_cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
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
            n_jobs=n_jobs,
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
            n_jobs=n_jobs,
            verbose=config.tuning_verbose,
            random_state=config.cv_random_state,
            return_train_score=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Starting {method} hyperparameter tuning...")
    sys.stdout.flush()
    search.fit(X_flat, y_flat)

    print(f"\nBest Score: {search.best_score_:.4f}")  # type: ignore
    print(f"Best Parameters: {search.best_params_}")  # type: ignore
    sys.stdout.flush()

    # Convert best_params: decode string-encoded lists and convert tuples to lists
    best_params = {}
    for k, v in search.best_params_.items():  # type: ignore
        if isinstance(v, tuple):
            best_params[k] = list(v)
        elif isinstance(v, str) and v.startswith('[') and v.endswith(']'):
            # String-encoded list from Bayesian search
            import ast
            try:
                best_params[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                best_params[k] = v
        else:
            best_params[k] = v

    return search, best_params