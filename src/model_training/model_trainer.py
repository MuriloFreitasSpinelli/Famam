"""
Model trainer for genre-conditioned LSTM music generation.

Consumes tf.data.Dataset from MusicDataset.to_tensorflow_dataset().
Builds LSTM model with genre conditioning via embedding.

Input tensor structure:
    - pianoroll: (128, max_time_steps)
    - genre_id: int32 scalar
"""

import json
import numpy as np  # type: ignore
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Input, LSTM, Bidirectional, Dense, Dropout, Embedding,
    Concatenate, Flatten, Reshape, TimeDistributed, Permute
)
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras import regularizers  # type: ignore

from .configs import ModelTrainingConfig
from .distribution_strategy import (
    DistributionStrategyFactory,
    calculate_global_batch_size,
    get_strategy_info,
)
from ..core.vocabulary import Vocabulary


def configure_gpu_memory():
    """
    Configure GPU memory growth to use all available VRAM.

    By default TensorFlow allocates a conservative amount of GPU memory.
    This enables memory growth so TensorFlow can use the full GPU capacity.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            # Memory growth must be set before GPUs are initialized
            print(f"Note: Could not set memory growth: {e}")
# ModelBundle imported lazily in save_bundle() to avoid circular import


def build_lstm_model(
    input_shape: Tuple[int, int],
    num_genres: int,
    config: ModelTrainingConfig,
    num_instruments: int = 129,
) -> Model:
    """
    Build LSTM model with genre and instrument conditioning.

    Args:
        input_shape: (num_pitches, max_time_steps) e.g. (128, 1000)
        num_genres: Number of genres in vocabulary
        config: ModelTrainingConfig with architecture settings
        num_instruments: Number of instruments (default 129: 0-127 + drums)

    Returns:
        Compiled Keras model
    """
    # Regularizer
    if config.l1_reg > 0 and config.l2_reg > 0:
        reg = regularizers.l1_l2(l1=config.l1_reg, l2=config.l2_reg)
    elif config.l1_reg > 0:
        reg = regularizers.l1(config.l1_reg)
    elif config.l2_reg > 0:
        reg = regularizers.l2(config.l2_reg)
    else:
        reg = None

    # === Inputs ===
    pianoroll_input = Input(shape=input_shape, name='pianoroll_input')
    genre_input = Input(shape=(1,), dtype='int32', name='genre_input')
    instrument_input = Input(shape=(1,), dtype='int32', name='instrument_input')
    drum_input = Input(shape=input_shape, name='drum_input')

    # === Genre embedding ===
    genre_emb = Embedding(
        input_dim=num_genres + 1,  # +1 for unknown/padding
        output_dim=config.genre_embedding_dim,
        name='genre_embedding'
    )(genre_input)
    genre_flat = Flatten(name='genre_flat')(genre_emb)

    # === Instrument embedding ===
    instrument_emb = Embedding(
        input_dim=num_instruments + 1,  # +1 for unknown/padding
        output_dim=config.instrument_embedding_dim,
        name='instrument_embedding'
    )(instrument_input)
    instrument_flat = Flatten(name='instrument_flat')(instrument_emb)

    # === Combine conditioning vectors ===
    conditioning = Concatenate(name='conditioning_concat')([genre_flat, instrument_flat])

    # === Process pianoroll through LSTM ===
    # Transpose: (128, time_steps) -> (time_steps, 128) for LSTM
    x = tf.keras.layers.Permute((2, 1), name='permute_input')(pianoroll_input)
    drum_x = tf.keras.layers.Permute((2, 1), name='permute_drum')(drum_input)
    time_steps = input_shape[1]

    # Concatenate pianoroll with drum track for alignment learning
    x = Concatenate(name='concat_drum')([x, drum_x])

    # Tile conditioning vector across time steps and concatenate
    conditioning_tiled = tf.keras.layers.RepeatVector(time_steps, name='conditioning_repeat')(conditioning)
    x = Concatenate(name='concat_conditioning')([x, conditioning_tiled])

    # === LSTM layers ===
    # All LSTM layers return sequences to preserve temporal structure
    for i, units in enumerate(config.lstm_units):
        lstm = LSTM(
            units,
            return_sequences=True,  # Always return sequences for seq-to-seq
            dropout=config.dropout_rate,
            recurrent_dropout=config.recurrent_dropout,
            kernel_regularizer=reg,
            name=f'lstm_{i}'
        )

        if config.bidirectional:
            lstm = Bidirectional(lstm, name=f'bilstm_{i}')

        x = lstm(x)

        if config.dropout_rate > 0 and i < len(config.lstm_units) - 1:
            x = Dropout(config.dropout_rate, name=f'dropout_lstm_{i}')(x)

    # === Dense layers (TimeDistributed to process each timestep) ===
    for i, units in enumerate(config.dense_units):
        x = TimeDistributed(
            Dense(units, activation='relu', kernel_regularizer=reg),
            name=f'dense_{i}'
        )(x)
        if config.dropout_rate > 0:
            x = Dropout(config.dropout_rate, name=f'dropout_dense_{i}')(x)

    # === Output - predict 128 pitches per timestep ===
    # Shape: (batch, time_steps, 128)
    x = TimeDistributed(
        Dense(input_shape[0], activation='sigmoid'),
        name='pitch_output'
    )(x)

    # Permute back to (128, time_steps) to match input pianoroll shape
    output = Permute((2, 1), name='pianoroll_output')(x)

    model = Model(
        inputs=[pianoroll_input, genre_input, instrument_input, drum_input],
        outputs=output,
        name='music_generator'
    )

    return model


class ModelTrainer:
    """
    Trainer for genre and instrument-conditioned LSTM music generation.

    Works with MusicDataset's tf.data.Dataset format.
    Supports distributed training via TensorFlow distribution strategies.
    """

    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model: Optional[Model] = None
        self.history = None
        self.num_genres = 0
        self.num_instruments = 129  # 0-127 + drums

        # Configure GPU memory before any TensorFlow operations
        configure_gpu_memory()

        # Setup distribution strategy
        self.strategy = self._setup_strategy()
        self.num_replicas = self.strategy.num_replicas_in_sync

        # Calculate effective batch size
        if config.batch_size_per_replica is not None:
            self.global_batch_size = calculate_global_batch_size(
                config.batch_size_per_replica, self.strategy
            )
        else:
            self.global_batch_size = config.batch_size

    def _setup_strategy(self) -> tf.distribute.Strategy:
        """
        Setup distribution strategy based on config.

        Returns:
            Configured tf.distribute.Strategy instance
        """
        strategy = DistributionStrategyFactory.create_strategy(
            strategy_name=self.config.distribution_strategy,
            devices=self.config.mirrored_devices,
            cross_device_ops=self.config.cross_device_ops,
        )

        if self.config.distribution_strategy != 'none':
            print(f"\nDistribution strategy: {self.config.distribution_strategy}")
            print(f"  Replicas in sync: {strategy.num_replicas_in_sync}")

        return strategy

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        is_training: bool = True,
    ) -> tf.data.Dataset:
        """
        Prepare dataset for training.

        Formats samples to match model input names including instrument and drum conditioning.
        Applies distributed training options when using distribution strategies.

        During training, randomly zeros out the input pianoroll (with probability
        controlled by config.input_dropout_rate) to force the model to learn
        generation from conditioning signals (genre, instrument, drums) alone.
        This enables generation with zero/random seeds at inference time.

        Args:
            dataset: Input tf.data.Dataset
            batch_size: Batch size (will use global_batch_size for distributed)
            shuffle: Whether to shuffle the dataset
            is_training: Whether this is a training dataset (affects sharding)

        Returns:
            Prepared tf.data.Dataset
        """
        input_dropout_rate = self.config.input_dropout_rate if is_training else 0.0

        def format_sample(sample):
            # Get drum pianoroll, default to zeros if not present
            drum_pianoroll = sample.get('drum_pianoroll', tf.zeros_like(sample['pianoroll']))
            pianoroll = sample['pianoroll']

            # During training, randomly zero out input to force conditioning-based generation
            if input_dropout_rate > 0:
                should_dropout = tf.random.uniform(()) < input_dropout_rate
                pianoroll_input = tf.cond(
                    should_dropout,
                    lambda: tf.zeros_like(pianoroll),
                    lambda: pianoroll
                )
            else:
                pianoroll_input = pianoroll

            inputs = {
                'pianoroll_input': pianoroll_input,
                'genre_input': tf.reshape(sample['genre_id'], (1,)),
                'instrument_input': tf.reshape(sample['instrument_id'], (1,)),
                'drum_input': drum_pianoroll,
            }
            # Target is always the original pianoroll
            target = sample['pianoroll']
            return inputs, target

        dataset = dataset.map(format_sample, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size)

        # Use global batch size for distributed training
        effective_batch_size = self.global_batch_size if is_training else batch_size
        dataset = dataset.batch(effective_batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # For multi-worker strategy, shard the dataset
        if (self.config.distribution_strategy == 'multi_worker_mirrored' and is_training):
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = (
                tf.data.experimental.AutoShardPolicy.DATA
            )
            dataset = dataset.with_options(options)

        return dataset

    def build_model(
        self,
        num_genres: int,
        input_shape: Optional[Tuple[int, int]] = None,
        num_instruments: int = 129,
    ) -> Model:
        """
        Build and compile the LSTM model.

        Model is built within the distribution strategy scope to enable
        distributed training across multiple devices/nodes.

        Args:
            num_genres: Number of genres in vocabulary
            input_shape: Optional override for (num_pitches, max_time_steps)
            num_instruments: Number of instruments (default 129: 0-127 + drums)
        """
        self.num_genres = num_genres
        self.num_instruments = num_instruments

        if input_shape is None:
            input_shape = (self.config.num_pitches, self.config.max_time_steps)

        print(f"\nBuilding model:")
        print(f"  Input shape: {input_shape}")
        print(f"  Num genres: {num_genres}")
        print(f"  Num instruments: {num_instruments}")
        print(f"  Genre embedding dim: {self.config.genre_embedding_dim}")
        print(f"  Instrument embedding dim: {self.config.instrument_embedding_dim}")
        print(f"  LSTM units: {self.config.lstm_units}")
        print(f"  Dense units: {self.config.dense_units}")
        print(f"  Bidirectional: {self.config.bidirectional}")

        if self.config.distribution_strategy != 'none':
            print(f"  Distribution strategy: {self.config.distribution_strategy}")
            print(f"  Replicas: {self.num_replicas}")
            print(f"  Global batch size: {self.global_batch_size}")

        # Build model within strategy scope for distributed training
        with self.strategy.scope():
            self.model = build_lstm_model(
                input_shape=input_shape,
                num_genres=num_genres,
                config=self.config,
                num_instruments=num_instruments,
            )

            # Compile
            optimizer = self._get_optimizer()
            self.model.compile(
                optimizer=optimizer,
                loss=self.config.loss_function,
                metrics=self.config.metrics,
            )

        print(f"\nModel: {self.config.model_name}")
        self.model.summary()

        return self.model

    def _get_optimizer(self):
        """Get optimizer based on config."""
        lr = self.config.learning_rate
        name = self.config.optimizer.lower()

        if name == 'adam':
            return keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=self.config.beta_1,
                beta_2=self.config.beta_2,
                epsilon=self.config.epsilon,
            )
        elif name == 'sgd':
            return keras.optimizers.SGD(
                learning_rate=lr,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov,
            )
        elif name == 'rmsprop':
            return keras.optimizers.RMSprop(
                learning_rate=lr,
                rho=self.config.rho,
                epsilon=self.config.epsilon,
            )
        elif name == 'adamax':
            return keras.optimizers.Adamax(learning_rate=lr)
        elif name == 'nadam':
            return keras.optimizers.Nadam(learning_rate=lr)
        else:
            return keras.optimizers.Adam(learning_rate=lr)

    def _get_callbacks(self) -> List:
        """Build training callbacks based on config."""
        callbacks = []
        config = self.config
        output_dir = Path(config.output_dir) / config.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.use_early_stopping:
            callbacks.append(EarlyStopping(
                monitor=config.early_stopping_monitor,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                mode=config.early_stopping_mode,
                restore_best_weights=config.restore_best_weights,
                verbose=1
            ))

        if config.use_checkpointing:
            ckpt_path = output_dir / 'checkpoints' / 'model-{epoch:03d}-{val_loss:.4f}.keras'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor=config.checkpoint_monitor,
                mode=config.checkpoint_mode,
                save_best_only=config.save_best_only,
                save_weights_only=config.save_weights_only,
                verbose=1
            ))

        if config.use_tensorboard:
            log_dir = Path(config.tensorboard_log_dir) / config.model_name
            callbacks.append(TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=config.tensorboard_histogram_freq,
                write_graph=config.tensorboard_write_graph,
            ))

        if config.lr_schedule == 'reduce_on_plateau':
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.lr_drop_factor,
                patience=config.lr_patience,
                min_lr=config.lr_min_lr,
                verbose=1
            ))

        return callbacks

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        num_genres: int,
        num_instruments: int = 129,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training tf.data.Dataset from MusicDataset
            val_dataset: Validation tf.data.Dataset
            num_genres: Number of genres in vocabulary
            num_instruments: Number of instruments (default 129)

        Returns:
            Training history dict
        """
        if self.model is None:
            self.build_model(num_genres, num_instruments=num_instruments)

        train_ds = self._prepare_dataset(
            train_dataset,
            self.config.batch_size,
            shuffle=self.config.shuffle,
            is_training=True,
        )
        val_ds = self._prepare_dataset(
            val_dataset,
            self.config.batch_size,
            shuffle=False,
            is_training=False,
        )

        callbacks = self._get_callbacks()

        print(f"\nTraining for {self.config.epochs} epochs...")
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        if self.config.save_final_model:
            output_dir = Path(self.config.output_dir) / self.config.model_name
            model_path = output_dir / 'final_model.keras'
            self.model.save(model_path)
            print(f"\nSaved model: {model_path}")

        # Save training history
        if self.config.save_history:
            output_dir = Path(self.config.output_dir) / self.config.model_name
            history_path = output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                history_data = {
                    k: [float(v) for v in vals]
                    for k, vals in self.history.history.items()
                }
                json.dump(history_data, f, indent=2)
            print(f"Saved history: {history_path}")

        return self.history.history

    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the model on test dataset.

        Args:
            test_dataset: Test tf.data.Dataset

        Returns:
            Dict of metric names to values
        """
        if self.model is None:
            raise ValueError("Train or load a model first")

        test_ds = self._prepare_dataset(
            test_dataset,
            self.config.batch_size,
            shuffle=False,
            is_training=False,
        )

        print("\nEvaluating...")
        results = self.model.evaluate(test_ds, verbose=1)
        results_dict = dict(zip(self.model.metrics_names, results))

        print("\nResults:")
        for name, value in results_dict.items():
            print(f"  {name}: {value:.4f}")

        return results_dict

    def save_bundle(
        self,
        filepath: str,
        vocabulary: Vocabulary,
    ):
        """
        Save the model bundled with vocabulary for generation.

        Args:
            filepath: Path to save the bundle
            vocabulary: Vocabulary from the training MusicDataset

        Returns:
            ModelBundle instance
        """
        # Import here to avoid circular import
        from ..core.model_bundle import ModelBundle

        if self.model is None:
            raise ValueError("Train or load a model first")

        bundle = ModelBundle(
            model=self.model,
            vocabulary=vocabulary,
            training_config=self.config,
            model_name=self.config.model_name,
        )
        bundle.save(filepath)
        return bundle


def train_from_music_dataset(
    datasets: Dict[str, tf.data.Dataset],
    config: ModelTrainingConfig,
    vocabulary: Vocabulary,
) -> Tuple[Model, Dict[str, Any], ModelTrainer]:
    """
    Convenience function to train from MusicDataset splits.

    Args:
        datasets: Dict with 'train', 'validation', 'test' keys
        config: ModelTrainingConfig
        vocabulary: Vocabulary from MusicDataset

    Returns:
        Tuple of (model, history, trainer)
    """
    trainer = ModelTrainer(config)

    history = trainer.train(
        datasets['train'],
        datasets['validation'],
        num_genres=vocabulary.num_genres,
        num_instruments=vocabulary.num_instruments,
    )

    if 'test' in datasets:
        trainer.evaluate(datasets['test'])

    return trainer.model, history, trainer  # type: ignore
