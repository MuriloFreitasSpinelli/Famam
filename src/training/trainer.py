"""
Unified trainer for music generation models.

Supports both Transformer and LSTM architectures with:
    - Masked loss computation (ignores PAD tokens)
    - Transformer learning rate schedule with warmup
    - Early stopping, checkpointing, TensorBoard
    - Distributed training support

Author: Murilo de Freitas Spinelli, Timofey
"""

import json
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, Callback
)

from ..config import TrainingConfig
from ..models.base_model import BaseMusicModel
from ..models.transformer_model import TransformerModel, build_transformer_from_config
from ..models.lstm_model import LSTMModel, build_lstm_from_config

if TYPE_CHECKING:
    from ..data.encoders import BaseEncoder


def configure_gpu_memory():
    """Configure GPU memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Note: Could not set memory growth: {e}")


class TransformerLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule from "Attention Is All You Need".

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step = tf.maximum(step, 1.0)

        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.math.pow(self.warmup_steps, -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': int(self.d_model.numpy()),
            'warmup_steps': int(self.warmup_steps.numpy()),
        }


class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    """
    Sparse categorical crossentropy that ignores PAD tokens.
    """

    def __init__(self, pad_token_id: int, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none',
        )

    def call(self, y_true, y_pred):
        loss = self.loss_fn(y_true, y_pred)
        mask = tf.cast(y_true != self.pad_token_id, tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pad_token_id': self.pad_token_id,
            'label_smoothing': self.label_smoothing,
        })
        return config


class MaskedAccuracy(keras.metrics.Metric):
    """Accuracy metric that ignores PAD tokens."""

    def __init__(self, pad_token_id: int, name='masked_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_token_id = pad_token_id
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        predictions = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        mask = tf.cast(y_true != self.pad_token_id, tf.float32)
        correct = tf.cast(predictions == y_true, tf.float32) * mask
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct / tf.maximum(self.total, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class Trainer:
    """
    Unified trainer for music generation models.

    Supports both Transformer and LSTM architectures.
    """

    def __init__(
        self,
        config: TrainingConfig,
        encoder: "BaseEncoder",
    ):
        """
        Initialize trainer.

        Args:
            config: TrainingConfig instance
            encoder: BaseEncoder instance (EventEncoder or REMIEncoder)
        """
        self.config = config
        self.encoder = encoder
        self.model: Optional[BaseMusicModel] = None
        self.history = None

        configure_gpu_memory()

        # Setup distribution strategy
        self.strategy = self._setup_strategy()
        self.num_replicas = self.strategy.num_replicas_in_sync

        # Calculate effective batch size
        if config.batch_size_per_replica is not None:
            self.global_batch_size = config.batch_size_per_replica * self.num_replicas
        else:
            self.global_batch_size = config.batch_size

    def _setup_strategy(self) -> tf.distribute.Strategy:
        """Setup distribution strategy."""
        strategy_name = self.config.distribution_strategy.lower()

        if strategy_name == 'none' or strategy_name == 'default':
            return tf.distribute.get_strategy()
        elif strategy_name == 'mirrored':
            return tf.distribute.MirroredStrategy()
        elif strategy_name == 'multi_worker_mirrored':
            return tf.distribute.MultiWorkerMirroredStrategy()
        else:
            print(f"Unknown strategy '{strategy_name}', using default")
            return tf.distribute.get_strategy()

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        is_training: bool = True,
    ) -> tf.data.Dataset:
        """
        Prepare dataset for training.

        Args:
            dataset: Input tf.data.Dataset with input_ids, attention_mask, labels
            batch_size: Batch size
            shuffle: Whether to shuffle
            is_training: Whether this is training data
        """
        def format_sample(sample):
            return (
                {
                    'input_ids': sample['input_ids'],
                    'attention_mask': sample['attention_mask'],
                },
                sample['labels'],
            )

        dataset = dataset.map(format_sample, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size)

        effective_batch_size = self.global_batch_size if is_training else batch_size
        dataset = dataset.batch(effective_batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def build_model(self) -> BaseMusicModel:
        """Build and compile the model based on config."""
        vocab_size = self.encoder.vocab_size
        pad_token_id = self.encoder.pad_token_id

        print(f"\nBuilding {self.config.model_type.upper()} model:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Max sequence length: {self.config.max_seq_length}")
        print(f"  Model dimension: {self.config.d_model}")

        with self.strategy.scope():
            # Build model based on type
            if self.config.model_type == "transformer":
                self.model = build_transformer_from_config(self.config, vocab_size)
                print(f"  Layers: {self.config.num_layers}")
                print(f"  Attention heads: {self.config.num_heads}")
                print(f"  FFN dimension: {self.config.d_ff}")
            else:
                self.model = build_lstm_from_config(self.config, vocab_size)
                print(f"  LSTM units: {self.config.lstm_units}")
                print(f"  Bidirectional: {self.config.bidirectional}")

            # Create optimizer
            optimizer = self._create_optimizer()

            # Create loss and metrics
            loss = MaskedSparseCategoricalCrossentropy(
                pad_token_id=pad_token_id,
                label_smoothing=self.config.label_smoothing,
            )
            accuracy = MaskedAccuracy(pad_token_id=pad_token_id)

            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[accuracy],
            )

        # Build to show summary
        self.model.build(input_shape={
            'input_ids': (None, self.config.max_seq_length),
            'attention_mask': (None, self.config.max_seq_length)
        })

        print(f"\nModel: {self.config.model_name}")
        self.model.summary()

        return self.model

    def _create_optimizer(self):
        """Create optimizer with learning rate schedule."""
        if self.config.use_lr_schedule and self.config.lr_schedule_type == "transformer":
            lr = TransformerLRSchedule(
                d_model=self.config.d_model,
                warmup_steps=self.config.warmup_steps,
            )
        else:
            lr = self.config.learning_rate

        optimizer_name = self.config.optimizer.lower()
        clip_norm = self.config.gradient_clip_value if self.config.use_gradient_clipping else None

        if optimizer_name == 'adamw':
            try:
                return keras.optimizers.AdamW(
                    learning_rate=lr,
                    weight_decay=self.config.weight_decay,
                    beta_1=self.config.beta_1,
                    beta_2=self.config.beta_2,
                    epsilon=self.config.epsilon,
                    clipnorm=clip_norm,
                )
            except AttributeError:
                print("AdamW not available, using Adam")
                optimizer_name = 'adam'

        if optimizer_name == 'adam':
            return keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=self.config.beta_1,
                beta_2=self.config.beta_2,
                epsilon=self.config.epsilon,
                clipnorm=clip_norm,
            )
        elif optimizer_name == 'sgd':
            return keras.optimizers.SGD(
                learning_rate=lr,
                clipnorm=clip_norm,
            )
        elif optimizer_name == 'rmsprop':
            return keras.optimizers.RMSprop(
                learning_rate=lr,
                clipnorm=clip_norm,
            )
        else:
            return keras.optimizers.Adam(learning_rate=lr, clipnorm=clip_norm)

    def _get_callbacks(self) -> List[Callback]:
        """Build training callbacks."""
        callbacks = []
        config = self.config
        output_dir = Path(config.output_dir) / config.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.use_early_stopping:
            callbacks.append(EarlyStopping(
                monitor=config.early_stopping_monitor,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                mode='min',
                restore_best_weights=True,
                verbose=1,
            ))

        if config.use_checkpointing:
            ckpt_path = output_dir / 'checkpoints' / 'model-{epoch:03d}-{val_loss:.4f}.keras'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor=config.checkpoint_monitor,
                mode='min',
                save_best_only=config.save_best_only,
                verbose=1,
            ))

        if config.use_tensorboard:
            log_dir = Path(config.tensorboard_log_dir) / config.model_name
            callbacks.append(TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
            ))

        return callbacks

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
    ) -> Tuple[BaseMusicModel, Dict[str, Any]]:
        """
        Train the model.

        Args:
            train_dataset: Training tf.data.Dataset
            val_dataset: Validation tf.data.Dataset

        Returns:
            Tuple of (model, history_dict)
        """
        if self.model is None:
            self.build_model()

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
            verbose=1,
        )

        # Save outputs
        output_dir = Path(self.config.output_dir) / self.config.model_name

        if self.config.save_final_model:
            model_path = output_dir / 'final_model'
            self.model.save(model_path, save_format='tf')
            print(f"\nSaved model: {model_path}")

        if self.config.save_history:
            history_path = output_dir / 'training_history.json'
            history_data = {
                k: [float(v) for v in vals]
                for k, vals in self.history.history.items()
            }
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"Saved history: {history_path}")

        # Save config
        config_path = output_dir / 'config.json'
        self.config.save(str(config_path))

        return self.model, self.history.history

    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Evaluate model on test dataset."""
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

    def load_model(self, model_path: str) -> BaseMusicModel:
        """Load a saved model."""
        self.model = keras.models.load_model(model_path)
        return self.model


def train_model(
    datasets: Dict[str, tf.data.Dataset],
    config: TrainingConfig,
    encoder: "BaseEncoder",
) -> Tuple[BaseMusicModel, Dict[str, Any], Trainer]:
    """
    Convenience function to train a model.

    Args:
        datasets: Dict with 'train', 'validation', (optional) 'test' keys
        config: TrainingConfig
        encoder: BaseEncoder instance

    Returns:
        Tuple of (model, history, trainer)
    """
    trainer = Trainer(config, encoder)

    model, history = trainer.train(
        datasets['train'],
        datasets['validation'],
    )

    if 'test' in datasets:
        trainer.evaluate(datasets['test'])

    return model, history, trainer
