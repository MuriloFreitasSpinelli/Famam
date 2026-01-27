"""
Trainer for Transformer-based autoregressive music generation.

Key differences from LSTM trainer:
- Loss: SparseCategoricalCrossentropy (not MSE)
- Masked loss computation (ignore PAD tokens)
- Transformer learning rate schedule with warmup
- Teacher forcing (input shifted by 1 = target)
"""

import json
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping, ModelCheckpoint, TensorBoard, Callback
)

from .configs.transformer_config import TransformerTrainingConfig
from .transformer_model import build_transformer_from_config, MusicTransformer
from .distribution_strategy import (
    DistributionStrategyFactory,
    calculate_global_batch_size,
)
from ..core.vocabulary import Vocabulary
from ..core.event_vocabulary import EventVocabulary


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

    This increases linearly during warmup, then decreases proportionally
    to the inverse square root of step number.
    """

    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Add small epsilon to avoid division by zero
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
    Sparse categorical crossentropy loss that ignores PAD tokens.

    The loss is computed only for non-PAD positions in the target.
    """

    def __init__(
        self,
        pad_token_id: int,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none',
        )

    def call(self, y_true, y_pred):
        """
        Compute masked loss.

        Args:
            y_true: Target token IDs (batch, seq_len)
            y_pred: Predicted logits (batch, seq_len, vocab_size)

        Returns:
            Scalar loss value
        """
        # Compute per-token loss
        loss = self.loss_fn(y_true, y_pred)

        # Create mask: 1 for non-PAD, 0 for PAD
        mask = tf.cast(y_true != self.pad_token_id, tf.float32)

        # Apply mask
        loss = loss * mask

        # Mean over non-PAD tokens
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pad_token_id': self.pad_token_id,
            'label_smoothing': self.label_smoothing,
        })
        return config


class MaskedAccuracy(keras.metrics.Metric):
    """
    Accuracy metric that ignores PAD tokens.
    """

    def __init__(self, pad_token_id: int, name='masked_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_token_id = pad_token_id
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update accuracy for non-PAD tokens."""
        # Get predictions (cast to int32 to match y_true dtype)
        predictions = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        # Create mask
        mask = tf.cast(y_true != self.pad_token_id, tf.float32)

        # Compute correct predictions
        correct = tf.cast(predictions == y_true, tf.float32) * mask

        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct / tf.maximum(self.total, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class TransformerTrainer:
    """
    Trainer for Transformer-based autoregressive music generation.

    Works with event sequences from MusicDataset.to_event_tensorflow_dataset().
    """

    def __init__(
        self,
        config: TransformerTrainingConfig,
        event_vocab: EventVocabulary,
    ):
        self.config = config
        self.event_vocab = event_vocab
        self.model: Optional[MusicTransformer] = None
        self.history = None

        # Configure GPU memory
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
        """Setup distribution strategy based on config."""
        strategy = DistributionStrategyFactory.create_strategy(
            strategy_name=self.config.distribution_strategy,
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

        Args:
            dataset: Input tf.data.Dataset with input_ids, attention_mask, labels
            batch_size: Batch size
            shuffle: Whether to shuffle
            is_training: Whether this is training data

        Returns:
            Prepared tf.data.Dataset
        """

        def format_sample(sample):
            """Format sample for model input."""
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

    def build_model(self) -> MusicTransformer:
        """
        Build and compile the Transformer model.

        Model is built within the distribution strategy scope.
        """
        vocab_size = self.event_vocab.vocab_size

        print(f"\nBuilding Transformer model:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Max sequence length: {self.config.max_seq_length}")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Model dimension: {self.config.d_model}")
        print(f"  Attention heads: {self.config.num_heads}")
        print(f"  FFN dimension: {self.config.d_ff}")
        print(f"  Dropout: {self.config.dropout_rate}")

        if self.config.distribution_strategy != 'none':
            print(f"  Distribution strategy: {self.config.distribution_strategy}")
            print(f"  Replicas: {self.num_replicas}")
            print(f"  Global batch size: {self.global_batch_size}")

        # Build model within strategy scope
        with self.strategy.scope():
            self.model = build_transformer_from_config(
                config=self.config,
                vocab_size=vocab_size,
            )

            # Create optimizer with LR schedule
            if self.config.use_transformer_lr_schedule:
                lr_schedule = TransformerLRSchedule(
                    d_model=self.config.d_model,
                    warmup_steps=self.config.warmup_steps,
                )
            else:
                lr_schedule = self.config.learning_rate

            if self.config.optimizer.lower() == 'adamw':
                # Try AdamW, fall back to Adam if not available (older TF/Keras)
                try:
                    optimizer = keras.optimizers.AdamW(
                        learning_rate=lr_schedule,
                        weight_decay=self.config.weight_decay,
                        beta_1=self.config.beta_1,
                        beta_2=self.config.beta_2,
                        epsilon=self.config.epsilon,
                        clipnorm=self.config.gradient_clip_value if self.config.use_gradient_clipping else None,
                    )
                except AttributeError:
                    print("Warning: AdamW not available, falling back to Adam")
                    optimizer = keras.optimizers.Adam(
                        learning_rate=lr_schedule,
                        beta_1=self.config.beta_1,
                        beta_2=self.config.beta_2,
                        epsilon=self.config.epsilon,
                        clipnorm=self.config.gradient_clip_value if self.config.use_gradient_clipping else None,
                    )
            else:
                optimizer = keras.optimizers.Adam(
                    learning_rate=lr_schedule,
                    beta_1=self.config.beta_1,
                    beta_2=self.config.beta_2,
                    epsilon=self.config.epsilon,
                    clipnorm=self.config.gradient_clip_value if self.config.use_gradient_clipping else None,
                )

            # Create masked loss
            loss = MaskedSparseCategoricalCrossentropy(
                pad_token_id=self.event_vocab.PAD_TOKEN,
                label_smoothing=self.config.label_smoothing,
            )

            # Create masked accuracy metric
            accuracy = MaskedAccuracy(
                pad_token_id=self.event_vocab.PAD_TOKEN,
            )

            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[accuracy],
            )

        print(f"\nModel: {self.config.model_name}")
        # Build model to get summary
        self.model.build(input_shape={'input_ids': (None, self.config.max_seq_length),
                                      'attention_mask': (None, self.config.max_seq_length)})
        self.model.summary()

        return self.model

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
    ) -> Dict[str, Any]:
        """
        Train the Transformer model.

        Args:
            train_dataset: Training tf.data.Dataset from MusicDataset.to_event_tensorflow_dataset()
            val_dataset: Validation tf.data.Dataset

        Returns:
            Training history dict
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

        # Save final model (use SavedModel format for subclassed models)
        if self.config.save_final_model:
            output_dir = Path(self.config.output_dir) / self.config.model_name
            model_path = output_dir / 'final_model'
            self.model.save(model_path, save_format='tf')
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

        return self.model, self.history.history

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
        Save the model bundled with vocabularies for generation.

        Args:
            filepath: Path to save the bundle
            vocabulary: Vocabulary from the training MusicDataset

        Returns:
            TransformerModelBundle instance
        """
        from ..core.model_bundle import TransformerModelBundle

        if self.model is None:
            raise ValueError("Train or load a model first")

        bundle = TransformerModelBundle(
            model=self.model,
            event_vocabulary=self.event_vocab,
            vocabulary=vocabulary,
            training_config=self.config,
            model_name=self.config.model_name,
        )
        bundle.save(filepath)
        return bundle


def train_transformer_from_music_dataset(
    datasets: Dict[str, tf.data.Dataset],
    config: TransformerTrainingConfig,
    vocabulary: Vocabulary,
    event_vocab: EventVocabulary,
) -> Tuple[MusicTransformer, Dict[str, Any], TransformerTrainer]:
    """
    Convenience function to train Transformer from MusicDataset event splits.

    Args:
        datasets: Dict with 'train', 'validation', 'test' keys
        config: TransformerTrainingConfig
        vocabulary: Vocabulary from MusicDataset
        event_vocab: EventVocabulary for token encoding

    Returns:
        Tuple of (model, history, trainer)
    """
    trainer = TransformerTrainer(config, event_vocab)

    history = trainer.train(
        datasets['train'],
        datasets['validation'],
    )

    if 'test' in datasets:
        trainer.evaluate(datasets['test'])

    return trainer.model, history, trainer
