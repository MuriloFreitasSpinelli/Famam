"""
Trainer module for conditioned music generation.

Consumes tf.data.Dataset directly from generate_tensorflow_dataset.py.
Automatically detects what conditioning is available (genre, instruments, both, or none)
and builds the appropriate model architecture.
"""

import keras
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
import json
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, LSTM, Bidirectional, Dense, Dropout, Embedding,
    Concatenate, Flatten, Reshape
)
from tensorflow.keras.callbacks import ( # type: ignore
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)

from training.configs.training_config import TrainingConfig


class ConditioningType(Enum):
    """What conditioning is available in the dataset."""
    NONE = auto()           # music-only
    GENRE = auto()          # music + genre_id
    INSTRUMENTS = auto()    # music + instrument_ids
    FULL = auto()           # music + genre_id + instrument_ids (+ artist_id)


@dataclass
class DatasetInfo:
    """Information extracted from the dataset."""
    music_shape: Tuple[int, ...]
    conditioning_type: ConditioningType
    has_genre: bool = False
    has_instruments: bool = False
    has_artist: bool = False
    max_tracks: int = 16


def detect_dataset_info(dataset: tf.data.Dataset) -> DatasetInfo:
    """
    Detect what's in the dataset by examining a sample.

    Args:
        dataset: tf.data.Dataset from generate_tensorflow_dataset

    Returns:
        DatasetInfo with detected structure
    """
    for sample in dataset.take(1):
        if not isinstance(sample, dict):
            # Plain tensor, no conditioning
            return DatasetInfo(
                music_shape=tuple(sample.shape), # type: ignore
                conditioning_type=ConditioningType.NONE,
            )

        music_shape = tuple(sample['music'].shape)
        has_genre = 'genre_id' in sample
        has_instruments = 'instrument_ids' in sample
        has_artist = 'artist_id' in sample
        max_tracks = sample['instrument_ids'].shape[0] if has_instruments else 16

        # Determine conditioning type
        if has_genre and has_instruments:
            cond_type = ConditioningType.FULL
        elif has_genre:
            cond_type = ConditioningType.GENRE
        elif has_instruments:
            cond_type = ConditioningType.INSTRUMENTS
        else:
            cond_type = ConditioningType.NONE

        return DatasetInfo(
            music_shape=music_shape,
            conditioning_type=cond_type,
            has_genre=has_genre,
            has_instruments=has_instruments,
            has_artist=has_artist,
            max_tracks=max_tracks,
        )

    raise ValueError("Dataset is empty")


def build_model(
    music_shape: Tuple[int, ...],
    dataset_info: DatasetInfo,
    num_genres: int = 0,
    num_artists: int = 0,
    num_instruments: int = 129,
    lstm_units: List[int] = None, # type: ignore
    dense_units: List[int] = None, # type: ignore
    dropout_rate: float = 0.2,
    recurrent_dropout: float = 0.1,
    bidirectional: bool = False,
    embedding_dim: int = 32,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
) -> Model:
    """
    Build model based on what conditioning is available.

    Only creates embedding layers for conditioning that exists in the dataset.
    """
    from tensorflow.keras import regularizers # type: ignore

    lstm_units = lstm_units or [128, 64]
    dense_units = dense_units or [64, 32]

    # Regularizer
    if l1_reg > 0 and l2_reg > 0:
        reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = regularizers.l1(l1_reg)
    elif l2_reg > 0:
        reg = regularizers.l2(l2_reg)
    else:
        reg = None

    # === Music input (always present) ===
    music_input = Input(shape=music_shape, name='music_input')

    # === Build conditioning inputs based on what's available ===
    inputs = [music_input]
    conditioning_layers = []

    if dataset_info.has_genre and num_genres > 0:
        genre_input = Input(shape=(1,), dtype='int32', name='genre_input')
        inputs.append(genre_input)

        genre_emb = Embedding(
            input_dim=num_genres + 1,  # +1 for unknown/padding
            output_dim=embedding_dim,
            name='genre_embedding'
        )(genre_input)
        conditioning_layers.append(Flatten(name='genre_flat')(genre_emb))

    if dataset_info.has_artist and num_artists > 0:
        artist_input = Input(shape=(1,), dtype='int32', name='artist_input')
        inputs.append(artist_input)

        artist_emb = Embedding(
            input_dim=num_artists + 1,
            output_dim=embedding_dim,
            name='artist_embedding'
        )(artist_input)
        conditioning_layers.append(Flatten(name='artist_flat')(artist_emb))

    if dataset_info.has_instruments:
        instrument_input = Input(
            shape=(dataset_info.max_tracks,),
            dtype='int32',
            name='instrument_input'
        )
        inputs.append(instrument_input)

        instrument_emb = Embedding(
            input_dim=num_instruments + 1,
            output_dim=embedding_dim,
            name='instrument_embedding'
        )(instrument_input)
        conditioning_layers.append(Flatten(name='instrument_flat')(instrument_emb))

    # === Process music through LSTM ===
    x = music_input

    # Ensure correct shape for LSTM (timesteps, features)
    if len(music_shape) == 1:
        x = Reshape((1, music_shape[0]))(x)
    elif len(music_shape) == 2:
        # (features, timesteps) -> transpose to (timesteps, features) for LSTM
        # Note: pianoroll from muspy is (128, timesteps), we need (timesteps, 128)
        x = tf.keras.layers.Permute((2, 1))(x) # type: ignore

    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)

        lstm = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=reg,
            name=f'lstm_{i}'
        )

        if bidirectional:
            lstm = Bidirectional(lstm, name=f'bilstm_{i}')

        x = lstm(x)

        if dropout_rate > 0 and i < len(lstm_units) - 1:
            x = Dropout(dropout_rate)(x)

    # === Merge with conditioning (if any) ===
    if conditioning_layers:
        conditioning = Concatenate(name='conditioning')(conditioning_layers)
        x = Concatenate(name='merge')([x, conditioning])

    # === Dense layers ===
    for i, units in enumerate(dense_units):
        x = Dense(units, activation='relu', kernel_regularizer=reg, name=f'dense_{i}')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # === Output - same shape as input music ===
    output_size = int(np.prod(music_shape))
    x = Dense(output_size, activation='sigmoid', name='output_dense')(x)

    if len(music_shape) > 1:
        output = Reshape(music_shape, name='output')(x)
    else:
        output = x

    model = Model(inputs=inputs, outputs=output, name='music_generator')
    return model


class Trainer:
    """
    Trainer that automatically adapts to the dataset structure.

    Detects conditioning type and uses vocabulary from the dataset.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: Optional[Model] = None
        self.history = None
        self.dataset_info: Optional[DatasetInfo] = None

        # Vocabulary sizes - should be set from dataset.vocabulary
        self.num_genres = 0
        self.num_artists = 0
        self.num_instruments = 129  # GM standard

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Prepare dataset for training.

        Formats samples to match model input names.
        """
        info = self.dataset_info

        def format_sample(sample):
            if not isinstance(sample, dict):
                # Music-only dataset
                return {'music_input': sample}, sample

            inputs = {'music_input': sample['music']}

            # Add conditioning inputs that exist
            if info.has_genre and 'genre_id' in sample: # type: ignore
                inputs['genre_input'] = tf.reshape(sample['genre_id'], (1,))

            if info.has_artist and 'artist_id' in sample: # type: ignore
                inputs['artist_input'] = tf.reshape(sample['artist_id'], (1,))

            if info.has_instruments and 'instrument_ids' in sample: # type: ignore
                inputs['instrument_input'] = sample['instrument_ids']

            # Target is the music (for reconstruction/prediction)
            target = sample['music']

            return inputs, target

        dataset = dataset.map(format_sample, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def build_model(
        self,
        dataset: tf.data.Dataset,
        num_genres: int = 0,
        num_artists: int = 0,
    ) -> Model:
        """
        Build model based on dataset structure and vocabulary.

        Args:
            dataset: tf.data.Dataset to analyze
            num_genres: From dataset.vocabulary.num_genres
            num_artists: From dataset.vocabulary.num_artists
        """
        self.dataset_info = detect_dataset_info(dataset)
        self.num_genres = num_genres
        self.num_artists = num_artists

        print(f"\nDataset analysis:")
        print(f"  Music shape: {self.dataset_info.music_shape}")
        print(f"  Conditioning: {self.dataset_info.conditioning_type.name}")
        print(f"  Has genre: {self.dataset_info.has_genre} (vocab size: {num_genres})")
        print(f"  Has instruments: {self.dataset_info.has_instruments}")
        print(f"  Has artist: {self.dataset_info.has_artist} (vocab size: {num_artists})")

        self.model = build_model(
            music_shape=self.dataset_info.music_shape,
            dataset_info=self.dataset_info,
            num_genres=num_genres,
            num_artists=num_artists,
            num_instruments=self.num_instruments,
            lstm_units=self.config.lstm_units,
            dense_units=self.config.dense_units,
            dropout_rate=self.config.dropout_rate,
            recurrent_dropout=self.config.recurrent_dropout,
            bidirectional=self.config.bidirectional,
            l1_reg=self.config.l1_reg,
            l2_reg=self.config.l2_reg,
        )

        # Compile
        optimizer = self._get_optimizer()
        self.model.compile( # type: ignore
            optimizer=optimizer,
            loss=self.config.loss_function,
            metrics=self.config.metrics,
        )

        print(f"\nModel: {self.config.model_name}")
        self.model.summary() # type: ignore

        return self.model

    def _get_optimizer(self):
        lr = self.config.learning_rate
        name = self.config.optimizer.lower()

        optimizers = {
            'adam': keras.optimizers.Adam(learning_rate=lr),
            'sgd': keras.optimizers.SGD(learning_rate=lr, momentum=self.config.momentum),
            'rmsprop': keras.optimizers.RMSprop(learning_rate=lr),
            'adamax': keras.optimizers.Adamax(learning_rate=lr),
            'nadam': keras.optimizers.Nadam(learning_rate=lr),
        }
        return optimizers.get(name, keras.optimizers.Adam(learning_rate=lr))

    def _get_callbacks(self) -> List:
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
        num_genres: int = 0,
        num_artists: int = 0,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training tf.data.Dataset
            val_dataset: Validation tf.data.Dataset
            num_genres: From dataset.vocabulary.num_genres
            num_artists: From dataset.vocabulary.num_artists
        """
        if self.model is None:
            self.build_model(train_dataset, num_genres, num_artists)

        train_ds = self._prepare_dataset(
            train_dataset,
            self.config.batch_size,
            shuffle=self.config.shuffle
        )
        val_ds = self._prepare_dataset(
            val_dataset,
            self.config.batch_size,
            shuffle=False
        )

        callbacks = self._get_callbacks()

        print(f"\nTraining for {self.config.epochs} epochs...")
        self.history = self.model.fit( # type: ignore
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Save
        if self.config.save_final_model:
            output_dir = Path(self.config.output_dir) / self.config.model_name
            model_path = output_dir / 'final_model.keras'
            self.model.save(model_path) # type: ignore
            print(f"\nSaved: {model_path}")

        if self.config.save_history:
            output_dir = Path(self.config.output_dir) / self.config.model_name
            history_path = output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump({k: [float(v) for v in vals]
                          for k, vals in self.history.history.items()}, f, indent=2)

        return self.history.history

    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Train the model first")

        test_ds = self._prepare_dataset(
            test_dataset,
            self.config.batch_size,
            shuffle=False
        )

        print("\nEvaluating...")
        results = self.model.evaluate(test_ds, verbose=1)
        results_dict = dict(zip(self.model.metrics_names, results))

        print("\nResults:")
        for name, value in results_dict.items():
            print(f"  {name}: {value:.4f}")

        return results_dict

    def save_with_vocabulary(
        self,
        filepath: str,
        vocabulary: "DatasetVocabulary",
        tensorflow_dataconfig: Optional["TensorflowDatasetConfig"] = None,
    ) -> "SavedModel":
        """
        Save the model bundled with vocabulary for generation.

        Args:
            filepath: Path to save the .h5 bundle
            vocabulary: DatasetVocabulary from the training dataset
            tensorflow_dataconfig: Optional config with representation type info

        Returns:
            SavedModel instance
        """
        if self.model is None or self.dataset_info is None:
            raise ValueError("Train the model first")

        from training.saved_model import SavedModel
        from dataclasses import asdict

        saved = SavedModel(
            model=self.model,
            vocabulary=vocabulary,
            dataset_info=self.dataset_info,
            training_config=self.config,
            tensorflow_dataconfig=tensorflow_dataconfig,
            model_name=self.config.model_name,
        )
        saved.save(filepath)
        return saved


# Import for type hints
from data.dataset_vocabulary import DatasetVocabulary
from data.configs.tensorflow_dataset_config import TensorflowDatasetConfig
from training.saved_model import SavedModel


def train_from_datasets(
    datasets: Dict[str, tf.data.Dataset],
    config: TrainingConfig,
    num_genres: int = 0,
    num_artists: int = 0,
) -> Tuple[Model, Dict[str, Any]]:
    """
    Convenience function to train from dataset dict.

    Args:
        datasets: Dict with 'train', 'validation', 'test' keys
        config: TrainingConfig
        num_genres: From dataset.vocabulary.num_genres
        num_artists: From dataset.vocabulary.num_artists
    """
    trainer = Trainer(config)
    history = trainer.train(
        datasets['train'],
        datasets['validation'],
        num_genres=num_genres,
        num_artists=num_artists,
    )
    trainer.evaluate(datasets['test'])
    return trainer.model, history
