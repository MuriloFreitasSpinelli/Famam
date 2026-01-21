"""
Training Pipeline Script

Runs the full hyperparameter tuning and training workflow.

Usage:
    # Run hyperparameter tuning only
    python scripts/train_pipeline.py tune --dataset my_dataset --experiment lstm_v1

    # Train using tuned parameters
    python scripts/train_pipeline.py train --tuning-results lstm_v1 --model-name lstm_v1_final

    # Run full pipeline (tune + train)
    python scripts/train_pipeline.py full --dataset my_dataset --experiment lstm_v1

    # List available tuning results
    python scripts/train_pipeline.py list

    # Specify representation type
    python scripts/train_pipeline.py train --dataset my_dataset --model-name test --representation piano-roll
"""

import argparse
import sys
from pathlib import Path
from functools import partial

# Add project root to path
script_dir = Path(__file__).resolve().parent  # src/training/scripts
src_root = script_dir.parent.parent  # src (contains data, training modules)
project_root = src_root.parent  # Famam root (contains data/datasets)
sys.path.insert(0, str(src_root))

import numpy as np
from data.configs.tensorflow_dataset_config import TensorflowDatasetConfig


def load_dataset(
    dataset_name: str,
    tf_config: TensorflowDatasetConfig = None, # type: ignore
    use_tensors: bool = True
):
    """
    Load dataset by name.

    Args:
        dataset_name: Name of the dataset (h5 file without extension)
        tf_config: TensorflowDatasetConfig with representation settings
        use_tensors: If True, load from pre-saved tensors (faster)

    Returns:
        Tuple of (datasets_dict, vocabulary, tf_config)
    """
    from data.tensorflow_dataset import load_tensors, tensors_exist
    from data.enhanced_music_dataset import EnhancedMusicDataset

    # Create default config if not provided
    if tf_config is None:
        tf_config = TensorflowDatasetConfig(
            tensor_name=dataset_name,
            tensor_type='full',
            representation_type='piano-roll',
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            output_dir='./models',
        )

    # Try loading pre-saved tensors first
    if use_tensors and tensors_exist(dataset_name):
        print(f"Loading pre-saved tensors: {dataset_name}")
        datasets = load_tensors(dataset_name)

        # Load vocabulary from original dataset
        dataset_path = project_root / "data" / "datasets" / f"{dataset_name}.h5"
        if dataset_path.exists():
            full_dataset = EnhancedMusicDataset.load(str(dataset_path))
            vocabulary = full_dataset.vocabulary
        else:
            from data.dataset_vocabulary import DatasetVocabulary
            vocabulary = DatasetVocabulary()
            print("Warning: Could not load vocabulary, using empty vocabulary")

        return datasets, vocabulary, tf_config

    # Load from EnhancedMusicDataset
    dataset_path = project_root / "data" / "datasets" / f"{dataset_name}.h5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    print(f"Representation: {tf_config.representation_type}")
    dataset = EnhancedMusicDataset.load(str(dataset_path))

    # Map representation names (handle both formats)
    rep_map = {
        'piano-roll': 'pianoroll',
        'pianoroll': 'pianoroll',
        'pitch': 'pitch',
        'event': 'event',
        'note': 'note',
    }
    representation = rep_map.get(tf_config.representation_type, tf_config.representation_type)

    # Create TF datasets with splits from config
    datasets = dataset.to_tensorflow_dataset_with_metadata(
        representation=representation,
        splits=(tf_config.train_split, tf_config.val_split, tf_config.test_split),
        random_state=42
    )

    return datasets, dataset.vocabulary, tf_config


def dataset_to_numpy_for_tuning(datasets, max_samples: int = 2000):
    """
    Convert TF dataset to numpy arrays for sklearn tuning.

    Args:
        datasets: Dict with 'train' tf.data.Dataset
        max_samples: Maximum samples to use

    Returns:
        Tuple of (X, y) numpy arrays
    """
    print(f"Converting dataset to numpy (max {max_samples} samples)...")

    music_list = []
    for i, sample in enumerate(datasets['train']):
        if i >= max_samples:
            break
        music_list.append(sample['music'].numpy())

    X = np.array(music_list)

    # For autoencoder-style training, y = X
    y = X.reshape(len(X), -1)  # Flatten for sklearn

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    return X, y


def create_tf_config(args) -> TensorflowDatasetConfig:
    """Create TensorflowDatasetConfig from args or load from JSON file."""
    # If config file is provided, load from it
    if hasattr(args, 'config') and args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading config from: {config_path}")
        return TensorflowDatasetConfig.load(str(config_path))

    # Otherwise, create from CLI arguments
    return TensorflowDatasetConfig(
        tensor_name=args.dataset,
        tensor_type=getattr(args, 'tensor_type', 'full'),
        representation_type=getattr(args, 'representation', 'piano-roll'),
        train_split=getattr(args, 'train_split', 0.8),
        val_split=getattr(args, 'val_split', 0.1),
        test_split=getattr(args, 'test_split', 0.1),
        output_dir=args.output_dir,
    )


def run_tuning(args):
    """Run hyperparameter tuning."""
    from training.configs.training_config import TrainingConfig
    from training.hyperparameter_tuning import (
        grid_search_hyperparameters,
        random_search_hyperparameters,
        KerasRegressorWrapper,
    )

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    # Create tf config and load dataset
    tf_config = create_tf_config(args)
    dataset_name = args.dataset if args.dataset else tf_config.tensor_name
    datasets, vocabulary, tf_config = load_dataset(dataset_name, tf_config)

    # Convert to numpy for sklearn
    X, y = dataset_to_numpy_for_tuning(datasets, max_samples=args.max_samples)

    # Create tuning config
    config = TrainingConfig(
        model_name=f"{args.experiment}_tuning",
        use_hyperparameter_tuning=True,
        tuning_method=args.method,
        tuning_cv_folds=args.cv_folds,
        tuning_scoring=args.scoring,
        tuning_n_jobs=args.n_jobs,
        n_iter=args.n_iter,
        epochs=args.tuning_epochs,  # Fewer epochs during tuning
        output_dir=args.output_dir,
        param_grid={
            'lstm_units': [[64], [128], [128, 64], [256, 128]],
            'dense_units': [[32], [64], [64, 32]],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
        } if args.param_grid is None else args.param_grid,
    )

    # Build function for creating models
    input_shape = X.shape[1:]
    output_size = y.shape[1]

    def build_model(
        input_shape,
        output_size,
        lstm_units=None,
        dense_units=None,
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,  # Ignored but passed by sklearn
        **kwargs
    ):
        """Build model for tuning."""
        import tensorflow as tf
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape # type: ignore

        lstm_units = lstm_units or [128]
        dense_units = dense_units or [64]

        inputs = Input(shape=input_shape)
        x = inputs

        # Reshape for LSTM if needed (timesteps, features)
        if len(input_shape) == 2:
            x = tf.keras.layers.Permute((2, 1))(x) # type: ignore

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

    build_fn = partial(build_model, input_shape=input_shape, output_size=output_size)

    # Create wrapper
    wrapper = KerasRegressorWrapper(
        build_fn=build_fn,
        epochs=config.epochs,
        verbose=0,
    )

    # Run search
    print(f"\nRunning {args.method} with {args.cv_folds}-fold CV...")
    print(f"Parameter grid: {config.param_grid}")

    if args.method == 'grid_search':
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(
            estimator=wrapper,
            param_grid=config.param_grid, # type: ignore
            cv=config.tuning_cv_folds,
            scoring=config.tuning_scoring,
            n_jobs=config.tuning_n_jobs,
            verbose=2,
            return_train_score=True,
        )
    else:  # random_search
        from sklearn.model_selection import RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=wrapper,
            param_distributions=config.param_grid, # type: ignore
            n_iter=config.n_iter,
            cv=config.tuning_cv_folds,
            scoring=config.tuning_scoring,
            n_jobs=config.tuning_n_jobs,
            verbose=2,
            random_state=42,
            return_train_score=True,
        )

    search.fit(X, y)

    print(f"\n{'=' * 70}")
    print("TUNING RESULTS")
    print(f"{'=' * 70}")
    print(f"Best Score: {search.best_score_:.4f}")
    print(f"Best Parameters: {search.best_params_}")

    # Save results
    results_path = config.save_tuning_results(search, args.experiment)

    print(f"\nResults saved to: {results_path}")
    print(f"\nTo train with these parameters, run:")
    print(f"  python scripts/train_pipeline.py train --tuning-results {args.experiment} --model-name {args.experiment}_final")

    return results_path


def run_training(args):
    """Run training with tuned or specified parameters."""
    from training.configs.training_config import TrainingConfig
    from training.trainer import Trainer

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    # Create tf config
    tf_config = create_tf_config(args)
    print(f"Representation type: {tf_config.representation_type}")

    # Get dataset name from args or config
    dataset_name = args.dataset if args.dataset else tf_config.tensor_name

    # Load config from tuning results or create new
    if args.tuning_results:
        results_path = Path(args.output_dir) / 'tuning' / f'{args.tuning_results}.json'
        if not results_path.exists():
            raise FileNotFoundError(f"Tuning results not found: {results_path}")

        config = TrainingConfig.from_tuning_results(
            str(results_path),
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_early_stopping=True,
            use_checkpointing=True,
            use_tensorboard=True,
            output_dir=args.output_dir,
        )
    else:
        # Use default config
        config = TrainingConfig(
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

    print(config.summary())

    # Load dataset with tf_config
    datasets, vocabulary, tf_config = load_dataset(dataset_name, tf_config)

    # Train
    trainer = Trainer(config)
    history = trainer.train(
        datasets['train'], # type: ignore
        datasets['validation'], # type: ignore
        num_genres=vocabulary.num_genres,
        num_artists=vocabulary.num_artists,
    )

    # Evaluate
    results = trainer.evaluate(datasets['test']) # type: ignore

    # Save with vocabulary and tf_config (for generation to know representation type)
    save_path = Path(args.output_dir) / config.model_name / 'model_bundle.h5'
    trainer.save_with_vocabulary(str(save_path), vocabulary, tf_config)

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Model saved to: {save_path}")
    print(f"Representation: {tf_config.representation_type}")
    print(f"Final test loss: {results.get('loss', 'N/A'):.4f}")

    return trainer, history


def run_full_pipeline(args):
    """Run full pipeline: tuning + training."""
    print("\n" + "=" * 70)
    print("FULL TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Tuning
    print("\n[Step 1/2] Hyperparameter Tuning")
    results_path = run_tuning(args)

    # Step 2: Training with best params
    print("\n[Step 2/2] Training with Best Parameters")
    args.tuning_results = args.experiment
    args.model_name = f"{args.experiment}_final"
    trainer, history = run_training(args)

    return trainer, history


def list_results(args):
    """List available tuning results."""
    from training.configs.training_config import TrainingConfig

    print("\n" + "=" * 70)
    print("AVAILABLE TUNING RESULTS")
    print("=" * 70)

    results = TrainingConfig.list_tuning_results(args.output_dir)

    if not results:
        print("\nNo tuning results found.")
        print(f"Run tuning first: python scripts/train_pipeline.py tune --dataset <name> --experiment <name>")
        return

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['experiment_name']}")
        print(f"   Score: {r['best_score']:.4f}")
        print(f"   Method: {r['tuning_method']}")
        print(f"   Best params: {r['best_params']}")
        print(f"   Path: {r['path']}")


def main():
    parser = argparse.ArgumentParser(
        description='Training Pipeline - Hyperparameter Tuning & Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Run hyperparameter tuning
    python scripts/train_pipeline.py tune --dataset my_music --experiment lstm_v1

    # Train with tuned parameters
    python scripts/train_pipeline.py train --dataset my_music --tuning-results lstm_v1 --model-name lstm_v1_final

    # Full pipeline (tune + train)
    python scripts/train_pipeline.py full --dataset my_music --experiment lstm_v1

    # List tuning results
    python scripts/train_pipeline.py list
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--dataset', type=str, help='Dataset name (h5 file without extension)')
    common_parser.add_argument('--config', type=str, help='Path to tensorflow_dataset_config JSON file')
    common_parser.add_argument('--output-dir', type=str, default='./models', help='Output directory')
    common_parser.add_argument('--representation', type=str, default='piano-roll',
                              choices=['piano-roll', 'pitch', 'event', 'note'],
                              help='Music representation type (saved with model for generation)')
    common_parser.add_argument('--tensor-type', type=str, default='full',
                              choices=['music-only', 'music-genre', 'music-instrument', 'full'],
                              help='What metadata to include')
    common_parser.add_argument('--train-split', type=float, default=0.8, help='Training split ratio')
    common_parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    common_parser.add_argument('--test-split', type=float, default=0.1, help='Test split ratio')
    common_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (reduce for large sequences)')

    # Tune command
    tune_parser = subparsers.add_parser('tune', parents=[common_parser], help='Run hyperparameter tuning')
    tune_parser.add_argument('--experiment', type=str, required=True, help='Experiment name for saving results')
    tune_parser.add_argument('--method', type=str, default='random_search', choices=['grid_search', 'random_search'],
                            help='Search method')
    tune_parser.add_argument('--cv-folds', type=int, default=3, help='Cross-validation folds')
    tune_parser.add_argument('--n-iter', type=int, default=20, help='Number of iterations (random search)')
    tune_parser.add_argument('--n-jobs', type=int, default=1, help='Parallel jobs (-1 for all cores)')
    tune_parser.add_argument('--max-samples', type=int, default=2000, help='Max samples for tuning')
    tune_parser.add_argument('--tuning-epochs', type=int, default=10, help='Epochs per tuning trial')
    tune_parser.add_argument('--scoring', type=str, default='neg_mean_squared_error', help='Scoring metric')
    tune_parser.add_argument('--param-grid', type=str, default=None, help='Custom param grid (JSON string)')

    # Train command
    train_parser = subparsers.add_parser('train', parents=[common_parser], help='Train model')
    train_parser.add_argument('--model-name', type=str, required=True, help='Name for the trained model')
    train_parser.add_argument('--tuning-results', type=str, help='Tuning experiment name to load params from')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')

    # Full pipeline command
    full_parser = subparsers.add_parser('full', parents=[common_parser], help='Run full pipeline (tune + train)')
    full_parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    full_parser.add_argument('--method', type=str, default='random_search', choices=['grid_search', 'random_search'])
    full_parser.add_argument('--cv-folds', type=int, default=3)
    full_parser.add_argument('--n-iter', type=int, default=20)
    full_parser.add_argument('--n-jobs', type=int, default=1)
    full_parser.add_argument('--max-samples', type=int, default=2000)
    full_parser.add_argument('--tuning-epochs', type=int, default=10)
    full_parser.add_argument('--epochs', type=int, default=100, help='Final training epochs')
    full_parser.add_argument('--scoring', type=str, default='neg_mean_squared_error')
    full_parser.add_argument('--param-grid', type=str, default=None)

    # List command
    list_parser = subparsers.add_parser('list', help='List tuning results')
    list_parser.add_argument('--output-dir', type=str, default='./models')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Run appropriate command
    if args.command == 'tune':
        if not args.dataset and not args.config:
            print("Error: --dataset or --config is required for tuning")
            return
        run_tuning(args)

    elif args.command == 'train':
        if not args.dataset and not args.config:
            print("Error: --dataset or --config is required for training")
            return
        run_training(args)

    elif args.command == 'full':
        if not args.dataset and not args.config:
            print("Error: --dataset or --config is required for full pipeline")
            return
        run_full_pipeline(args)

    elif args.command == 'list':
        list_results(args)


if __name__ == '__main__':
    main()
