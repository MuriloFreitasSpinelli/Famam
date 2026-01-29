"""
Command Line Interface for the music generation pipeline.

Usage:
    python -m src_v4.client.cli <command> [options]

Commands:
    config      Create, view, or edit configuration files
    dataset     Build or inspect datasets
    train       Train a model
    generate    Generate music
    info        Show information about models or datasets
    interactive Run interactive mode
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Any

from .pipeline import MusicPipeline, PipelineConfig
from ..data_preprocessing.config import MusicDatasetConfig
from ..model_training.config import TrainingConfig


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="music-gen",
        description="Music Generation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === Config command ===
    config_parser = subparsers.add_parser("config", help="Manage configuration files")
    config_sub = config_parser.add_subparsers(dest="config_action")

    # config create
    config_create = config_sub.add_parser("create", help="Create a new config file")
    config_create.add_argument("-o", "--output", default="config.json", help="Output file path")
    config_create.add_argument("--encoder", choices=["event", "remi"], default="event")
    config_create.add_argument("--model", choices=["transformer", "lstm"], default="transformer")
    config_create.add_argument("--name", default="music_model", help="Model name")
    config_create.add_argument("--d-model", type=int, default=256)
    config_create.add_argument("--layers", type=int, default=4)
    config_create.add_argument("--heads", type=int, default=8)
    config_create.add_argument("--epochs", type=int, default=100)
    config_create.add_argument("--batch-size", type=int, default=32)
    config_create.add_argument("--seq-length", type=int, default=1024)

    # config show
    config_show = config_sub.add_parser("show", help="Show config contents")
    config_show.add_argument("file", help="Config file to show")

    # config edit
    config_edit = config_sub.add_parser("edit", help="Edit a config value")
    config_edit.add_argument("file", help="Config file to edit")
    config_edit.add_argument("key", help="Key to edit (e.g., epochs, d_model)")
    config_edit.add_argument("value", help="New value")

    # config wizard
    config_wizard = config_sub.add_parser("wizard", help="Interactive config creation wizard")
    config_wizard.add_argument(
        "type",
        choices=["dataset", "training"],
        help="Type of config to create (dataset or training)"
    )

    # === Dataset command ===
    dataset_parser = subparsers.add_parser("dataset", help="Build or inspect datasets")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_action")

    # dataset build
    dataset_build = dataset_sub.add_parser("build", help="Build dataset from MIDI files")
    dataset_build.add_argument("midi_dir", help="Directory containing MIDI files")
    dataset_build.add_argument("-o", "--output", required=True, help="Output .h5 file path")
    dataset_build.add_argument("-c", "--config", help="Config file to use")
    dataset_build.add_argument("--encoder", choices=["event", "remi"], default="event")
    dataset_build.add_argument("--resolution", type=int, default=24)
    dataset_build.add_argument("--seq-length", type=int, default=1024)

    # dataset info
    dataset_info = dataset_sub.add_parser("info", help="Show dataset information")
    dataset_info.add_argument("file", help="Dataset .h5 file")

    # dataset instruments
    dataset_instruments = dataset_sub.add_parser("instruments", help="Show most used instruments in dataset")
    dataset_instruments.add_argument("file", help="Dataset .h5 file")
    dataset_instruments.add_argument("-n", "--top", type=int, default=20, help="Number of top instruments to show")

    # dataset from-config
    dataset_from_config = dataset_sub.add_parser("from-config", help="Build dataset from a config file")
    dataset_from_config.add_argument("config", help="Dataset config JSON file")

    # === Train command ===
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_sub = train_parser.add_subparsers(dest="train_action")

    # train run (default behavior)
    train_run = train_sub.add_parser("run", help="Train with options")
    train_run.add_argument("dataset", help="Dataset .h5 file")
    train_run.add_argument("-c", "--config", help="Config file (or uses defaults)")
    train_run.add_argument("-o", "--output", help="Output model path")
    train_run.add_argument("--epochs", type=int, help="Override epochs")
    train_run.add_argument("--batch-size", type=int, help="Override batch size")
    train_run.add_argument("--resume", help="Resume from checkpoint")

    # train from-config
    train_from_config = train_sub.add_parser("from-config", help="Train model using config file")
    train_from_config.add_argument("dataset", help="Dataset .h5 file")
    train_from_config.add_argument("config", help="Training config JSON file")
    train_from_config.add_argument("--epochs", type=int, help="Override epochs from config")
    train_from_config.add_argument("--batch-size", type=int, help="Override batch size from config")
    train_from_config.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")

    # === Generate command ===
    gen_parser = subparsers.add_parser("generate", help="Generate music")
    gen_parser.add_argument("model", help="Model bundle .h5 file")
    gen_parser.add_argument("-o", "--output", required=True, help="Output MIDI file")
    gen_parser.add_argument("--genre", type=int, default=0, help="Genre ID")
    gen_parser.add_argument("--instruments", type=int, nargs="+", help="Instrument IDs")
    gen_parser.add_argument("--no-drums", action="store_true", help="Exclude drums")
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument("--top-k", type=int, default=50)
    gen_parser.add_argument("--top-p", type=float, default=0.9)
    gen_parser.add_argument("--length", type=int, help="Max sequence length")
    gen_parser.add_argument("--min-length", type=int, help="Min length before allowing EOS")
    gen_parser.add_argument("--ignore-eos", action="store_true", help="Don't stop at EOS, generate full length")

    # === Info command ===
    info_parser = subparsers.add_parser("info", help="Show model or dataset info")
    info_parser.add_argument("file", help="Model or dataset file")

    # === Interactive command ===
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive mode")
    interactive_parser.add_argument("-c", "--config", help="Config file to load")

    # === Menu command ===
    menu_parser = subparsers.add_parser("menu", help="Run menu-driven interface")

    return parser


# =============================================================================
# Interactive Config Wizard Helpers
# =============================================================================

# Default config directories
CONFIG_BASE_DIR = Path(__file__).parent.parent.parent / "configs"
DATASET_CONFIG_DIR = CONFIG_BASE_DIR / "music_dataset"
TRAINING_CONFIG_DIR = CONFIG_BASE_DIR / "model_training"


def prompt_input(prompt: str, default: Any = None, value_type: type = str) -> Any:
    """Prompt user for input with a default value."""
    if default is not None:
        prompt_str = f"{prompt} [{default}]: "
    else:
        prompt_str = f"{prompt}: "

    try:
        value = input(prompt_str).strip()
        if not value:
            return default

        # Convert to appropriate type
        if value_type == bool:
            return value.lower() in ('true', '1', 'yes', 'y')
        elif value_type == int:
            return int(value)
        elif value_type == float:
            return float(value)
        elif value_type == list:
            # Comma-separated values
            return [v.strip() for v in value.split(',') if v.strip()]
        else:
            return value
    except (ValueError, EOFError):
        return default


def prompt_choice(prompt: str, choices: List[str], default: str = None) -> str:
    """Prompt user to select from choices."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")

    while True:
        try:
            value = input("Enter choice (number or name): ").strip()
            if not value and default:
                return default
            # Try as number
            if value.isdigit():
                idx = int(value) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            # Try as name
            if value in choices:
                return value
            print(f"Invalid choice. Please enter 1-{len(choices)} or a valid option name.")
        except (ValueError, EOFError):
            if default:
                return default
            continue


def create_dataset_config_wizard() -> Optional[MusicDatasetConfig]:
    """Interactive wizard for creating a MusicDatasetConfig."""
    print("\n" + "=" * 60)
    print("  Dataset Configuration Wizard")
    print("=" * 60)
    print("Press Enter to accept default values.\n")

    # Basic info
    name = prompt_input("Config name", "my_dataset", str)

    # Input directories
    print("\nEnter MIDI input directories (comma-separated):")
    input_dirs_str = prompt_input("Input directories", "", str)
    input_dirs = [d.strip() for d in input_dirs_str.split(',') if d.strip()] if input_dirs_str else []

    output_path = prompt_input("Dataset output path (.h5)", f"data/datasets/{name}.h5", str)

    # Genre filtering
    print("\n--- Genre Filtering ---")
    genre_tsv_path = prompt_input("Genre TSV file path (optional)", None, str)
    allowed_genres_str = prompt_input("Allowed genres (comma-separated, leave empty for all)", "", str)
    allowed_genres = [g.strip() for g in allowed_genres_str.split(',') if g.strip()] if allowed_genres_str else None

    # Track filtering
    print("\n--- Track Filtering ---")
    min_tracks = prompt_input("Minimum tracks per file", 1, int)
    max_tracks = prompt_input("Maximum tracks per file", 16, int)
    min_notes_per_track = prompt_input("Minimum notes per track", 1, int)

    # Duration filtering
    print("\n--- Duration Filtering ---")
    min_duration = prompt_input("Minimum duration (seconds, leave empty for none)", None, float)
    max_duration = prompt_input("Maximum duration (seconds, leave empty for none)", None, float)

    # Resolution & timing
    print("\n--- Resolution & Timing ---")
    resolution = prompt_input("Ticks per quarter note", 24, int)

    # Encoding
    print("\n--- Encoding ---")
    encoder_type = prompt_choice("Encoder type:", ["event", "remi"], "remi")
    max_seq_length = prompt_input("Max sequence length", 2048, int)
    encode_velocity = prompt_input("Encode velocity (true/false)", True, bool)

    # Segmentation
    print("\n--- Segmentation ---")
    use_segmentation = prompt_input("Enable segmentation (true/false)", False, bool)
    segment_length = None
    if use_segmentation:
        segment_length = prompt_input("Segment length (time steps)", 512, int)

    # Augmentation
    print("\n--- Augmentation ---")
    enable_transposition = prompt_input("Enable transposition augmentation (true/false)", False, bool)
    enable_tempo_variation = prompt_input("Enable tempo variation (true/false)", False, bool)

    # Processing
    print("\n--- Processing ---")
    max_samples = prompt_input("Max samples (leave empty for unlimited)", None, int)
    random_seed = prompt_input("Random seed (leave empty for none)", None, int)

    try:
        config = MusicDatasetConfig(
            name=name,
            input_dirs=input_dirs,
            output_path=output_path,
            genre_tsv_path=genre_tsv_path,
            allowed_genres=allowed_genres,
            min_tracks=min_tracks,
            max_tracks=max_tracks,
            min_notes_per_track=min_notes_per_track,
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            resolution=resolution,
            encoder_type=encoder_type,
            max_seq_length=max_seq_length,
            encode_velocity=encode_velocity,
            segment_length=segment_length,
            enable_transposition=enable_transposition,
            enable_tempo_variation=enable_tempo_variation,
            max_samples=max_samples,
            random_seed=random_seed,
        )
        return config
    except ValueError as e:
        print(f"\nError: {e}")
        return None


def create_training_config_wizard() -> Optional[TrainingConfig]:
    """Interactive wizard for creating a TrainingConfig."""
    print("\n" + "=" * 60)
    print("  Training Configuration Wizard")
    print("=" * 60)
    print("Press Enter to accept default values.\n")

    # Model identification
    model_name = prompt_input("Model name", "my_model", str)
    model_type = prompt_choice("Model type:", ["transformer", "lstm"], "transformer")

    # Shared architecture
    print("\n--- Model Architecture ---")
    max_seq_length = prompt_input("Max sequence length", 2048, int)
    d_model = prompt_input("Embedding dimension (d_model)", 512, int)
    dropout_rate = prompt_input("Dropout rate", 0.1, float)

    # Architecture-specific
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    use_relative_attention = True
    lstm_units = (512, 512)
    bidirectional = False

    if model_type == "transformer":
        print("\n--- Transformer Architecture ---")
        num_layers = prompt_input("Number of layers", 6, int)
        num_heads = prompt_input("Number of attention heads", 8, int)
        d_ff = prompt_input("Feed-forward dimension (d_ff)", 2048, int)
        use_relative_attention = prompt_input("Use relative attention (true/false)", True, bool)
    else:
        print("\n--- LSTM Architecture ---")
        lstm_units_str = prompt_input("LSTM units (comma-separated, e.g., 512,512)", "512,512", str)
        lstm_units = tuple(int(u.strip()) for u in lstm_units_str.split(','))
        bidirectional = prompt_input("Bidirectional (true/false)", False, bool)

    # Training hyperparameters
    print("\n--- Training Hyperparameters ---")
    batch_size = prompt_input("Batch size", 16, int)
    epochs = prompt_input("Epochs", 100, int)
    learning_rate = prompt_input("Learning rate", 1e-4, float)
    warmup_steps = prompt_input("Warmup steps", 4000, int)
    label_smoothing = prompt_input("Label smoothing", 0.1, float)

    # Optimizer
    print("\n--- Optimizer ---")
    optimizer = prompt_choice("Optimizer:", ["adam", "adamw", "sgd", "rmsprop"], "adam")
    weight_decay = prompt_input("Weight decay", 0.01, float)

    # Learning rate schedule
    print("\n--- Learning Rate Schedule ---")
    use_lr_schedule = prompt_input("Use LR schedule (true/false)", True, bool)
    lr_schedule_type = "transformer"
    if use_lr_schedule:
        lr_schedule_type = prompt_choice("LR schedule type:", ["transformer", "cosine", "constant"], "transformer")

    # Regularization
    print("\n--- Regularization ---")
    use_gradient_clipping = prompt_input("Use gradient clipping (true/false)", True, bool)
    gradient_clip_value = 1.0
    if use_gradient_clipping:
        gradient_clip_value = prompt_input("Gradient clip value", 1.0, float)

    # Early stopping
    print("\n--- Early Stopping ---")
    use_early_stopping = prompt_input("Use early stopping (true/false)", True, bool)
    early_stopping_patience = 10
    if use_early_stopping:
        early_stopping_patience = prompt_input("Early stopping patience", 10, int)

    # Checkpointing
    print("\n--- Checkpointing ---")
    use_checkpointing = prompt_input("Save checkpoints (true/false)", True, bool)
    save_best_only = True
    if use_checkpointing:
        save_best_only = prompt_input("Save best only (true/false)", True, bool)

    # TensorBoard
    use_tensorboard = prompt_input("Use TensorBoard (true/false)", True, bool)

    # Output
    print("\n--- Output ---")
    output_dir = prompt_input("Output directory", "./models", str)

    # Random seed
    random_seed = prompt_input("Random seed (leave empty for none)", 42, int)

    try:
        config = TrainingConfig(
            model_name=model_name,
            model_type=model_type,
            max_seq_length=max_seq_length,
            d_model=d_model,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            use_relative_attention=use_relative_attention,
            lstm_units=lstm_units,
            bidirectional=bidirectional,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            label_smoothing=label_smoothing,
            optimizer=optimizer,
            weight_decay=weight_decay,
            use_lr_schedule=use_lr_schedule,
            lr_schedule_type=lr_schedule_type,
            use_gradient_clipping=use_gradient_clipping,
            gradient_clip_value=gradient_clip_value,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=early_stopping_patience,
            use_checkpointing=use_checkpointing,
            save_best_only=save_best_only,
            use_tensorboard=use_tensorboard,
            output_dir=output_dir,
            random_seed=random_seed,
        )
        return config
    except ValueError as e:
        print(f"\nError: {e}")
        return None


# =============================================================================
# Command handlers
# =============================================================================

def cmd_config(args) -> int:
    """Handle config commands."""
    if args.config_action == "create":
        config = PipelineConfig(
            encoder_type=args.encoder,
            model_type=args.model,
            model_name=args.name,
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_seq_length=args.seq_length,
        )
        config.save(args.output)
        print(f"Created config: {args.output}")
        return 0

    elif args.config_action == "show":
        config = PipelineConfig.load(args.file)
        print(json.dumps(config.__dict__, indent=2))
        return 0

    elif args.config_action == "edit":
        config = PipelineConfig.load(args.file)
        if not hasattr(config, args.key):
            print(f"Error: Unknown config key '{args.key}'")
            return 1

        # Convert value to appropriate type
        old_value = getattr(config, args.key)
        if isinstance(old_value, bool):
            new_value = args.value.lower() in ('true', '1', 'yes')
        elif isinstance(old_value, int):
            new_value = int(args.value)
        elif isinstance(old_value, float):
            new_value = float(args.value)
        else:
            new_value = args.value

        setattr(config, args.key, new_value)
        config.save(args.file)
        print(f"Updated {args.key}: {old_value} -> {new_value}")
        return 0

    elif args.config_action == "wizard":
        # Ensure config directories exist
        DATASET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        TRAINING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        if args.type == "dataset":
            config = create_dataset_config_wizard()
            if config is None:
                return 1

            # Prompt for filename
            default_filename = f"{config.name}.json"
            filename = prompt_input("\nConfig filename", default_filename, str)
            if not filename.endswith('.json'):
                filename += '.json'

            output_path = DATASET_CONFIG_DIR / filename
            config.save(str(output_path))
            print(f"\nDataset config saved to: {output_path}")

        elif args.type == "training":
            config = create_training_config_wizard()
            if config is None:
                return 1

            # Prompt for filename
            default_filename = f"{config.model_name}.json"
            filename = prompt_input("\nConfig filename", default_filename, str)
            if not filename.endswith('.json'):
                filename += '.json'

            output_path = TRAINING_CONFIG_DIR / filename
            config.save(str(output_path))
            print(f"\nTraining config saved to: {output_path}")

        return 0

    else:
        print("Usage: music-gen config <create|show|edit|wizard> [options]")
        return 1


def cmd_dataset(args) -> int:
    """Handle dataset commands."""
    if args.dataset_action == "build":
        if args.config:
            config = PipelineConfig.load(args.config)
        else:
            config = PipelineConfig(
                encoder_type=args.encoder,
                resolution=args.resolution,
                max_seq_length=args.seq_length,
            )

        pipeline = MusicPipeline(config)
        pipeline.build_dataset(
            midi_dir=args.midi_dir,
            output_path=args.output,
        )
        print(f"\nDataset saved to: {args.output}")
        return 0

    elif args.dataset_action == "info":
        from ..data_preprocessing import MusicDataset
        dataset = MusicDataset.load(args.file)
        print(f"\nDataset: {args.file}")
        print(f"  Entries: {len(dataset)}")
        print(f"  Genres: {dataset.vocabulary.num_genres}")
        print(f"  Active instruments: {dataset.vocabulary.num_active_instruments}")
        print(f"\nGenres: {list(dataset.vocabulary.genre_to_id.keys())}")
        return 0

    elif args.dataset_action == "instruments":
        from ..data_preprocessing import MusicDataset
        dataset = MusicDataset.load(args.file)
        dataset.print_instrument_stats(top_n=args.top)
        return 0

    elif args.dataset_action == "from-config":
        from ..data_preprocessing.scripts.dataset_builder import build_and_save_dataset

        print(f"\nLoading config from: {args.config}")
        config = MusicDatasetConfig.load(args.config)

        print(f"\nBuilding dataset: {config.name}")
        print(f"  Input directories: {config.input_dirs}")
        print(f"  Output path: {config.output_path}")
        print(f"  Encoder: {config.encoder_type}")

        dataset = build_and_save_dataset(config)

        print(f"\nDataset built successfully!")
        print(f"  Entries: {len(dataset)}")
        print(f"  Output: {config.output_path}")
        return 0

    else:
        print("Usage: music-gen dataset <build|info|instruments|from-config> [options]")
        return 1


def cmd_train(args) -> int:
    """Handle train command."""
    if args.train_action == "run" or args.train_action is None:
        # Original train behavior
        if hasattr(args, 'config') and args.config:
            config = PipelineConfig.load(args.config)
        else:
            config = PipelineConfig()

        # Override with command line args
        if hasattr(args, 'epochs') and args.epochs:
            config.epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            config.batch_size = args.batch_size

        pipeline = MusicPipeline(config)
        pipeline.load_dataset(args.dataset)

        print(f"\nTraining {config.model_type} model...")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")

        pipeline.train()

        output_path = args.output or f"models/{config.model_name}"
        pipeline.save_model(output_path)
        print(f"\nModel saved to: {output_path}")

        return 0

    elif args.train_action == "from-config":
        from ..data_preprocessing import MusicDataset
        from ..data_preprocessing.encoders import MultiTrackEncoder
        from ..model_training.pipeline.trainer import Trainer

        # Load dataset
        print(f"\nLoading dataset from: {args.dataset}")
        dataset = MusicDataset.load(args.dataset)
        print(f"  Entries: {len(dataset)}")
        print(f"  Genres: {dataset.vocabulary.num_genres}")
        print(f"  Tracks: {dataset.count_tracks()}")

        # Load training config
        print(f"\nLoading training config from: {args.config}")
        config = TrainingConfig.load(args.config)

        # Override with command line args
        if args.epochs:
            config.epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size

        print(config.summary())

        # Load dataset config to get encoder settings
        dataset_config_path = Path(args.dataset).with_suffix('.config.json')
        if dataset_config_path.exists():
            dataset_config = MusicDatasetConfig.load(str(dataset_config_path))
            resolution = dataset_config.resolution
            positions_per_bar = dataset_config.positions_per_bar
        else:
            resolution = dataset.resolution
            positions_per_bar = 16

        # Create MultiTrackEncoder for multitrack training
        encoder = MultiTrackEncoder(
            num_genres=max(1, dataset.vocabulary.num_genres),
            resolution=resolution,
            positions_per_bar=positions_per_bar,
        )

        print(f"\nEncoder: {type(encoder).__name__}")
        print(f"  Vocab size: {encoder.vocab_size}")
        print(f"  Resolution: {resolution}")
        print(f"  Positions per bar: {positions_per_bar}")

        # Prepare data for training with train/val split
        print(f"\nPreparing training data...")
        val_ratio = args.val_split
        train_ratio = 1.0 - val_ratio

        datasets = dataset.to_multitrack_dataset(
            encoder=encoder,
            splits=(train_ratio, val_ratio, 0.0),  # train, val, test
            random_state=42,
            min_tracks=2,
        )

        train_dataset = datasets['train']
        val_dataset = datasets['validation']

        # Count samples (estimate based on entries)
        total_entries = len(dataset)
        train_count = int(total_entries * train_ratio)
        val_count = total_entries - train_count

        print(f"  Train samples: ~{train_count}")
        print(f"  Validation samples: ~{val_count}")

        # Train
        trainer = Trainer(config, encoder)
        trainer.build_model()
        model, history = trainer.train(train_dataset, val_dataset)

        # Save as model bundle (.h5)
        print(f"\nSaving model bundle...")
        try:
            from ..model_training.model_bundle import ModelBundle
            output_dir = Path(config.output_dir) / config.model_name
            bundle_path = output_dir / f"{config.model_name}.h5"

            print(f"  Creating bundle at: {bundle_path}")
            bundle = ModelBundle(
                model=trainer.model,  # Use trainer.model directly
                encoder=encoder,
                config=config,
                model_name=config.model_name,
            )
            bundle.save(bundle_path)

            print(f"\nTraining complete!")
            print(f"  Model bundle saved to: {bundle_path}")
        except Exception as e:
            print(f"\nError saving model bundle: {e}")
            import traceback
            traceback.print_exc()
            return 1

        return 0

    else:
        print("Usage: music-gen train <run|from-config> [options]")
        return 1


def cmd_generate(args) -> int:
    """Handle generate command."""
    from ..model_training.model_bundle import load_model_bundle
    from ..data_preprocessing.encoders import MultiTrackEncoder
    import muspy

    # Load model bundle
    print(f"\nLoading model from: {args.model}")
    bundle = load_model_bundle(args.model)
    print(bundle.summary())

    # Generate tokens
    print(f"\nGenerating music...")
    print(f"  Genre ID: {args.genre}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")

    max_length = args.length or bundle.max_seq_length
    min_length = args.min_length
    ignore_eos = args.ignore_eos

    # Enforce model's max sequence length
    if max_length > bundle.max_seq_length:
        print(f"  Warning: Capping length to model max ({bundle.max_seq_length})")
        max_length = bundle.max_seq_length

    if min_length and min_length > max_length:
        min_length = max_length

    if min_length:
        print(f"  Min length: {min_length}")
    if ignore_eos:
        print(f"  Ignoring EOS (will generate full {max_length} tokens)")

    tokens = bundle.generate(
        genre_id=args.genre,
        max_length=max_length,
        min_length=min_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        ignore_eos=ignore_eos,
    )

    print(f"  Generated {len(tokens)} tokens")

    # Decode tokens to MIDI
    if isinstance(bundle.encoder, MultiTrackEncoder):
        music = bundle.encoder.decode_to_music(tokens)
    else:
        # For single-track encoders
        events = bundle.decode_tokens(tokens)
        music = bundle.encoder.decode_to_music(tokens)

    # Filter drums if requested
    if args.no_drums:
        music.tracks = [t for t in music.tracks if not t.is_drum]

    # Save MIDI
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    muspy.write_midi(output_path, music)

    print(f"\nSaved to: {args.output}")
    print(f"  Tracks: {len(music.tracks)}")
    for i, track in enumerate(music.tracks):
        name = "Drums" if track.is_drum else f"Program {track.program}"
        print(f"    {i}: {name} ({len(track.notes)} notes)")

    return 0


def cmd_info(args) -> int:
    """Handle info command."""
    filepath = Path(args.file)

    if not filepath.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    # Try to load as model bundle first
    try:
        from ..model_training import load_model_bundle
        bundle = load_model_bundle(args.file)
        print(bundle.summary())
        return 0
    except Exception:
        pass

    # Try as dataset
    try:
        from ..data_preprocessing import MusicDataset
        dataset = MusicDataset.load(args.file)
        print(f"\nDataset: {args.file}")
        print(f"  Entries: {len(dataset)}")
        print(f"  Genres: {dataset.vocabulary.num_genres}")
        print(f"  Active instruments: {dataset.vocabulary.num_active_instruments}")
        return 0
    except Exception:
        pass

    # Try as config
    try:
        config = PipelineConfig.load(args.file)
        print(json.dumps(config.__dict__, indent=2))
        return 0
    except Exception:
        pass

    print(f"Error: Could not identify file type: {args.file}")
    return 1


def cmd_interactive(args) -> int:
    """Run interactive mode."""
    if args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig()

    shell = InteractiveShell(config)
    shell.run()
    return 0


def cmd_menu(args) -> int:
    """Run menu-driven interface."""
    from .menu_cli import MenuCLI
    cli = MenuCLI()
    cli.run()
    return 0


# =============================================================================
# Interactive Shell
# =============================================================================

class InteractiveShell:
    """Interactive command shell for the music pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline = MusicPipeline(config)
        self.running = True

    def run(self):
        """Run the interactive shell."""
        print("\n" + "=" * 60)
        print("  Music Generation Pipeline - Interactive Mode")
        print("=" * 60)
        print("Type 'help' for available commands, 'quit' to exit.\n")

        while self.running:
            try:
                line = input("music> ").strip()
                if not line:
                    continue
                self.execute(line)
            except KeyboardInterrupt:
                print("\n(Use 'quit' to exit)")
            except EOFError:
                self.running = False

        print("\nGoodbye!")

    def execute(self, line: str):
        """Execute a command."""
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        commands = {
            "help": self.cmd_help,
            "quit": self.cmd_quit,
            "exit": self.cmd_quit,
            "status": self.cmd_status,
            "config": self.cmd_config,
            "set": self.cmd_set,
            "save": self.cmd_save,
            "load": self.cmd_load,
            "dataset": self.cmd_dataset,
            "train": self.cmd_train,
            "generate": self.cmd_generate,
            "genres": self.cmd_genres,
            "instruments": self.cmd_instruments,
            "create": self.cmd_create_config,
            "build": self.cmd_build_from_config,
            "train-config": self.cmd_train_from_config,
            "gen": self.cmd_gen_from_bundle,
        }

        if cmd in commands:
            try:
                commands[cmd](args)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")

    def cmd_help(self, args):
        """Show help."""
        print("""
Available commands:

  status              Show current pipeline status
  config              Show current configuration
  set <key> <value>   Set a config value
  save config <file>  Save config to file
  save model <file>   Save trained model
  load config <file>  Load config from file
  load dataset <file> Load dataset from file
  load model <file>   Load model from file

  dataset build <dir> [output]  Build dataset from MIDI directory
  dataset info                  Show dataset info

  train [epochs]      Train the model

  generate <output> [options]   Generate music
    --genre <id>                Genre ID (default: 0)
    --instruments <ids>         Instrument IDs (space separated)
    --temp <float>              Temperature (default: 1.0)
    --no-drums                  Exclude drums

  genres              List available genres
  instruments         List available instruments (MIDI programs)
  instruments stats [N]  Show top N most used instruments in dataset

  create dataset      Create a new dataset config (interactive wizard)
  create training     Create a new training config (interactive wizard)

  build <config>      Build dataset from config file
  train-config <dataset> <config>  Train model using config files
  gen [model.h5] [output.mid]
                      Generate music (prompts for all options interactively)

  quit / exit         Exit interactive mode
""")

    def cmd_quit(self, args):
        """Exit the shell."""
        self.running = False

    def cmd_status(self, args):
        """Show pipeline status."""
        print(self.pipeline.summary())

    def cmd_config(self, args):
        """Show current config."""
        print(json.dumps(self.config.__dict__, indent=2))

    def cmd_set(self, args):
        """Set a config value."""
        if len(args) < 2:
            print("Usage: set <key> <value>")
            return

        key, value = args[0], " ".join(args[1:])

        if not hasattr(self.config, key):
            print(f"Unknown config key: {key}")
            return

        old_value = getattr(self.config, key)
        if isinstance(old_value, bool):
            new_value = value.lower() in ('true', '1', 'yes')
        elif isinstance(old_value, int):
            new_value = int(value)
        elif isinstance(old_value, float):
            new_value = float(value)
        else:
            new_value = value

        setattr(self.config, key, new_value)
        self.pipeline.config = self.config
        print(f"Set {key} = {new_value}")

    def cmd_save(self, args):
        """Save config or model."""
        if len(args) < 2:
            print("Usage: save <config|model> <file>")
            return

        what, filepath = args[0], args[1]

        if what == "config":
            self.config.save(filepath)
            print(f"Saved config to: {filepath}")
        elif what == "model":
            path = self.pipeline.save_model(filepath)
            print(f"Saved model to: {path}")
        else:
            print("Usage: save <config|model> <file>")

    def cmd_load(self, args):
        """Load config, dataset, or model."""
        if len(args) < 2:
            print("Usage: load <config|dataset|model> <file>")
            return

        what, filepath = args[0], args[1]

        if what == "config":
            self.config = PipelineConfig.load(filepath)
            self.pipeline = MusicPipeline(self.config)
            print(f"Loaded config from: {filepath}")
        elif what == "dataset":
            self.pipeline.load_dataset(filepath)
            print(f"Loaded dataset: {len(self.pipeline.dataset)} entries")
        elif what == "model":
            self.pipeline.load_model(filepath)
            print(f"Loaded model: {self.pipeline.bundle.model_name}")
        else:
            print("Usage: load <config|dataset|model> <file>")

    def cmd_dataset(self, args):
        """Dataset commands."""
        if not args:
            print("Usage: dataset <build|info> [args]")
            return

        action = args[0]

        if action == "build":
            if len(args) < 2:
                print("Usage: dataset build <midi_dir> [output.h5]")
                return
            midi_dir = args[1]
            output = args[2] if len(args) > 2 else "dataset.h5"
            self.pipeline.build_dataset(midi_dir=midi_dir, output_path=output)
            print(f"Dataset built: {len(self.pipeline.dataset)} entries")

        elif action == "info":
            if self.pipeline.dataset is None:
                print("No dataset loaded. Use 'load dataset <file>' first.")
                return
            print(f"Entries: {len(self.pipeline.dataset)}")
            print(f"Genres: {self.pipeline.vocabulary.num_genres}")
            print(f"Active instruments: {self.pipeline.vocabulary.num_active_instruments}")

    def cmd_train(self, args):
        """Train the model."""
        if self.pipeline.dataset is None and self.pipeline.encoder is None:
            print("No dataset loaded. Use 'load dataset <file>' or 'dataset build <dir>' first.")
            return

        epochs = int(args[0]) if args else None
        self.pipeline.train(epochs=epochs)
        print("Training complete.")

    def cmd_generate(self, args):
        """Generate music."""
        if self.pipeline.model is None:
            print("No model loaded. Use 'load model <file>' or 'train' first.")
            return

        if not args:
            print("Usage: generate <output.mid> [--genre <id>] [--temp <float>] [--no-drums]")
            return

        output = args[0]
        genre_id = 0
        temperature = 1.0
        instruments = None
        include_drums = True

        i = 1
        while i < len(args):
            if args[i] == "--genre" and i + 1 < len(args):
                genre_id = int(args[i + 1])
                i += 2
            elif args[i] == "--temp" and i + 1 < len(args):
                temperature = float(args[i + 1])
                i += 2
            elif args[i] == "--instruments" and i + 1 < len(args):
                # Collect all following integers
                instruments = []
                i += 1
                while i < len(args) and not args[i].startswith("--"):
                    instruments.append(int(args[i]))
                    i += 1
            elif args[i] == "--no-drums":
                include_drums = False
                i += 1
            else:
                i += 1

        self.pipeline.generate_midi(
            output_path=output,
            genre_id=genre_id,
            instrument_ids=instruments,
            include_drums=include_drums,
            temperature=temperature,
        )
        print(f"Generated: {output}")

    def cmd_genres(self, args):
        """List available genres."""
        genres = self.pipeline.list_genres()
        if not genres:
            print("No genres available. Load a dataset or model first.")
            return
        print("Available genres:")
        for i, genre in enumerate(genres):
            print(f"  {i}: {genre}")

    def cmd_instruments(self, args):
        """List or show stats for instruments."""
        if args and args[0] == "stats":
            # Show instrument usage statistics
            if self.pipeline.dataset is None:
                print("No dataset loaded. Use 'load dataset <file>' first.")
                return
            top_n = int(args[1]) if len(args) > 1 else 20
            self.pipeline.print_instrument_stats(top_n)
        else:
            # List available instruments
            instruments = self.pipeline.list_instruments()
            if not instruments:
                print("No instruments available. Load a dataset or model first.")
                return
            print("Available instruments:")
            for i, inst in enumerate(instruments[:20]):  # Show first 20
                print(f"  {i}: {inst}")
            if len(instruments) > 20:
                print(f"  ... and {len(instruments) - 20} more")
            print("\nUse 'instruments stats [N]' to see top N used instruments.")

    def cmd_create_config(self, args):
        """Create a new configuration file interactively."""
        if not args:
            print("Usage: create <dataset|training>")
            return

        config_type = args[0].lower()

        # Ensure directories exist
        DATASET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        TRAINING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        if config_type == "dataset":
            config = create_dataset_config_wizard()
            if config is None:
                print("Config creation cancelled or failed.")
                return

            default_filename = f"{config.name}.json"
            filename = prompt_input("\nConfig filename", default_filename, str)
            if not filename.endswith('.json'):
                filename += '.json'

            output_path = DATASET_CONFIG_DIR / filename
            config.save(str(output_path))
            print(f"\nDataset config saved to: {output_path}")

        elif config_type == "training":
            config = create_training_config_wizard()
            if config is None:
                print("Config creation cancelled or failed.")
                return

            default_filename = f"{config.model_name}.json"
            filename = prompt_input("\nConfig filename", default_filename, str)
            if not filename.endswith('.json'):
                filename += '.json'

            output_path = TRAINING_CONFIG_DIR / filename
            config.save(str(output_path))
            print(f"\nTraining config saved to: {output_path}")

        else:
            print("Unknown config type. Use 'create dataset' or 'create training'.")

    def cmd_build_from_config(self, args):
        """Build dataset from a config file."""
        if not args:
            print("Usage: build <config_file>")
            print("\nAvailable configs in configs/music_dataset/:")
            for f in DATASET_CONFIG_DIR.glob("*.json"):
                print(f"  {f.name}")
            return

        config_path = args[0]
        # If just a filename, look in the default config dir
        if not Path(config_path).exists():
            config_path = DATASET_CONFIG_DIR / config_path
            if not config_path.exists() and not str(args[0]).endswith('.json'):
                config_path = DATASET_CONFIG_DIR / f"{args[0]}.json"

        if not Path(config_path).exists():
            print(f"Config file not found: {config_path}")
            return

        from ..data_preprocessing.scripts.dataset_builder import build_and_save_dataset

        print(f"\nLoading config from: {config_path}")
        config = MusicDatasetConfig.load(str(config_path))

        print(f"\nBuilding dataset: {config.name}")
        print(f"  Input directories: {config.input_dirs}")
        print(f"  Output path: {config.output_path}")
        print(f"  Encoder: {config.encoder_type}")

        dataset = build_and_save_dataset(config)

        print(f"\nDataset built successfully!")
        print(f"  Entries: {len(dataset)}")
        print(f"  Output: {config.output_path}")

    def cmd_train_from_config(self, args):
        """Train model using dataset and training config files."""
        if len(args) < 2:
            print("Usage: train-config <dataset.h5> <training_config.json>")
            print("\nAvailable training configs in configs/model_training/:")
            for f in TRAINING_CONFIG_DIR.glob("*.json"):
                print(f"  {f.name}")
            return

        dataset_path = args[0]
        config_path = args[1]

        # If config is just a filename, look in the default config dir
        if not Path(config_path).exists():
            config_path = TRAINING_CONFIG_DIR / config_path
            if not config_path.exists() and not str(args[1]).endswith('.json'):
                config_path = TRAINING_CONFIG_DIR / f"{args[1]}.json"

        if not Path(dataset_path).exists():
            print(f"Dataset file not found: {dataset_path}")
            return

        if not Path(config_path).exists():
            print(f"Config file not found: {config_path}")
            return

        from ..data_preprocessing import MusicDataset
        from ..data_preprocessing.encoders import MultiTrackEncoder
        from ..model_training.pipeline.trainer import Trainer

        # Load dataset
        print(f"\nLoading dataset from: {dataset_path}")
        dataset = MusicDataset.load(dataset_path)
        print(f"  Entries: {len(dataset)}")
        print(f"  Genres: {dataset.vocabulary.num_genres}")
        print(f"  Tracks: {dataset.count_tracks()}")

        # Load training config
        print(f"\nLoading training config from: {config_path}")
        config = TrainingConfig.load(str(config_path))
        print(config.summary())

        # Load dataset config to get encoder settings
        dataset_config_path = Path(dataset_path).with_suffix('.config.json')
        if dataset_config_path.exists():
            dataset_config = MusicDatasetConfig.load(str(dataset_config_path))
            resolution = dataset_config.resolution
            positions_per_bar = dataset_config.positions_per_bar
        else:
            resolution = dataset.resolution
            positions_per_bar = 16

        # Create MultiTrackEncoder for multitrack training
        encoder = MultiTrackEncoder(
            num_genres=max(1, dataset.vocabulary.num_genres),
            resolution=resolution,
            positions_per_bar=positions_per_bar,
        )

        print(f"\nEncoder: {type(encoder).__name__}")
        print(f"  Vocab size: {encoder.vocab_size}")
        print(f"  Resolution: {resolution}")
        print(f"  Positions per bar: {positions_per_bar}")

        # Prepare data with train/val split
        print(f"\nPreparing training data...")
        datasets = dataset.to_multitrack_dataset(
            encoder=encoder,
            splits=(0.9, 0.1, 0.0),  # 90% train, 10% val
            random_state=42,
            min_tracks=2,
        )

        train_dataset = datasets['train']
        val_dataset = datasets['validation']

        # Estimate counts
        total_entries = len(dataset)
        train_count = int(total_entries * 0.9)
        val_count = total_entries - train_count

        print(f"  Train samples: ~{train_count}")
        print(f"  Validation samples: ~{val_count}")

        # Train
        trainer = Trainer(config, encoder)
        trainer.build_model()
        model, history = trainer.train(train_dataset, val_dataset)

        # Save as model bundle (.h5)
        print(f"\nSaving model bundle...")
        try:
            from ..model_training.model_bundle import ModelBundle
            output_dir = Path(config.output_dir) / config.model_name
            bundle_path = output_dir / f"{config.model_name}.h5"

            print(f"  Creating bundle at: {bundle_path}")
            bundle = ModelBundle(
                model=trainer.model,  # Use trainer.model directly
                encoder=encoder,
                config=config,
                model_name=config.model_name,
            )
            bundle.save(bundle_path)

            print(f"\nTraining complete!")
            print(f"  Model bundle saved to: {bundle_path}")
        except Exception as e:
            print(f"\nError saving model bundle: {e}")
            import traceback
            traceback.print_exc()

    def cmd_gen_from_bundle(self, args):
        """Generate music from a model bundle (interactive prompts)."""
        from ..model_training.model_bundle import load_model_bundle
        from ..data_preprocessing.encoders import MultiTrackEncoder
        import muspy

        # Get model path
        if args:
            model_path = args[0]
        else:
            model_path = prompt_input("Model bundle path (.h5)", "models/multitrack_rock/multitrack_rock.h5", str)

        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            return

        # Load model bundle first to show info
        print(f"\nLoading model from: {model_path}")
        bundle = load_model_bundle(model_path)
        print(bundle.summary())

        # Get output path
        if len(args) > 1:
            output_path = args[1]
        else:
            output_path = prompt_input("Output MIDI path", "output/generated.mid", str)

        # Prompt for generation options
        print("\n--- Generation Options ---")
        print(f"  (Model max sequence length: {bundle.max_seq_length})")
        genre_id = prompt_input("Genre ID", 0, int)
        temperature = prompt_input("Temperature (0.1-2.0, higher=more random)", 1.0, float)
        top_k = prompt_input("Top-k sampling (0=disabled)", 50, int)
        top_p = prompt_input("Top-p/nucleus sampling (0.0-1.0)", 0.9, float)
        max_length = prompt_input(f"Max sequence length (max: {bundle.max_seq_length})", bundle.max_seq_length, int)

        # Enforce max length limit
        if max_length > bundle.max_seq_length:
            print(f"  Warning: Capping max_length to model limit ({bundle.max_seq_length})")
            max_length = bundle.max_seq_length

        min_length = prompt_input(f"Min length before EOS (0=no minimum, max: {max_length})", 0, int)
        ignore_eos = prompt_input("Ignore EOS token? (true/false)", False, bool)
        no_drums = prompt_input("Exclude drums? (true/false)", False, bool)

        min_length = min_length if min_length > 0 else None
        if min_length and min_length > max_length:
            min_length = max_length

        # Generate tokens
        print(f"\nGenerating music...")
        print(f"  Genre ID: {genre_id}")
        print(f"  Temperature: {temperature}")
        print(f"  Top-k: {top_k}")
        print(f"  Top-p: {top_p}")
        if min_length:
            print(f"  Min length: {min_length}")
        if ignore_eos:
            print(f"  Ignoring EOS (generating full {max_length} tokens)")

        tokens = bundle.generate(
            genre_id=genre_id,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            ignore_eos=ignore_eos,
        )

        print(f"  Generated {len(tokens)} tokens")

        # Decode tokens to MIDI
        if isinstance(bundle.encoder, MultiTrackEncoder):
            music = bundle.encoder.decode_to_music(tokens)
        else:
            music = bundle.encoder.decode_to_music(tokens)

        # Filter drums if requested
        if no_drums:
            music.tracks = [t for t in music.tracks if not t.is_drum]

        # Save MIDI
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        muspy.write_midi(out_path, music)

        print(f"\nSaved to: {output_path}")
        print(f"  Tracks: {len(music.tracks)}")
        for i, track in enumerate(music.tracks):
            name = "Drums" if track.is_drum else f"Program {track.program}"
            print(f"    {i}: {name} ({len(track.notes)} notes)")


# =============================================================================
# Main entry point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "config": cmd_config,
        "dataset": cmd_dataset,
        "train": cmd_train,
        "generate": cmd_generate,
        "info": cmd_info,
        "interactive": cmd_interactive,
        "menu": cmd_menu,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
