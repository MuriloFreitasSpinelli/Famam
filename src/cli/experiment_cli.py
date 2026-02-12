"""
Experimentation CLI - Menu-based interface for dataset and training workflows.

Features:
    - Interactive config creation (dataset and training)
    - Dataset building from configs
    - Model training with config files
    - Model bundle creation from checkpoints
    - Dataset inspection

Author: Murilo de Freitas Spinelli
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Any

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str, width: int = 60):
    """Print a styled header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_menu(title: str, options: List[str], width: int = 60):
    """Print a menu with numbered options."""
    print_header(title, width)
    for i, desc in enumerate(options, 1):
        print(f"  [{i}] {desc}")
    print("-" * width)
    print("  [0] Back / Exit")
    print("=" * width)


def get_input(prompt: str, default: Any = None) -> str:
    """Get user input with optional default."""
    if default is not None:
        prompt_str = f"{prompt} [{default}]: "
    else:
        prompt_str = f"{prompt}: "
    try:
        value = input(prompt_str).strip()
        return value if value else (str(default) if default is not None else "")
    except (EOFError, KeyboardInterrupt):
        return str(default) if default is not None else ""


def get_int(prompt: str, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    """Get integer input with validation."""
    while True:
        try:
            value = int(get_input(prompt, default) or default)
            if min_val is not None and value < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("  Please enter a valid number")


def get_float(prompt: str, default: float = 0.0, min_val: float = None, max_val: float = None) -> float:
    """Get float input with validation."""
    while True:
        try:
            value = float(get_input(prompt, default) or default)
            if min_val is not None and value < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("  Please enter a valid number")


def get_bool(prompt: str, default: bool = False) -> bool:
    """Get boolean input."""
    default_str = "y" if default else "n"
    value = get_input(f"{prompt} (y/n)", default_str).lower()
    return value in ('y', 'yes', 'true', '1')


def get_choice(prompt: str, max_choice: int) -> int:
    """Get menu choice."""
    while True:
        try:
            choice = int(input(f"\n{prompt}: "))
            if 0 <= choice <= max_choice:
                return choice
            print(f"  Please enter 0-{max_choice}")
        except (ValueError, EOFError, KeyboardInterrupt):
            print("  Please enter a number")


def get_choice_from_list(prompt: str, options: List[str], default: str = None) -> str:
    """Get a choice from a list of options."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt == default else ""
        print(f"  [{i}] {opt}{marker}")

    while True:
        value = get_input("Enter choice (number or name)", default)
        if value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(options):
                return options[idx]
        if value in options:
            return value
        if not value and default:
            return default
        print(f"  Invalid choice. Enter 1-{len(options)} or option name.")


def wait_for_enter():
    """Wait for user to press Enter."""
    input("\nPress Enter to continue...")


# =============================================================================
# Config Directories
# =============================================================================

CONFIG_BASE_DIR = Path(__file__).parent.parent.parent / "configs"
DATASET_CONFIG_DIR = CONFIG_BASE_DIR / "music_dataset"
TRAINING_CONFIG_DIR = CONFIG_BASE_DIR / "model_training"


def ensure_config_dirs():
    """Ensure config directories exist."""
    DATASET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)



def create_dataset_config_wizard():
    """Interactive wizard for creating a MusicDatasetConfig."""
    from ..config import MusicDatasetConfig

    print_header("Dataset Configuration Wizard")
    print("Press Enter to accept default values.\n")

    # Basic info
    name = get_input("Config name", "my_dataset")

    # Input directories
    print("\nEnter MIDI input directories (comma-separated):")
    input_dirs_str = get_input("Input directories", "")
    input_dirs = [d.strip() for d in input_dirs_str.split(',') if d.strip()] if input_dirs_str else []

    output_path = get_input("Dataset output path (.h5)", f"data/datasets/{name}.h5")

    # Genre filtering
    print("\n--- Genre Filtering ---")
    genre_tsv_path = get_input("Genre TSV file path (optional, Enter to skip)", "")
    genre_tsv_path = genre_tsv_path if genre_tsv_path else None

    allowed_genres_str = get_input("Allowed genres (comma-separated, Enter for all)", "")
    allowed_genres = [g.strip() for g in allowed_genres_str.split(',') if g.strip()] if allowed_genres_str else None

    # Track filtering
    print("\n--- Track Filtering ---")
    min_tracks = get_int("Minimum tracks per file", 1, min_val=1)
    max_tracks = get_int("Maximum tracks per file", 16, min_val=1)
    min_notes_per_track = get_int("Minimum notes per track", 1, min_val=0)

    # Duration filtering
    print("\n--- Duration Filtering ---")
    min_duration_str = get_input("Minimum duration (seconds, Enter for none)", "")
    min_duration = float(min_duration_str) if min_duration_str else None

    max_duration_str = get_input("Maximum duration (seconds, Enter for none)", "")
    max_duration = float(max_duration_str) if max_duration_str else None

    # Resolution
    print("\n--- Resolution & Encoding ---")
    resolution = get_int("Ticks per quarter note", 24, min_val=1)
    encoder_type = get_choice_from_list("Encoder type:", ["multitrack", "event", "remi"], "multitrack")
    max_seq_length = get_int("Max sequence length", 2048, min_val=64)
    encode_velocity = get_bool("Encode velocity", True)

    # Segmentation
    print("\n--- Segmentation ---")
    use_segmentation = get_bool("Enable segmentation", False)
    segment_length = None
    if use_segmentation:
        segment_length = get_int("Segment length (ticks)", 512, min_val=64)

    # Augmentation
    print("\n--- Augmentation ---")
    enable_transposition = get_bool("Enable transposition augmentation", False)
    enable_tempo_variation = get_bool("Enable tempo variation", False)

    # Processing
    print("\n--- Processing ---")
    max_samples_str = get_input("Max samples (Enter for unlimited)", "")
    max_samples = int(max_samples_str) if max_samples_str else None

    random_seed_str = get_input("Random seed (Enter for none)", "42")
    random_seed = int(random_seed_str) if random_seed_str else None

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
    except Exception as e:
        print(f"\nError creating config: {e}")
        return None


# =============================================================================
# Training Config Wizard
# =============================================================================

def create_training_config_wizard():
    """Interactive wizard for creating a TrainingConfig."""
    from ..config import TrainingConfig

    print_header("Training Configuration Wizard")
    print("Press Enter to accept default values.\n")

    # Model identification
    model_name = get_input("Model name", "my_model")
    model_type = get_choice_from_list("Model type:", ["transformer", "lstm"], "lstm")

    # Architecture
    print("\n--- Model Architecture ---")
    max_seq_length = get_int("Max sequence length", 2048, min_val=64)
    d_model = get_int("Embedding dimension (d_model)", 256, min_val=32)
    dropout_rate = get_float("Dropout rate", 0.1, min_val=0.0, max_val=0.9)

    # Architecture-specific
    num_layers = 4
    num_heads = 8
    d_ff = 1024
    use_relative_attention = True
    lstm_units = (256, 256)
    bidirectional = False

    if model_type == "transformer":
        print("\n--- Transformer Architecture ---")
        num_layers = get_int("Number of layers", 4, min_val=1)
        num_heads = get_int("Number of attention heads", 8, min_val=1)
        d_ff = get_int("Feed-forward dimension (d_ff)", 1024, min_val=32)
        use_relative_attention = get_bool("Use relative attention", True)
    else:
        print("\n--- LSTM Architecture ---")
        lstm_units_str = get_input("LSTM units (comma-separated)", "256,256")
        lstm_units = tuple(int(u.strip()) for u in lstm_units_str.split(','))
        bidirectional = get_bool("Bidirectional", False)

    # Training hyperparameters
    print("\n--- Training Hyperparameters ---")
    batch_size = get_int("Batch size", 16, min_val=1)
    epochs = get_int("Epochs", 100, min_val=1)
    learning_rate = get_float("Learning rate", 1e-4, min_val=1e-8)
    warmup_steps = get_int("Warmup steps", 4000, min_val=0)
    label_smoothing = get_float("Label smoothing", 0.1, min_val=0.0, max_val=1.0)

    # Optimizer
    print("\n--- Optimizer ---")
    optimizer = get_choice_from_list("Optimizer:", ["adam", "adamw", "sgd"], "adam")
    weight_decay = get_float("Weight decay", 0.01, min_val=0.0)

    # Learning rate schedule
    print("\n--- Learning Rate Schedule ---")
    use_lr_schedule = get_bool("Use LR schedule", True)
    lr_schedule_type = "transformer"
    if use_lr_schedule:
        lr_schedule_type = get_choice_from_list("LR schedule type:", ["transformer", "cosine", "constant"], "transformer")

    # Regularization
    print("\n--- Regularization ---")
    use_gradient_clipping = get_bool("Use gradient clipping", True)
    gradient_clip_value = 1.0
    if use_gradient_clipping:
        gradient_clip_value = get_float("Gradient clip value", 1.0, min_val=0.1)

    # Early stopping
    print("\n--- Early Stopping ---")
    use_early_stopping = get_bool("Use early stopping", True)
    early_stopping_patience = 10
    if use_early_stopping:
        early_stopping_patience = get_int("Early stopping patience", 10, min_val=1)

    # Checkpointing
    print("\n--- Checkpointing ---")
    use_checkpointing = get_bool("Save checkpoints", True)
    save_best_only = True
    if use_checkpointing:
        save_best_only = get_bool("Save best only", True)

    use_tensorboard = get_bool("Use TensorBoard", True)

    # Output
    print("\n--- Output ---")
    output_dir = get_input("Output directory", "./models")
    random_seed = get_int("Random seed", 42)

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
    except Exception as e:
        print(f"\nError creating config: {e}")
        return None



class ExperimentCLI:
    """Menu-driven CLI for experimentation workflows."""

    def __init__(self):
        self.running = True
        ensure_config_dirs()

    def run(self):
        """Run the experiment CLI."""
        clear_screen()
        print_header("Experiment CLI - Dataset & Training")
        print("  Manage datasets, configs, and training.\n")

        while self.running:
            self.main_menu()

        print("\nGoodbye!")


    def main_menu(self):
        """Show main menu."""
        options = [
            "Config Management",
            "Dataset Operations",
            "Model Training",
            "View Information",
        ]
        print_menu("Main Menu", options)

        choice = get_choice("Select option", len(options))

        if choice == 0:
            self.running = False
        elif choice == 1:
            self.config_menu()
        elif choice == 2:
            self.dataset_menu()
        elif choice == 3:
            self.training_menu()
        elif choice == 4:
            self.info_menu()


    def config_menu(self):
        """Config management menu."""
        while True:
            options = [
                "Create Dataset Config",
                "Create Training Config",
                "List Configs",
                "View Config",
            ]
            print_menu("Config Management", options)
            choice = get_choice("Select option", len(options))

            if choice == 0:
                return
            elif choice == 1:
                self.create_dataset_config()
            elif choice == 2:
                self.create_training_config()
            elif choice == 3:
                self.list_configs()
            elif choice == 4:
                self.view_config()

    def create_dataset_config(self):
        """Create a new dataset config."""
        config = create_dataset_config_wizard()
        if config is None:
            wait_for_enter()
            return

        default_filename = f"{config.name}.json"
        filename = get_input("\nConfig filename", default_filename)
        if not filename.endswith('.json'):
            filename += '.json'

        output_path = DATASET_CONFIG_DIR / filename
        config.save(str(output_path))
        print(f"\nDataset config saved to: {output_path}")
        wait_for_enter()

    def create_training_config(self):
        """Create a new training config."""
        config = create_training_config_wizard()
        if config is None:
            wait_for_enter()
            return

        default_filename = f"{config.model_name}.json"
        filename = get_input("\nConfig filename", default_filename)
        if not filename.endswith('.json'):
            filename += '.json'

        output_path = TRAINING_CONFIG_DIR / filename
        config.save(str(output_path))
        print(f"\nTraining config saved to: {output_path}")
        wait_for_enter()

    def list_configs(self):
        """List available configs."""
        print_header("Available Configs")

        print("\nDataset Configs:")
        dataset_configs = list(DATASET_CONFIG_DIR.glob("*.json"))
        if dataset_configs:
            for cfg in dataset_configs:
                print(f"  - {cfg.name}")
        else:
            print("  (none)")

        print("\nTraining Configs:")
        training_configs = list(TRAINING_CONFIG_DIR.glob("*.json"))
        if training_configs:
            for cfg in training_configs:
                print(f"  - {cfg.name}")
        else:
            print("  (none)")

        wait_for_enter()

    def view_config(self):
        """View a config file."""
        import json

        print_header("View Config")
        config_path = get_input("Config file path")

        if not config_path:
            return

        # Try to find in config dirs if just filename
        path = Path(config_path)
        if not path.exists():
            for cfg_dir in [DATASET_CONFIG_DIR, TRAINING_CONFIG_DIR]:
                candidate = cfg_dir / config_path
                if candidate.exists():
                    path = candidate
                    break
                candidate = cfg_dir / f"{config_path}.json"
                if candidate.exists():
                    path = candidate
                    break

        if not path.exists():
            print(f"\nConfig not found: {config_path}")
            wait_for_enter()
            return

        with open(path, 'r') as f:
            config_data = json.load(f)

        print(f"\n{path.name}:")
        print(json.dumps(config_data, indent=2))
        wait_for_enter()


    def dataset_menu(self):
        """Dataset operations menu."""
        while True:
            options = [
                "Build Dataset from Config",
                "View Dataset Info",
                "View Instrument Statistics",
            ]
            print_menu("Dataset Operations", options)
            choice = get_choice("Select option", len(options))

            if choice == 0:
                return
            elif choice == 1:
                self.build_dataset()
            elif choice == 2:
                self.view_dataset_info()
            elif choice == 3:
                self.view_instrument_stats()

    def build_dataset(self):
        """Build dataset from config."""
        from ..config import MusicDatasetConfig
        from ..data import build_and_save_dataset

        print_header("Build Dataset")

        # List available configs
        configs = list(DATASET_CONFIG_DIR.glob("*.json"))
        if configs:
            print("\nAvailable dataset configs:")
            for i, cfg in enumerate(configs, 1):
                print(f"  [{i}] {cfg.name}")
            print()

        config_path = get_input("Dataset config file")
        if not config_path:
            return

        # Resolve path
        path = Path(config_path)
        if not path.exists():
            path = DATASET_CONFIG_DIR / config_path
            if not path.exists():
                path = DATASET_CONFIG_DIR / f"{config_path}.json"

        if not path.exists():
            print(f"\nConfig not found: {config_path}")
            wait_for_enter()
            return

        print(f"\nLoading config: {path}")
        config = MusicDatasetConfig.load(str(path))

        print(f"\nBuilding dataset: {config.name}")
        print(f"  Input directories: {config.input_dirs}")
        print(f"  Output: {config.output_path}")
        print(f"  Encoder: {config.encoder_type}")

        confirm = get_bool("\nProceed with build", True)
        if not confirm:
            return

        try:
            dataset = build_and_save_dataset(config)
            print(f"\nDataset built successfully!")
            print(f"  Entries: {len(dataset)}")
            print(f"  Tracks: {dataset.count_tracks()}")
            print(f"  Genres: {dataset.vocabulary.num_genres}")
        except Exception as e:
            print(f"\nError building dataset: {e}")
            import traceback
            traceback.print_exc()

        wait_for_enter()

    def view_dataset_info(self):
        """View dataset information."""
        from ..data import MusicDataset

        print_header("Dataset Info")

        # Find datasets in data/datasets only
        datasets = []
        dataset_dir = Path("data/datasets")
        if dataset_dir.exists():
            datasets = [d for d in dataset_dir.glob("*.h5") if "model_bundle" not in d.name]

        if datasets:
            print("\nFound datasets:")
            for i, ds in enumerate(datasets[:10], 1):
                print(f"  [{i}] {ds}")
            print()

        dataset_path = get_input("Dataset file path (.h5)")
        if not dataset_path:
            return

        # Handle numeric selection
        if dataset_path.isdigit():
            idx = int(dataset_path) - 1
            if 0 <= idx < len(datasets):
                dataset_path = str(datasets[idx])

        if not Path(dataset_path).exists():
            print(f"\nDataset not found: {dataset_path}")
            wait_for_enter()
            return

        try:
            dataset = MusicDataset.load(dataset_path)
            stats = dataset.get_stats()

            print(f"\nDataset: {dataset_path}")
            print(f"  Entries: {stats['num_entries']}")
            print(f"  Tracks: {stats['num_tracks']}")
            print(f"  Total Notes: {stats['total_notes']}")
            print(f"  Resolution: {stats['resolution']}")
            print(f"  Max Sequence Length: {stats['max_seq_length']}")
            print(f"\nGenres ({stats['num_genres']}):")
            for genre in stats['genres'][:10]:
                print(f"    - {genre}")
            if len(stats['genres']) > 10:
                print(f"    ... and {len(stats['genres']) - 10} more")
            print(f"\nActive Instruments: {stats['num_active_instruments']}")
        except Exception as e:
            print(f"\nError loading dataset: {e}")

        wait_for_enter()

    def view_instrument_stats(self):
        """View instrument usage statistics."""
        from ..data import MusicDataset

        print_header("Instrument Statistics")

        # Find datasets in data/datasets only
        datasets = []
        dataset_dir = Path("data/datasets")
        if dataset_dir.exists():
            datasets = [d for d in dataset_dir.glob("*.h5") if "model_bundle" not in d.name]

        if datasets:
            print("\nFound datasets:")
            for i, ds in enumerate(datasets[:10], 1):
                print(f"  [{i}] {ds}")
            print()

        dataset_path = get_input("Dataset file path (.h5)")
        if not dataset_path:
            return

        # Handle numeric selection
        if dataset_path.isdigit():
            idx = int(dataset_path) - 1
            if 0 <= idx < len(datasets):
                dataset_path = str(datasets[idx])

        if not Path(dataset_path).exists():
            print("\nDataset not found")
            wait_for_enter()
            return

        try:
            dataset = MusicDataset.load(dataset_path)
            top_n = get_int("Show top N instruments", 20, min_val=1)
            dataset.print_instrument_stats(top_n)
        except Exception as e:
            print(f"\nError: {e}")

        wait_for_enter()


    def training_menu(self):
        """Model training menu."""
        while True:
            options = [
                "Train Model from Config",
                "Create Bundle from Checkpoint",
            ]
            print_menu("Model Training", options)
            choice = get_choice("Select option", len(options))

            if choice == 0:
                return
            elif choice == 1:
                self.train_model()
            elif choice == 2:
                self.create_bundle()

    def train_model(self):
        """Train model from config files."""
        from ..config import MusicDatasetConfig, TrainingConfig
        from ..data import MusicDataset
        from ..data import MultiTrackEncoder
        from ..training import Trainer
        from ..models import ModelBundle

        print_header("Train Model")

        # Get dataset
        dataset_path = get_input("Dataset file path (.h5)")
        if not dataset_path or not Path(dataset_path).exists():
            print("\nDataset not found")
            wait_for_enter()
            return

        # List training configs
        configs = list(TRAINING_CONFIG_DIR.glob("*.json"))
        if configs:
            print("\nAvailable training configs:")
            for i, cfg in enumerate(configs, 1):
                print(f"  [{i}] {cfg.name}")
            print()

        config_path = get_input("Training config file")
        if not config_path:
            return

        # Resolve config path
        path = Path(config_path)
        if not path.exists():
            path = TRAINING_CONFIG_DIR / config_path
            if not path.exists():
                path = TRAINING_CONFIG_DIR / f"{config_path}.json"

        if not path.exists():
            print(f"\nConfig not found: {config_path}")
            wait_for_enter()
            return

        try:
            # Load dataset
            print(f"\nLoading dataset: {dataset_path}")
            dataset = MusicDataset.load(dataset_path)
            print(f"  Entries: {len(dataset)}")
            print(f"  Genres: {dataset.vocabulary.num_genres}")

            # Load config
            print(f"\nLoading config: {path}")
            config = TrainingConfig.load(str(path))
            print(config.summary())

            # Load dataset config for encoder settings
            dataset_config_path = Path(dataset_path).with_suffix('.config.json')
            if dataset_config_path.exists():
                dataset_config = MusicDatasetConfig.load(str(dataset_config_path))
                resolution = dataset_config.resolution
                positions_per_bar = dataset_config.positions_per_bar
            else:
                resolution = dataset.resolution
                positions_per_bar = 16

            # Create encoder
            encoder = MultiTrackEncoder(
                num_genres=max(1, dataset.vocabulary.num_genres),
                resolution=resolution,
                positions_per_bar=positions_per_bar,
            )
            print(f"\nEncoder: MultiTrackEncoder")
            print(f"  Vocab size: {encoder.vocab_size}")

            # Get validation split
            val_split = get_float("Validation split", 0.1, min_val=0.01, max_val=0.5)

            confirm = get_bool("\nStart training", True)
            if not confirm:
                return

            # Prepare data
            print("\nPreparing data...")
            datasets = dataset.to_multitrack_dataset(
                encoder=encoder,
                splits=(1.0 - val_split, val_split, 0.0),
                random_state=42,
                min_tracks=2,
            )

            # Train
            trainer = Trainer(config, encoder)
            trainer.build_model()
            model, history = trainer.train(datasets['train'], datasets['validation'])

            # Save bundle with vocabulary
            print("\nSaving model bundle...")
            output_dir = Path(config.output_dir) / config.model_name
            bundle_path = output_dir / "checkpoints" / "model_bundle.h5"

            bundle = ModelBundle(
                model=trainer.model,
                encoder=encoder,
                config=config,
                model_name=config.model_name,
                vocabulary=dataset.vocabulary,  # Include vocabulary!
            )
            bundle.save(bundle_path)

            print(f"\nTraining complete!")
            print(f"  Model bundle: {bundle_path}")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

        wait_for_enter()

    def create_bundle(self):
        """Create model bundle from checkpoint."""
        from tensorflow import keras
        from ..config import MusicDatasetConfig, TrainingConfig
        from ..data import MusicDataset
        from ..models import ModelBundle
        from ..data import EventEncoder, REMIEncoder, MultiTrackEncoder
        from ..models import (
            TransformerModel, TransformerBlock,
            RelativeMultiHeadAttention, RelativePositionalEmbedding,
            LSTMModel, LSTMWithAttention,
        )

        print_header("Create Bundle from Checkpoint")

        # List available checkpoints
        checkpoint_dirs = [Path("./models"), Path("./checkpoints")]
        print("\nSearching for checkpoints...")
        found_checkpoints = []
        for ckpt_dir in checkpoint_dirs:
            if ckpt_dir.exists():
                found_checkpoints.extend(ckpt_dir.rglob("*.keras"))

        if found_checkpoints:
            print("\nFound checkpoints:")
            for i, ckpt in enumerate(found_checkpoints[:10], 1):
                print(f"  [{i}] {ckpt}")
            if len(found_checkpoints) > 10:
                print(f"  ... and {len(found_checkpoints) - 10} more")
            print()

        checkpoint_path = get_input("Checkpoint file (.keras)")
        if not checkpoint_path:
            return

        # Handle numeric selection
        if checkpoint_path.isdigit():
            idx = int(checkpoint_path) - 1
            if 0 <= idx < len(found_checkpoints):
                checkpoint_path = str(found_checkpoints[idx])

        if not Path(checkpoint_path).exists():
            print("\nCheckpoint not found")
            wait_for_enter()
            return

        # List training configs
        configs = list(TRAINING_CONFIG_DIR.glob("*.json"))
        if configs:
            print("\nAvailable training configs:")
            for i, cfg in enumerate(configs, 1):
                print(f"  [{i}] {cfg.name}")
            print()

        config_path = get_input("Training config file (.json)")
        if not config_path:
            return

        # Handle numeric selection
        if config_path.isdigit():
            idx = int(config_path) - 1
            if 0 <= idx < len(configs):
                config_path = str(configs[idx])

        # Resolve config path
        if not Path(config_path).exists():
            for cfg_dir in [TRAINING_CONFIG_DIR]:
                candidate = cfg_dir / config_path
                if candidate.exists():
                    config_path = str(candidate)
                    break
                candidate = cfg_dir / f"{config_path}.json"
                if candidate.exists():
                    config_path = str(candidate)
                    break

        if not Path(config_path).exists():
            print("\nTraining config not found")
            wait_for_enter()
            return

        # List available datasets
        datasets_found = list(Path(".").rglob("*.h5"))
        datasets_found = [d for d in datasets_found if "model_bundle" not in d.name]
        if datasets_found:
            print("\nFound datasets:")
            for i, ds in enumerate(datasets_found[:10], 1):
                print(f"  [{i}] {ds}")
            print()

        dataset_path = get_input("Dataset file (.h5) for vocabulary")
        if not dataset_path:
            return

        # Handle numeric selection
        if dataset_path.isdigit():
            idx = int(dataset_path) - 1
            if 0 <= idx < len(datasets_found):
                dataset_path = str(datasets_found[idx])

        if not Path(dataset_path).exists():
            print("\nDataset not found")
            wait_for_enter()
            return

        default_output = str(Path(checkpoint_path).parent / "model_bundle")
        output_path = get_input("Output bundle path", default_output)

        try:
            # Load training config
            print("\nLoading training config...")
            config = TrainingConfig.load(config_path)
            print(f"  Model: {config.model_name} ({config.model_type})")

            # Load dataset for vocabulary
            print("Loading dataset for vocabulary...")
            dataset = MusicDataset.load(dataset_path)
            vocabulary = dataset.vocabulary
            print(f"  Genres: {vocabulary.num_genres}")
            print(f"  Instruments: {vocabulary.num_active_instruments}")

            # Try to load dataset config for encoder settings
            dataset_config_path = Path(dataset_path).with_suffix('.config.json')
            encoder_type = "multitrack"  # default
            resolution = dataset.resolution
            positions_per_bar = 32
            encode_velocity = True

            if dataset_config_path.exists():
                print(f"Loading dataset config: {dataset_config_path}")
                dataset_config = MusicDatasetConfig.load(str(dataset_config_path))
                encoder_type = dataset_config.encoder_type
                resolution = dataset_config.resolution
                positions_per_bar = dataset_config.positions_per_bar
                encode_velocity = dataset_config.encode_velocity
                print(f"  Encoder type: {encoder_type}")
            else:
                # Ask user for encoder type
                encoder_type = get_choice_from_list(
                    "Encoder type (no dataset config found):",
                    ["multitrack", "event", "remi"],
                    "multitrack"
                )

            # Create encoder based on type
            print(f"Creating {encoder_type} encoder...")
            if encoder_type == "event":
                encoder = EventEncoder(
                    num_genres=vocabulary.num_genres,
                    resolution=resolution,
                    encode_velocity=encode_velocity,
                )
            elif encoder_type == "remi":
                encoder = REMIEncoder(
                    num_genres=vocabulary.num_genres,
                    resolution=resolution,
                    positions_per_bar=positions_per_bar,
                    encode_velocity=encode_velocity,
                )
            else:  # multitrack
                encoder = MultiTrackEncoder(
                    num_genres=vocabulary.num_genres,
                    resolution=resolution,
                    positions_per_bar=positions_per_bar,
                    encode_velocity=encode_velocity,
                )
            print(f"  Vocab size: {encoder.vocab_size}")

            # Load model
            print("Loading model from checkpoint...")
            custom_objects = {
                'TransformerModel': TransformerModel,
                'TransformerBlock': TransformerBlock,
                'RelativeMultiHeadAttention': RelativeMultiHeadAttention,
                'RelativePositionalEmbedding': RelativePositionalEmbedding,
                'LSTMModel': LSTMModel,
                'LSTMWithAttention': LSTMWithAttention,
            }
            model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects, compile=False)

            # Create bundle with vocabulary
            print("Creating bundle...")
            bundle = ModelBundle(
                model=model,
                encoder=encoder,
                config=config,
                model_name=config.model_name,
                vocabulary=vocabulary,
            )
            bundle.save(output_path)

            print(f"\nBundle created: {output_path}.h5")
            print(bundle.summary())

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

        wait_for_enter()


    def info_menu(self):
        """Information viewing menu."""
        while True:
            options = [
                "View Model Bundle Info",
                "View Dataset Info",
            ]
            print_menu("View Information", options)
            choice = get_choice("Select option", len(options))

            if choice == 0:
                return
            elif choice == 1:
                self.view_bundle_info()
            elif choice == 2:
                self.view_dataset_info()

    def view_bundle_info(self):
        """View model bundle information."""
        from ..models import load_model_bundle

        print_header("Model Bundle Info")

        bundle_path = get_input("Bundle file path (.h5)")
        if not bundle_path or not Path(bundle_path).exists():
            print("\nBundle not found")
            wait_for_enter()
            return

        try:
            bundle = load_model_bundle(bundle_path)
            print(bundle.summary())
        except Exception as e:
            print(f"\nError loading bundle: {e}")

        wait_for_enter()


def main():
    """Main entry point."""
    cli = ExperimentCLI()
    cli.run()


if __name__ == "__main__":
    main()
