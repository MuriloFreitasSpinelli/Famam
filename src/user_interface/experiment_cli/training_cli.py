"""CLI for model tuning and training."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

from src.core import MusicDataset, Vocabulary
from src.model_training import ModelTrainingConfig, ModelTrainer, train_from_music_dataset
from src.model_training.configs.slurm_config import SlurmConfig
from src.model_training.configs.transformer_config import TransformerTrainingConfig
from src.model_training.slurm_generator import SlurmScriptGenerator
from src.model_tuning import ModelTuningConfig, tune_from_music_dataset

from .prompts import (
    print_header,
    print_menu,
    get_choice,
    get_input,
    get_int,
    get_float,
    get_bool,
    get_path,
    get_list_input,
    select_config_file,
    confirm,
    list_config_files,
)


# Config directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs"
TRAINING_CONFIG_DIR = CONFIGS_DIR / "model_training"
TRANSFORMER_CONFIG_DIR = CONFIGS_DIR / "transformer_training"
TUNING_CONFIG_DIR = CONFIGS_DIR / "model_tuning"
SLURM_CONFIG_DIR = CONFIGS_DIR / "slurm"
DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
SLURM_JOBS_DIR = PROJECT_ROOT / "slurm_jobs"
SLURM_LOGS_DIR = PROJECT_ROOT / "slurm_logs"


def ensure_config_dirs():
    """Ensure all config directories exist."""
    TRAINING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TUNING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def prompt_training_config(is_tuned: bool = False) -> ModelTrainingConfig:
    """Interactively create a ModelTrainingConfig."""
    print_header("Create Training Configuration")

    print("Model identification:")
    print("-" * 40)

    name = get_input("Model name", default="my_model")
    if is_tuned and not name.endswith("_tuned"):
        name = f"{name}_tuned"

    print("\nInput shape (must match dataset):")
    print("-" * 40)
    num_pitches = get_int("Number of pitches", default=128, min_val=1)
    max_time_steps = get_int("Max time steps", default=1000, min_val=1)

    print("\nLSTM architecture:")
    print("-" * 40)

    lstm_units_str = get_input("LSTM units (comma-separated)", default="128,64")
    lstm_units = [int(x.strip()) for x in lstm_units_str.split(",")]

    dense_units_str = get_input("Dense units (comma-separated)", default="64,32")
    dense_units = [int(x.strip()) for x in dense_units_str.split(",")]

    dropout_rate = get_float("Dropout rate", default=0.2, min_val=0.0, max_val=1.0)
    recurrent_dropout = get_float("Recurrent dropout", default=0.1, min_val=0.0, max_val=1.0)
    bidirectional = get_bool("Bidirectional LSTM", default=False)

    print("\nEmbedding dimensions:")
    print("-" * 40)
    genre_embedding_dim = get_int("Genre embedding dim", default=16, min_val=1)

    print("\nTraining hyperparameters:")
    print("-" * 40)
    batch_size = get_int("Batch size", default=32, min_val=1)
    epochs = get_int("Epochs", default=100, min_val=1)
    learning_rate = get_float("Learning rate", default=0.001, min_val=0.0)

    print("\nGeneration training (input dropout):")
    print("-" * 40)
    print("Input dropout randomly zeros input during training, forcing the model")
    print("to generate from conditioning (genre/instrument/drums) rather than just")
    print("reconstructing input. This enables generation with zero seeds.")
    input_dropout_rate = get_float("Input dropout rate", default=0.3, min_val=0.0, max_val=1.0)

    print("\nOptimizer:")
    optimizers = ["adam", "sgd", "rmsprop", "adamax", "nadam"]
    print_menu(optimizers, "Select optimizer:")
    opt_choice = get_choice(len(optimizers))
    optimizer = optimizers[opt_choice - 1]

    print("\nLoss function:")
    loss_functions = ["mse", "mae", "binary_crossentropy", "huber"]
    print_menu(loss_functions, "Select loss function:")
    loss_choice = get_choice(len(loss_functions))
    loss_function = loss_functions[loss_choice - 1]

    print("\nRegularization:")
    print("-" * 40)
    l1_reg = get_float("L1 regularization", default=0.0, min_val=0.0)
    l2_reg = get_float("L2 regularization", default=0.0, min_val=0.0)
    use_gradient_clipping = get_bool("Use gradient clipping", default=False)
    gradient_clip_value = 1.0
    if use_gradient_clipping:
        gradient_clip_value = get_float("Gradient clip value", default=1.0, min_val=0.0)

    print("\nEarly stopping:")
    print("-" * 40)
    use_early_stopping = get_bool("Use early stopping", default=True)
    early_stopping_patience = 10
    if use_early_stopping:
        early_stopping_patience = get_int("Early stopping patience", default=10, min_val=1)

    print("\nCheckpointing:")
    print("-" * 40)
    use_checkpointing = get_bool("Save checkpoints", default=True)

    print("\nTensorBoard:")
    print("-" * 40)
    use_tensorboard = get_bool("Use TensorBoard logging", default=True)

    print("\nLearning rate schedule:")
    lr_schedules = ["constant", "reduce_on_plateau", "exponential_decay", "cosine_decay"]
    print_menu(lr_schedules, "Select LR schedule:")
    lr_choice = get_choice(len(lr_schedules))
    lr_schedule = lr_schedules[lr_choice - 1]

    config = ModelTrainingConfig(
        model_name=name,
        num_pitches=num_pitches,
        max_time_steps=max_time_steps,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        bidirectional=bidirectional,
        genre_embedding_dim=genre_embedding_dim,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        input_dropout_rate=input_dropout_rate,
        optimizer=optimizer,
        loss_function=loss_function,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        use_gradient_clipping=use_gradient_clipping,
        gradient_clip_value=gradient_clip_value,
        use_early_stopping=use_early_stopping,
        early_stopping_patience=early_stopping_patience,
        use_checkpointing=use_checkpointing,
        use_tensorboard=use_tensorboard,
        lr_schedule=lr_schedule,
        output_dir=str(MODELS_DIR),
    )

    # Always save config
    ensure_config_dirs()
    save_path = TRAINING_CONFIG_DIR / f"{name}.json"
    config.save(str(save_path))
    print(f"\nConfiguration saved to: {save_path}")

    return config


def prompt_transformer_config() -> TransformerTrainingConfig:
    """Interactively create a TransformerTrainingConfig."""
    print_header("Create Transformer Training Configuration")

    print("Model identification:")
    print("-" * 40)

    name = get_input("Model name", default="music_transformer")

    print("\nTransformer architecture:")
    print("-" * 40)

    # Preset options for quick configuration
    presets = [
        ("Small (4 layers, 256 dim - fast training)", "small"),
        ("Medium (6 layers, 512 dim - balanced)", "medium"),
        ("Large (12 layers, 768 dim - high quality)", "large"),
        ("Custom (configure manually)", "custom"),
    ]
    print("\nSelect architecture preset:")
    for i, (desc, _) in enumerate(presets, 1):
        print(f"  [{i}] {desc}")
    print()

    preset_choice = get_choice(len(presets))
    preset = presets[preset_choice - 1][1]

    if preset == "small":
        max_seq_length = 512
        num_layers = 4
        d_model = 256
        num_heads = 4
        d_ff = 1024
    elif preset == "medium":
        max_seq_length = 1024
        num_layers = 6
        d_model = 512
        num_heads = 8
        d_ff = 2048
    elif preset == "large":
        max_seq_length = 1024
        num_layers = 12
        d_model = 768
        num_heads = 12
        d_ff = 3072
    else:
        # Custom configuration
        max_seq_length = get_int("Max sequence length", default=2048, min_val=128)
        num_layers = get_int("Number of transformer layers", default=6, min_val=1)
        d_model = get_int("Model dimension (d_model)", default=512, min_val=64)
        num_heads = get_int("Number of attention heads", default=8, min_val=1)
        d_ff = get_int("Feed-forward dimension", default=2048, min_val=256)

        # Validate d_model is divisible by num_heads
        while d_model % num_heads != 0:
            print(f"Warning: d_model ({d_model}) must be divisible by num_heads ({num_heads})")
            d_model = get_int("Model dimension (d_model)", default=512, min_val=64)

    dropout_rate = get_float("Dropout rate", default=0.1, min_val=0.0, max_val=0.5)

    print("\nTraining hyperparameters:")
    print("-" * 40)
    batch_size = get_int("Batch size (reduce if OOM)", default=4, min_val=1)
    epochs = get_int("Epochs", default=100, min_val=1)
    learning_rate = get_float("Learning rate", default=0.0001, min_val=0.0)
    warmup_steps = get_int("Warmup steps", default=4000, min_val=0)

    print("\nLabel smoothing:")
    print("-" * 40)
    label_smoothing = get_float("Label smoothing", default=0.1, min_val=0.0, max_val=0.5)

    print("\nOptimizer:")
    optimizers = ["adam", "adamw", "sgd", "rmsprop"]
    print_menu(optimizers, "Select optimizer:")
    opt_choice = get_choice(len(optimizers))
    optimizer = optimizers[opt_choice - 1]

    weight_decay = 0.01
    if optimizer == "adamw":
        weight_decay = get_float("Weight decay", default=0.01, min_val=0.0)

    print("\nLearning rate schedule:")
    print("-" * 40)
    use_transformer_lr_schedule = get_bool("Use Transformer LR schedule (warmup + decay)?", default=True)

    print("\nRegularization:")
    print("-" * 40)
    use_gradient_clipping = get_bool("Use gradient clipping", default=True)
    gradient_clip_value = 1.0
    if use_gradient_clipping:
        gradient_clip_value = get_float("Gradient clip value", default=1.0, min_val=0.0)

    print("\nEarly stopping:")
    print("-" * 40)
    use_early_stopping = get_bool("Use early stopping", default=True)
    early_stopping_patience = 10
    if use_early_stopping:
        early_stopping_patience = get_int("Early stopping patience", default=10, min_val=1)

    print("\nCheckpointing:")
    print("-" * 40)
    use_checkpointing = get_bool("Save checkpoints", default=True)

    print("\nTensorBoard:")
    print("-" * 40)
    use_tensorboard = get_bool("Use TensorBoard logging", default=True)

    config = TransformerTrainingConfig(
        model_name=name,
        max_seq_length=max_seq_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        label_smoothing=label_smoothing,
        optimizer=optimizer,
        weight_decay=weight_decay,
        use_transformer_lr_schedule=use_transformer_lr_schedule,
        use_gradient_clipping=use_gradient_clipping,
        gradient_clip_value=gradient_clip_value,
        use_early_stopping=use_early_stopping,
        early_stopping_patience=early_stopping_patience,
        use_checkpointing=use_checkpointing,
        use_tensorboard=use_tensorboard,
        output_dir=str(MODELS_DIR),
    )

    # Always save config
    ensure_config_dirs()
    save_path = TRANSFORMER_CONFIG_DIR / f"{name}.json"
    config.save(str(save_path))
    print(f"\nConfiguration saved to: {save_path}")

    return config


def get_transformer_config() -> Optional[TransformerTrainingConfig]:
    """Get TransformerTrainingConfig either from file or by creating new one."""
    print_header("Transformer Training Configuration")

    options = [
        "Load existing config",
        "Create new config",
        "Back",
    ]
    print_menu(options)
    choice = get_choice(len(options))

    if choice == 1:
        ensure_config_dirs()
        config_path = select_config_file(TRANSFORMER_CONFIG_DIR, "transformer training")
        if config_path:
            config = TransformerTrainingConfig.load(str(config_path))
            print(f"\nLoaded config: {config.model_name}")
            print(f"  Architecture: {config.num_layers} layers, d_model={config.d_model}")
            print(f"  Epochs: {config.epochs}")
            print(f"  Learning rate: {config.learning_rate}")
            return config
        return None
    elif choice == 2:
        return prompt_transformer_config()
    else:
        return None


def prompt_tuning_config() -> ModelTuningConfig:
    """Interactively create a ModelTuningConfig."""
    print_header("Create Tuning Configuration")

    print("Tuning method:")
    print("-" * 40)
    methods = ["random_search", "grid_search", "bayesian"]
    print_menu(methods, "Select tuning method:")
    method_choice = get_choice(len(methods))
    tuning_method = methods[method_choice - 1]

    print("\nCross-validation settings:")
    print("-" * 40)
    tuning_cv_folds = get_int("CV folds", default=3, min_val=2)

    n_iter = 10
    bayesian_n_calls = 50
    bayesian_n_initial_points = 10

    if tuning_method == "random_search":
        n_iter = get_int("Number of iterations", default=10, min_val=1)
    elif tuning_method == "bayesian":
        bayesian_n_calls = get_int("Bayesian n_calls", default=50, min_val=1)
        bayesian_n_initial_points = get_int("Initial random points", default=10, min_val=1)

    print("\nParameter grid:")
    print("-" * 40)
    print("Define parameter ranges to search:")

    # LSTM units options
    print("\nLSTM units configurations (comma-separated lists, semicolon between configs):")
    print("Example: 64;128;128,64;256,128")
    lstm_str = get_input("LSTM units options", default="64;128;128,64;256,128")
    lstm_options = []
    for opt in lstm_str.split(";"):
        lstm_options.append([int(x.strip()) for x in opt.split(",")])

    # Dense units options
    print("\nDense units configurations:")
    dense_str = get_input("Dense units options", default="32;64;64,32")
    dense_options = []
    for opt in dense_str.split(";"):
        dense_options.append([int(x.strip()) for x in opt.split(",")])

    # Dropout rates
    dropout_str = get_input("Dropout rates (comma-separated)", default="0.1,0.2,0.3")
    dropout_options = [float(x.strip()) for x in dropout_str.split(",")]

    # Learning rates
    lr_str = get_input("Learning rates (comma-separated)", default="0.0001,0.001,0.01")
    lr_options = [float(x.strip()) for x in lr_str.split(",")]

    param_grid = {
        'lstm_units': lstm_options,
        'dense_units': dense_options,
        'dropout_rate': dropout_options,
        'learning_rate': lr_options,
    }

    config = ModelTuningConfig(
        use_hyperparameter_tuning=True,
        tuning_method=tuning_method,
        tuning_cv_folds=tuning_cv_folds,
        n_iter=n_iter,
        bayesian_n_calls=bayesian_n_calls,
        bayesian_n_initial_points=bayesian_n_initial_points,
        param_grid=param_grid,
        output_dir=str(MODELS_DIR),
    )

    # Always save config
    ensure_config_dirs()
    config_name = get_input("Configuration name", default="tuning_config")
    save_path = TUNING_CONFIG_DIR / f"{config_name}.json"
    config.save(str(save_path))
    print(f"\nConfiguration saved to: {save_path}")

    return config


def get_training_config(is_tuned: bool = False) -> Optional[ModelTrainingConfig]:
    """Get ModelTrainingConfig either from file or by creating new one."""
    print_header("Training Configuration")

    options = [
        "Load existing config",
        "Create new config",
        "Back",
    ]
    print_menu(options)
    choice = get_choice(len(options))

    if choice == 1:
        ensure_config_dirs()
        config_path = select_config_file(TRAINING_CONFIG_DIR, "training")
        if config_path:
            config = ModelTrainingConfig.load(str(config_path))
            print(f"\nLoaded config: {config.model_name}")
            print(f"  LSTM units: {config.lstm_units}")
            print(f"  Epochs: {config.epochs}")
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Input dropout rate: {config.input_dropout_rate}")
            return config
        return None
    elif choice == 2:
        return prompt_training_config(is_tuned=is_tuned)
    else:
        return None


def get_tuning_config() -> Optional[ModelTuningConfig]:
    """Get ModelTuningConfig either from file or by creating new one."""
    print_header("Tuning Configuration")

    options = [
        "Load existing config",
        "Create new config",
        "Back",
    ]
    print_menu(options)
    choice = get_choice(len(options))

    if choice == 1:
        ensure_config_dirs()
        config_path = select_config_file(TUNING_CONFIG_DIR, "tuning")
        if config_path:
            config = ModelTuningConfig.load(str(config_path))
            print(f"\nLoaded tuning config:")
            print(f"  Method: {config.tuning_method}")
            print(f"  CV folds: {config.tuning_cv_folds}")
            return config
        return None
    elif choice == 2:
        return prompt_tuning_config()
    else:
        return None


def select_dataset() -> Optional[MusicDataset]:
    """Select and load a dataset."""
    print_header("Select Dataset")

    dataset_files = list(DATASET_DIR.glob("*.h5"))
    if not dataset_files:
        print(f"No datasets found in {DATASET_DIR}")
        print("Please create a dataset first using 'Create Dataset' option.")
        input("\nPress Enter to continue...")
        return None

    print("Available datasets:")
    print("-" * 40)
    for i, f in enumerate(dataset_files, 1):
        print(f"  [{i}] {f.name}")
    print()

    choice = get_choice(len(dataset_files), "Select dataset: ")
    dataset_path = dataset_files[choice - 1]

    print(f"\nLoading dataset: {dataset_path.name}...")
    dataset = MusicDataset.load(str(dataset_path))
    print(f"  Entries: {len(dataset)}")
    print(f"  Tracks: {dataset.count_tracks()}")
    print(f"  Genres: {list(dataset.vocabulary.genre_to_id.keys())}")

    return dataset


def prompt_distribution_config() -> Tuple[str, Optional[int]]:
    """
    Prompt for distribution strategy configuration.

    Returns:
        Tuple of (strategy_name, batch_size_per_replica)
    """
    print_header("Distribution Strategy")

    strategies = [
        ("None (single GPU)", "none"),
        ("MirroredStrategy (multi-GPU, single node)", "mirrored"),
        ("MultiWorkerMirroredStrategy (multi-node)", "multi_worker_mirrored"),
    ]

    print("Select distribution strategy:")
    for i, (desc, _) in enumerate(strategies, 1):
        print(f"  [{i}] {desc}")
    print()

    choice = get_choice(len(strategies))
    strategy = strategies[choice - 1][1]

    batch_size_per_replica = None
    if strategy != 'none':
        if get_bool("Set batch size per replica?", default=False):
            batch_size_per_replica = get_int("Batch size per replica", default=32, min_val=1)

    return strategy, batch_size_per_replica


def prompt_slurm_config() -> SlurmConfig:
    """Interactively create a SlurmConfig."""
    print_header("SLURM Job Configuration")

    print("Job identification:")
    print("-" * 40)
    job_name = get_input("Job name", default="famam_train")

    print("\nGPU partition:")
    partitions = [
        ("gpu_h100 (NVIDIA H100, high performance)", "gpu_h100"),
        ("gpu_a100_il (NVIDIA A100, general purpose)", "gpu_a100_il"),
        ("gpu_a100_short (NVIDIA A100, max 30 min)", "gpu_a100_short"),
    ]
    for i, (desc, _) in enumerate(partitions, 1):
        print(f"  [{i}] {desc}")
    print()
    part_choice = get_choice(len(partitions))
    partition = partitions[part_choice - 1][1]

    print("\nResource allocation:")
    print("-" * 40)
    nodes = get_int("Number of nodes", default=1, min_val=1)
    gpus_per_node = get_int("GPUs per node", default=1, min_val=1, max_val=8)

    # Auto-calculate recommended CPUs and memory
    recommended_cpus = gpus_per_node * 8
    recommended_memory = gpus_per_node * 32

    cpus_per_task = get_int(f"CPUs per task (recommended: {recommended_cpus})",
                           default=recommended_cpus, min_val=1)
    memory_gb = get_int(f"Memory in GB (recommended: {recommended_memory})",
                       default=recommended_memory, min_val=1)

    print("\nTime limit:")
    print("-" * 40)
    if partition == 'gpu_a100_short':
        print("Note: gpu_a100_short has a 30-minute time limit")
        time_limit = get_input("Time limit (HH:MM:SS)", default="00:30:00")
    else:
        time_limit = get_input("Time limit (HH:MM:SS)", default="24:00:00")

    print("\nEnvironment:")
    print("-" * 40)
    conda_env = get_input("Conda environment name (leave empty to skip)", default="")
    if not conda_env:
        conda_env = None

    # Module loads
    default_modules = "devel/cuda/12.1,devel/python/3.11"
    modules_str = get_input("Modules to load (comma-separated)", default=default_modules)
    module_loads = [m.strip() for m in modules_str.split(",") if m.strip()]

    # Email notifications
    email = None
    if get_bool("Enable email notifications?", default=False):
        email = get_input("Email address")

    # output_dir: where to save the .sh script file locally (absolute path)
    # log_dir: where SLURM writes logs on the cluster (relative path, used in script)
    config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
        memory_gb=memory_gb,
        time_limit=time_limit,
        conda_env=conda_env,
        module_loads=module_loads,
        email=email,
        output_dir=str(SLURM_JOBS_DIR),
        log_dir="slurm_logs",
    )

    # Save config
    SLURM_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SLURM_CONFIG_DIR / f"{job_name}.json"
    config.save(str(save_path))
    print(f"\nSLURM configuration saved to: {save_path}")

    return config


def run_distributed_training():
    """Run distributed training with strategy selection."""
    print_header("Distributed Training")

    # Select dataset
    dataset = select_dataset()
    if dataset is None:
        return

    # Get training config
    training_config = get_training_config()
    if training_config is None:
        print("Training cancelled.")
        return

    # Ensure max_time_steps matches the dataset
    if training_config.max_time_steps != dataset.max_time_steps:
        print(f"\nNote: Updating config max_time_steps from {training_config.max_time_steps} "
              f"to {dataset.max_time_steps} to match dataset.")
        training_config.max_time_steps = dataset.max_time_steps

    # Get distribution config
    strategy, batch_size_per_replica = prompt_distribution_config()

    # Update config with distribution settings
    training_config.distribution_strategy = strategy
    if batch_size_per_replica:
        training_config.batch_size_per_replica = batch_size_per_replica

    # Confirm
    print_header("Distributed Training Summary")
    print(f"Model: {training_config.model_name}")
    print(f"Dataset: {len(dataset)} entries, {dataset.count_tracks()} tracks")
    print(f"Strategy: {strategy}")
    print(f"LSTM units: {training_config.lstm_units}")
    print(f"Epochs: {training_config.epochs}")
    if batch_size_per_replica:
        print(f"Batch size per replica: {batch_size_per_replica}")
    else:
        print(f"Batch size: {training_config.batch_size}")
    print(f"Input dropout rate: {training_config.input_dropout_rate}")
    print()

    if not confirm("Start distributed training?", default=True):
        print("Training cancelled.")
        return

    # Split dataset
    print("\nSplitting dataset...")
    splits = (0.8, 0.1, 0.1)
    datasets = dataset.to_tensorflow_dataset(splits=splits, random_state=42)

    # Run training
    print("\n" + "=" * 60)
    print(f"Starting distributed training with {strategy}...")
    print("=" * 60 + "\n")

    try:
        model, history, trainer = train_from_music_dataset(
            datasets=datasets,
            config=training_config,
            vocabulary=dataset.vocabulary,
        )

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

        # Save model bundle
        if confirm("\nSave model bundle for generation?", default=True):
            bundle_path = MODELS_DIR / training_config.model_name / "model_bundle"
            trainer.save_bundle(str(bundle_path), dataset.vocabulary)
            print(f"Model bundle saved to: {bundle_path}.h5")

    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    input("\nPress Enter to continue...")


def run_slurm_generator():
    """Generate SLURM job scripts for cluster execution."""
    print_header("SLURM Script Generator")

    options = [
        "Generate training script",
        "Generate tuning script",
        "Back",
    ]
    print_menu(options)
    choice = get_choice(len(options))

    if choice == 3:
        return

    # Get SLURM config
    print("\n--- SLURM Configuration ---")
    slurm_options = [
        "Load existing SLURM config",
        "Create new SLURM config",
    ]
    print_menu(slurm_options)
    slurm_choice = get_choice(len(slurm_options))

    if slurm_choice == 1:
        SLURM_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = select_config_file(SLURM_CONFIG_DIR, "SLURM")
        if config_path:
            slurm_config = SlurmConfig.load(str(config_path))
        else:
            print("No config selected, using defaults.")
            slurm_config = prompt_slurm_config()
    else:
        slurm_config = prompt_slurm_config()

    generator = SlurmScriptGenerator(slurm_config)

    if choice == 1:
        # Generate training script
        _generate_training_slurm_script(generator, slurm_config)
    elif choice == 2:
        # Generate tuning script
        _generate_tuning_slurm_script(generator, slurm_config)


def _generate_training_slurm_script(generator: SlurmScriptGenerator, slurm_config: SlurmConfig):
    """Generate a training SLURM script."""
    print("\n--- Training Configuration ---")

    # Select training config
    config_path = select_config_file(TRAINING_CONFIG_DIR, "training")
    if config_path is None:
        print("No training config selected.")
        return

    # Select dataset
    dataset_files = list(DATASET_DIR.glob("*.h5"))
    if not dataset_files:
        print(f"No datasets found in {DATASET_DIR}")
        return

    print("\nAvailable datasets:")
    for i, f in enumerate(dataset_files, 1):
        print(f"  [{i}] {f.name}")
    print()
    ds_choice = get_choice(len(dataset_files))
    dataset_path = dataset_files[ds_choice - 1]

    # Distribution strategy
    strategies = ["none", "mirrored", "multi_worker_mirrored"]
    print("\nDistribution strategy:")
    for i, s in enumerate(strategies, 1):
        print(f"  [{i}] {s}")
    print()

    # Default strategy based on config
    default_strategy = 1
    if slurm_config.nodes > 1:
        default_strategy = 3
    elif slurm_config.gpus_per_node > 1:
        default_strategy = 2

    strat_choice = get_choice(len(strategies), f"Select strategy [{default_strategy}]: ")
    strategy = strategies[strat_choice - 1] if strat_choice else strategies[default_strategy - 1]

    # Convert absolute paths to relative paths (relative to project root)
    # This ensures paths work on the cluster regardless of local machine paths
    config_path_rel = config_path.relative_to(PROJECT_ROOT).as_posix()
    dataset_path_rel = dataset_path.relative_to(PROJECT_ROOT).as_posix()

    # Generate script
    print("\n" + "=" * 60)
    script_path = generator.save_training_script(
        config_path=config_path_rel,
        dataset_path=dataset_path_rel,
        strategy=strategy,
        project_dir=".",
    )

    print("\nScript generated successfully!")
    print(f"To submit: sbatch {script_path}")
    print("=" * 60)

    input("\nPress Enter to continue...")


def _generate_tuning_slurm_script(generator: SlurmScriptGenerator, slurm_config: SlurmConfig):
    """Generate a tuning SLURM script."""
    print("\n--- Tuning Configuration ---")

    # Select tuning config
    tuning_config_path = select_config_file(TUNING_CONFIG_DIR, "tuning")
    if tuning_config_path is None:
        print("No tuning config selected.")
        return

    # Select training config (base for tuning)
    print("\nSelect base training config:")
    training_config_path = select_config_file(TRAINING_CONFIG_DIR, "training")
    if training_config_path is None:
        print("No training config selected.")
        return

    # Select dataset
    dataset_files = list(DATASET_DIR.glob("*.h5"))
    if not dataset_files:
        print(f"No datasets found in {DATASET_DIR}")
        return

    print("\nAvailable datasets:")
    for i, f in enumerate(dataset_files, 1):
        print(f"  [{i}] {f.name}")
    print()
    ds_choice = get_choice(len(dataset_files))
    dataset_path = dataset_files[ds_choice - 1]

    max_samples = get_int("Max samples for tuning", default=1000, min_val=100)

    # Convert absolute paths to relative paths (relative to project root)
    # This ensures paths work on the cluster regardless of local machine paths
    tuning_config_path_rel = tuning_config_path.relative_to(PROJECT_ROOT).as_posix()
    training_config_path_rel = training_config_path.relative_to(PROJECT_ROOT).as_posix()
    dataset_path_rel = dataset_path.relative_to(PROJECT_ROOT).as_posix()

    # Generate script
    print("\n" + "=" * 60)
    script_path = generator.save_tuning_script(
        tuning_config_path=tuning_config_path_rel,
        training_config_path=training_config_path_rel,
        dataset_path=dataset_path_rel,
        max_samples=max_samples,
        project_dir=".",
    )

    print("\nScript generated successfully!")
    print(f"To submit: sbatch {script_path}")
    print("=" * 60)

    input("\nPress Enter to continue...")


def run_tuning_only():
    """Run hyperparameter tuning only."""
    print_header("Hyperparameter Tuning")

    # Select dataset
    dataset = select_dataset()
    if dataset is None:
        return

    # Get tuning config
    tuning_config = get_tuning_config()
    if tuning_config is None:
        print("Tuning cancelled.")
        return

    # Confirm
    print_header("Tuning Summary")
    print(f"Dataset: {len(dataset)} entries, {dataset.count_tracks()} tracks")
    print(f"Tuning method: {tuning_config.tuning_method}")
    print(f"CV folds: {tuning_config.tuning_cv_folds}")
    print(f"Parameter grid: {tuning_config.param_grid}")
    print()

    if not confirm("Start hyperparameter tuning?", default=True):
        print("Tuning cancelled.")
        return

    # Get max samples for tuning (subset for speed)
    max_samples = get_int("Max samples for tuning (subset for speed)", default=1000, min_val=100)

    # Convert to tensorflow dataset
    print("\nConverting dataset to TensorFlow format...")
    tf_dataset = dataset.to_tensorflow_dataset()

    # Run tuning
    print("\n" + "=" * 60)
    print("Starting hyperparameter tuning...")
    print("=" * 60 + "\n")

    try:
        search, best_params = tune_from_music_dataset(
            train_dataset=tf_dataset,
            config=tuning_config,
            num_genres=dataset.vocabulary.num_genres,
            input_shape=(dataset.config.num_pitches if dataset.config else 128,
                        dataset.max_time_steps),
            max_samples=max_samples,
        )

        # Save results
        results_path = tuning_config.save_tuning_results(search, "tuning_experiment")

        print("\n" + "=" * 60)
        print("Tuning complete!")
        print("=" * 60)
        print(f"Best score: {search.best_score_:.4f}")
        print(f"Best params: {best_params}")
        print(f"Results saved to: {results_path}")

        # Offer to create training config from best params
        if confirm("\nCreate training config from best parameters?", default=True):
            model_name = get_input("Model name", default="tuned_model")
            if not model_name.endswith("_tuned"):
                model_name = f"{model_name}_tuned"

            training_config = ModelTuningConfig.from_tuning_results(
                results_path,
                model_name=model_name,
                epochs=get_int("Training epochs", default=100, min_val=1),
            )

            # Save training config
            ensure_config_dirs()
            save_path = TRAINING_CONFIG_DIR / f"{model_name}.json"
            training_config.save(str(save_path))
            print(f"Training config saved to: {save_path}")

    except Exception as e:
        print(f"\nError during tuning: {e}")
        raise

    input("\nPress Enter to continue...")


def run_training_only():
    """Run model training only."""
    print_header("Model Training")

    # Select dataset
    dataset = select_dataset()
    if dataset is None:
        return

    # Get training config
    training_config = get_training_config()
    if training_config is None:
        print("Training cancelled.")
        return

    # Ensure max_time_steps matches the dataset
    if training_config.max_time_steps != dataset.max_time_steps:
        print(f"\nNote: Updating config max_time_steps from {training_config.max_time_steps} "
              f"to {dataset.max_time_steps} to match dataset.")
        training_config.max_time_steps = dataset.max_time_steps

    # Confirm
    print_header("Training Summary")
    print(f"Model: {training_config.model_name}")
    print(f"Dataset: {len(dataset)} entries, {dataset.count_tracks()} tracks")
    print(f"LSTM units: {training_config.lstm_units}")
    print(f"Dense units: {training_config.dense_units}")
    print(f"Epochs: {training_config.epochs}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Input dropout rate: {training_config.input_dropout_rate}")
    print()

    if not confirm("Start training?", default=True):
        print("Training cancelled.")
        return

    # Split dataset
    print("\nSplitting dataset...")
    splits = (0.8, 0.1, 0.1)
    datasets = dataset.to_tensorflow_dataset(splits=splits, random_state=42)

    print(f"  Train/Val/Test split: {splits}")

    # Run training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    try:
        model, history, trainer = train_from_music_dataset(
            datasets=datasets,
            config=training_config,
            vocabulary=dataset.vocabulary,
        )

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

        # Save model bundle
        if confirm("\nSave model bundle for generation?", default=True):
            bundle_path = MODELS_DIR / training_config.model_name / "model_bundle"
            trainer.save_bundle(str(bundle_path), dataset.vocabulary)
            print(f"Model bundle saved to: {bundle_path}.h5")

    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    input("\nPress Enter to continue...")


def run_transformer_training():
    """Run Transformer model training."""
    print_header("Transformer Model Training")

    # Select dataset
    dataset = select_dataset()
    if dataset is None:
        return

    # Get transformer training config
    training_config = get_transformer_config()
    if training_config is None:
        print("Training cancelled.")
        return

    # Show configuration summary
    print_header("Transformer Training Summary")
    print(f"Model: {training_config.model_name}")
    print(f"Dataset: {len(dataset)} entries, {dataset.count_tracks()} tracks")
    print(f"Architecture: {training_config.num_layers} layers, d_model={training_config.d_model}")
    print(f"Attention heads: {training_config.num_heads}")
    print(f"Max sequence length: {training_config.max_seq_length}")
    print(f"Epochs: {training_config.epochs}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Warmup steps: {training_config.warmup_steps}")
    print()

    if not confirm("Start Transformer training?", default=True):
        print("Training cancelled.")
        return

    # Import transformer trainer
    from src.model_training.transformer_trainer import TransformerTrainer
    from src.core.event_vocabulary import EventVocabulary

    # Create event vocabulary
    event_vocab = EventVocabulary(
        num_genres=dataset.vocabulary.num_genres,
        num_instruments=dataset.vocabulary.num_instruments,
    )

    print(f"\nEvent vocabulary size: {event_vocab.vocab_size}")

    # Convert dataset to event format
    print("\nConverting dataset to event sequences...")
    splits = (0.8, 0.1, 0.1)
    datasets = dataset.to_event_tensorflow_dataset(
        event_vocab=event_vocab,
        splits=splits,
        max_seq_length=training_config.max_seq_length,
        random_state=42,
    )

    print(f"  Train/Val/Test split: {splits}")

    # Run training
    print("\n" + "=" * 60)
    print("Starting Transformer training...")
    print("=" * 60 + "\n")

    try:
        trainer = TransformerTrainer(training_config, event_vocab)

        model, history = trainer.train(
            train_dataset=datasets['train'],
            val_dataset=datasets['validation'],
        )

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

        # Save model bundle
        if confirm("\nSave model bundle for generation?", default=True):
            bundle_path = MODELS_DIR / training_config.model_name / "model_bundle"
            trainer.save_bundle(str(bundle_path), dataset.vocabulary)
            print(f"Model bundle saved to: {bundle_path}.h5")

    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    input("\nPress Enter to continue...")


def run_tuning_then_training():
    """Run hyperparameter tuning followed by training with best params."""
    print_header("Tune & Train Pipeline")

    # Select dataset
    dataset = select_dataset()
    if dataset is None:
        return

    # Get tuning config
    print("\n--- Step 1: Tuning Configuration ---")
    tuning_config = get_tuning_config()
    if tuning_config is None:
        print("Pipeline cancelled.")
        return

    # Get base training settings (will be overridden by tuning results)
    print("\n--- Step 2: Base Training Settings ---")
    print("These settings will be used for final training (architecture will be tuned):")

    model_name = get_input("Model name (will add _tuned suffix)", default="my_model")
    if not model_name.endswith("_tuned"):
        model_name = f"{model_name}_tuned"

    final_epochs = get_int("Final training epochs", default=100, min_val=1)
    max_tuning_samples = get_int("Max samples for tuning", default=1000, min_val=100)

    # Confirm
    print_header("Pipeline Summary")
    print(f"Model: {model_name}")
    print(f"Dataset: {len(dataset)} entries")
    print(f"Tuning method: {tuning_config.tuning_method}")
    print(f"Tuning samples: {max_tuning_samples}")
    print(f"Final training epochs: {final_epochs}")
    print()

    if not confirm("Start tune & train pipeline?", default=True):
        print("Pipeline cancelled.")
        return

    # Convert dataset
    print("\nPreparing dataset...")
    splits = (0.8, 0.1, 0.1)
    datasets = dataset.to_tensorflow_dataset(splits=splits, random_state=42)
    full_dataset = dataset.to_tensorflow_dataset()

    input_shape = (128, dataset.max_time_steps)

    # === TUNING PHASE ===
    print("\n" + "=" * 60)
    print("PHASE 1: Hyperparameter Tuning")
    print("=" * 60 + "\n")

    try:
        search, best_params = tune_from_music_dataset(
            train_dataset=full_dataset,
            config=tuning_config,
            num_genres=dataset.vocabulary.num_genres,
            input_shape=input_shape,
            max_samples=max_tuning_samples,
        )

        print(f"\nBest score: {search.best_score_:.4f}")
        print(f"Best params: {best_params}")

        # Save tuning results
        results_path = tuning_config.save_tuning_results(search, f"{model_name}_tuning")

    except Exception as e:
        print(f"\nError during tuning: {e}")
        raise

    # === TRAINING PHASE ===
    print("\n" + "=" * 60)
    print("PHASE 2: Training with Best Parameters")
    print("=" * 60 + "\n")

    try:
        # Create training config from tuning results
        training_config = ModelTuningConfig.from_tuning_results(
            results_path,
            model_name=model_name,
            epochs=final_epochs,
            num_pitches=input_shape[0],
            max_time_steps=input_shape[1],
        )

        # Save the tuned training config
        ensure_config_dirs()
        config_save_path = TRAINING_CONFIG_DIR / f"{model_name}.json"
        training_config.save(str(config_save_path))
        print(f"Training config saved to: {config_save_path}")

        # Train
        model, history, trainer = train_from_music_dataset(
            datasets=datasets,
            config=training_config,
            vocabulary=dataset.vocabulary,
        )

        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)

        # Save model bundle
        bundle_path = MODELS_DIR / model_name / "model_bundle"
        trainer.save_bundle(str(bundle_path), dataset.vocabulary)
        print(f"Model bundle saved to: {bundle_path}.h5")

        # Summary
        print("\nSaved artifacts:")
        print(f"  - Tuning results: {results_path}")
        print(f"  - Training config: {config_save_path}")
        print(f"  - Model bundle: {bundle_path}.h5")

    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    input("\nPress Enter to continue...")


def run_tune_model():
    """Main entry point for tuning menu."""
    while True:
        print_header("Model Tuning & Training")

        options = [
            "Train LSTM model (pianoroll-based)",
            "Train Transformer model (event-based, autoregressive)",
            "Tune only (LSTM hyperparameter search)",
            "Tune then Train (LSTM full pipeline)",
            "Distributed Training (multi-GPU/multi-node)",
            "Generate SLURM Script (for cluster)",
            "Back to main menu",
        ]
        print_menu(options)
        choice = get_choice(len(options))

        if choice == 1:
            run_training_only()
        elif choice == 2:
            run_transformer_training()
        elif choice == 3:
            run_tuning_only()
        elif choice == 4:
            run_tuning_then_training()
        elif choice == 5:
            run_distributed_training()
        elif choice == 6:
            run_slurm_generator()
        elif choice == 7:
            break
