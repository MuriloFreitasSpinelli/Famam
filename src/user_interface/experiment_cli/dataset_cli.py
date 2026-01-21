"""CLI for dataset creation."""

from pathlib import Path
from typing import Optional, Tuple

from src.data_processing import (
    MusicDatasetConfig,
    PreprocessingConfig,
    build_and_save_dataset,
)

from .prompts import (
    print_header,
    print_menu,
    get_choice,
    get_input,
    get_int,
    get_float,
    get_bool,
    get_optional_int,
    get_optional_float,
    get_path,
    get_list_input,
    get_optional_list,
    select_config_file,
    confirm,
)


# Default config directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs"
MUSIC_DATASET_CONFIG_DIR = CONFIGS_DIR / "music_dataset"
PREPROCESSING_CONFIG_DIR = CONFIGS_DIR / "preprocessing"


def ensure_config_dirs():
    """Ensure all config directories exist."""
    MUSIC_DATASET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def prompt_music_dataset_config() -> MusicDatasetConfig:
    """Interactively create a MusicDatasetConfig."""
    print_header("Create Music Dataset Configuration")

    print("Required fields:")
    print("-" * 40)

    name = get_input("Dataset name", default="my_dataset")

    print("\nInput directories (where MIDI files are located):")
    default_input = str(PROJECT_ROOT / "data" / "midi")
    input_dirs = get_list_input("Input directories", default=[default_input])

    default_output = str(PROJECT_ROOT / "data" / "datasets" / f"{name}.h5")
    output_path = get_path("Output path (.h5 file)", default=default_output)

    print("\nOptional fields (press Enter for defaults):")
    print("-" * 40)

    genre_tsv_path: Optional[str] = None
    if get_bool("Do you have a genre.tsv file?", default=False):
        genre_tsv_path = get_path("Genre TSV path", must_exist=True)

    allowed_genres = get_optional_list("Allowed genres (filter by genre)")

    print("\nTrack filtering:")
    min_tracks = get_int("Minimum tracks per file", default=1, min_val=1)
    max_tracks = get_int("Maximum tracks per file", default=16, min_val=min_tracks)
    min_notes_per_track = get_int("Minimum notes per track", default=1, min_val=0)

    print("\nDuration filtering:")
    min_duration = get_optional_float("Minimum duration (seconds)")
    max_duration = get_optional_float("Maximum duration (seconds)")

    print("\nPianoroll settings:")
    resolution = get_int("Resolution (ticks per quarter note)", default=24, min_val=1)
    max_time_steps = get_int("Max time steps", default=1000, min_val=1)

    print("\nProcessing options:")
    max_samples = get_optional_int("Max samples (for testing, empty for all)")
    random_seed = get_optional_int("Random seed")
    verbose = get_bool("Verbose output", default=True)

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
        max_time_steps=max_time_steps,
        max_samples=max_samples,
        random_seed=random_seed,
        verbose=verbose,
    )

    # Always save config
    ensure_config_dirs()
    save_path = MUSIC_DATASET_CONFIG_DIR / f"{name}.json"
    config.save(str(save_path))
    print(f"\nConfiguration saved to: {save_path}")

    return config


def prompt_preprocessing_config() -> PreprocessingConfig:
    """Interactively create a PreprocessingConfig."""
    print_header("Create Preprocessing Configuration")

    print("Resolution settings:")
    print("-" * 40)
    target_resolution = get_int("Target resolution (ticks per quarter)", default=24, min_val=1)

    print("\nQuantization:")
    quantize = get_bool("Enable quantization", default=True)
    quantize_grid = get_int("Quantize grid (ticks)", default=1, min_val=1) if quantize else 1

    print("\nTrack cleanup:")
    remove_empty = get_bool("Remove empty tracks", default=True)

    print("\nSegmentation (split music into fixed-length segments):")
    enable_segmentation = get_bool("Enable segmentation", default=False)
    segment_length: Optional[int] = None
    max_padding_ratio = 0.7
    if enable_segmentation:
        segment_length = get_int("Segment length (time steps)", default=2400, min_val=1)
        max_padding_ratio = get_float(
            "Max padding ratio (discard if padding exceeds this)",
            default=0.7,
            min_val=0.0,
            max_val=1.0,
        )

    print("\nAugmentation - Transposition:")
    enable_transposition = get_bool("Enable transposition augmentation", default=False)
    transposition_semitones = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
    if enable_transposition:
        print("Default semitones: -5 to +6 (excluding 0)")
        if get_bool("Use custom semitone range?", default=False):
            semitones_str = get_input(
                "Semitones (comma-separated integers)",
                default="-5,-4,-3,-2,-1,1,2,3,4,5,6",
            )
            transposition_semitones = tuple(int(x.strip()) for x in semitones_str.split(","))

    print("\nAugmentation - Tempo Variation:")
    enable_tempo = get_bool("Enable tempo variation augmentation", default=False)
    tempo_range = (0.9, 1.1)
    tempo_steps = 3
    if enable_tempo:
        tempo_min = get_float("Tempo variation min", default=0.9, min_val=0.1)
        tempo_max = get_float("Tempo variation max", default=1.1, min_val=tempo_min)
        tempo_range = (tempo_min, tempo_max)
        tempo_steps = get_int("Tempo variation steps", default=3, min_val=1)

    config = PreprocessingConfig(
        target_resolution=target_resolution,
        quantize=quantize,
        quantize_grid=quantize_grid,
        remove_empty_tracks=remove_empty,
        segment_length=segment_length,
        max_padding_ratio=max_padding_ratio,
        enable_transposition=enable_transposition,
        transposition_semitones=transposition_semitones,
        enable_tempo_variation=enable_tempo,
        tempo_variation_range=tempo_range,
        tempo_variation_steps=tempo_steps,
    )

    # Always save config
    ensure_config_dirs()
    config_name = get_input("Configuration name", default="custom")
    save_path = PREPROCESSING_CONFIG_DIR / f"{config_name}.json"
    config.save(str(save_path))
    print(f"\nConfiguration saved to: {save_path}")

    return config


def get_music_dataset_config() -> Optional[MusicDatasetConfig]:
    """Get MusicDatasetConfig either from file or by creating new one."""
    print_header("Music Dataset Configuration")

    options = [
        "Load existing config",
        "Create new config",
        "Back",
    ]
    print_menu(options)
    choice = get_choice(len(options))

    if choice == 1:
        ensure_config_dirs()
        config_path = select_config_file(MUSIC_DATASET_CONFIG_DIR, "dataset")
        if config_path:
            config = MusicDatasetConfig.load(str(config_path))
            print(f"\nLoaded config: {config.name}")
            print(f"  Input dirs: {config.input_dirs}")
            print(f"  Output: {config.output_path}")
            return config
        return None
    elif choice == 2:
        return prompt_music_dataset_config()
    else:
        return None


def get_preprocessing_config() -> Optional[PreprocessingConfig]:
    """Get PreprocessingConfig either from file or by creating new one."""
    print_header("Preprocessing Configuration")

    options = [
        "Load existing config",
        "Create new config",
        "Use defaults (no augmentation)",
        "Back",
    ]
    print_menu(options)
    choice = get_choice(len(options))

    if choice == 1:
        ensure_config_dirs()
        config_path = select_config_file(PREPROCESSING_CONFIG_DIR, "preprocessing")
        if config_path:
            config = PreprocessingConfig.load(str(config_path))
            print(f"\nLoaded preprocessing config:")
            print(f"  Resolution: {config.target_resolution}")
            print(f"  Quantize: {config.quantize}")
            print(f"  Segmentation: {config.segment_length or 'disabled'}")
            print(f"  Transposition: {'enabled' if config.enable_transposition else 'disabled'}")
            print(f"  Tempo variation: {'enabled' if config.enable_tempo_variation else 'disabled'}")
            return config
        return None
    elif choice == 2:
        return prompt_preprocessing_config()
    elif choice == 3:
        return PreprocessingConfig.default()
    else:
        return None


def run_dataset_creation():
    """Run the full dataset creation workflow."""
    print_header("Create New Dataset")

    # Step 1: Get Music Dataset Config
    dataset_config = get_music_dataset_config()
    if dataset_config is None:
        print("Dataset creation cancelled.")
        return

    # Step 2: Get Preprocessing Config
    preprocessing_config = get_preprocessing_config()
    if preprocessing_config is None:
        print("Dataset creation cancelled.")
        return

    # Step 3: Confirm and build
    print_header("Configuration Summary")
    print("Dataset Configuration:")
    print(f"  Name: {dataset_config.name}")
    print(f"  Input dirs: {dataset_config.input_dirs}")
    print(f"  Output: {dataset_config.output_path}")
    print(f"  Max samples: {dataset_config.max_samples or 'all'}")
    print()
    print("Preprocessing Configuration:")
    print(f"  Resolution: {preprocessing_config.target_resolution}")
    print(f"  Segmentation: {preprocessing_config.segment_length or 'disabled'}")
    print(f"  Transposition: {'enabled' if preprocessing_config.enable_transposition else 'disabled'}")
    print(f"  Tempo variation: {'enabled' if preprocessing_config.enable_tempo_variation else 'disabled'}")
    print()

    if not confirm("Proceed with dataset creation?", default=True):
        print("Dataset creation cancelled.")
        return

    # Build the dataset
    print()
    print("=" * 60)
    print("Building dataset...")
    print("=" * 60)
    print()

    try:
        dataset = build_and_save_dataset(dataset_config, preprocessing_config)

        print()
        print("=" * 60)
        print("Dataset created successfully!")
        print("=" * 60)
        print(f"  Entries: {len(dataset)}")
        print(f"  Total tracks: {dataset.count_tracks()}")
        print(f"  Genres: {list(dataset.vocabulary.genre_to_id.keys())}")
        print(f"  Output: {dataset_config.output_path}")

    except Exception as e:
        print()
        print("=" * 60)
        print(f"Error creating dataset: {e}")
        print("=" * 60)
        raise

    input("\nPress Enter to continue...")
