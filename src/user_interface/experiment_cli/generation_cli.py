"""CLI for music generation."""

from pathlib import Path
from typing import Optional

from src.music_generation import MusicGenerator

from .prompts import (
    print_header,
    print_menu,
    get_choice,
    get_input,
    get_int,
    get_float,
    get_path,
    confirm,
)


# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"


def select_model_bundle() -> Optional[str]:
    """Select a model bundle for generation."""
    print_header("Select Model")

    # Find all model bundles
    bundle_files = list(MODELS_DIR.glob("**/*.h5"))
    bundle_files = [f for f in bundle_files if "checkpoint" not in str(f).lower()]

    if not bundle_files:
        print(f"No model bundles found in {MODELS_DIR}")
        print("Please train a model first using 'Tune & Train Model' option.")
        input("\nPress Enter to continue...")
        return None

    print("Available models:")
    print("-" * 40)
    for i, f in enumerate(bundle_files, 1):
        rel_path = f.relative_to(MODELS_DIR)
        print(f"  [{i}] {rel_path}")
    print()

    choice = get_choice(len(bundle_files), "Select model: ")
    return str(bundle_files[choice - 1])


def run_single_generation(generator: MusicGenerator):
    """Generate a single MIDI file."""
    print_header("Generate Single Track")

    # Select genre
    genres = generator.list_genres()
    print("Available genres:")
    print("-" * 40)
    for i, genre in enumerate(genres, 1):
        print(f"  [{i}] {genre}")
    print()

    genre_choice = get_choice(len(genres), "Select genre: ")
    genre = genres[genre_choice - 1]

    # Generation parameters
    print("\nGeneration parameters:")
    print("-" * 40)
    temperature = get_float("Temperature (higher = more random)", default=1.0, min_val=0.1, max_val=2.0)
    threshold = get_float("Threshold (for binarizing notes)", default=0.5, min_val=0.0, max_val=1.0)
    tempo = get_float("Tempo (BPM)", default=120.0, min_val=40.0, max_val=240.0)

    # Output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    default_output = str(OUTPUT_DIR / f"generated_{genre}.mid")
    output_path = get_path("Output path", default=default_output)

    # Generate
    print("\nGenerating...")
    try:
        music = generator.generate_midi(
            genre=genre,
            output_path=output_path,
            temperature=temperature,
            threshold=threshold,
            tempo=tempo,
        )
        print(f"\nGenerated MIDI saved to: {output_path}")
        print(f"  Tracks: {len(music.tracks)}")
        if music.tracks:
            print(f"  Notes: {sum(len(t.notes) for t in music.tracks)}")

    except Exception as e:
        print(f"\nError during generation: {e}")
        raise

    input("\nPress Enter to continue...")


def run_batch_generation(generator: MusicGenerator):
    """Generate multiple MIDI files."""
    print_header("Generate Multiple Tracks")

    # Select genre
    genres = generator.list_genres()
    print("Available genres:")
    print("-" * 40)
    for i, genre in enumerate(genres, 1):
        print(f"  [{i}] {genre}")
    print()

    genre_choice = get_choice(len(genres), "Select genre: ")
    genre = genres[genre_choice - 1]

    # Generation parameters
    print("\nGeneration parameters:")
    print("-" * 40)
    count = get_int("Number of tracks to generate", default=5, min_val=1, max_val=100)
    temperature = get_float("Temperature", default=1.0, min_val=0.1, max_val=2.0)
    threshold = get_float("Threshold", default=0.5, min_val=0.0, max_val=1.0)
    tempo = get_float("Tempo (BPM)", default=120.0, min_val=40.0, max_val=240.0)

    # Output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    default_output_dir = str(OUTPUT_DIR / genre)
    output_dir = get_path("Output directory", default=default_output_dir)

    # Generate
    print(f"\nGenerating {count} tracks...")
    try:
        paths = generator.generate_batch_midi(
            genre=genre,
            output_dir=output_dir,
            count=count,
            temperature=temperature,
            threshold=threshold,
            tempo=tempo,
        )

        print(f"\nGenerated {len(paths)} MIDI files:")
        for p in paths[:5]:
            print(f"  - {Path(p).name}")
        if len(paths) > 5:
            print(f"  ... and {len(paths) - 5} more")

    except Exception as e:
        print(f"\nError during generation: {e}")
        raise

    input("\nPress Enter to continue...")


def run_generate_music():
    """Main entry point for generation menu."""
    print_header("Music Generation")

    # Select model
    bundle_path = select_model_bundle()
    if bundle_path is None:
        return

    # Load generator
    print(f"\nLoading model...")
    try:
        generator = MusicGenerator.from_bundle_path(bundle_path)
        print(generator.summary())
    except Exception as e:
        print(f"Error loading model: {e}")
        input("\nPress Enter to continue...")
        return

    while True:
        print_header("Generation Options")

        options = [
            "Generate single track",
            "Generate multiple tracks",
            "Select different model",
            "Back to main menu",
        ]
        print_menu(options)
        choice = get_choice(len(options))

        if choice == 1:
            run_single_generation(generator)
        elif choice == 2:
            run_batch_generation(generator)
        elif choice == 3:
            new_bundle = select_model_bundle()
            if new_bundle:
                print(f"\nLoading model...")
                generator = MusicGenerator.from_bundle_path(new_bundle)
                print(generator.summary())
        elif choice == 4:
            break
