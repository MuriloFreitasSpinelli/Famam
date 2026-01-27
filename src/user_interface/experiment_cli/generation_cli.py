"""CLI for music generation."""

from pathlib import Path
from typing import Optional, List, Union

from src.music_generation import MusicGenerator, TransformerMusicGenerator, create_generator

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

    # Find all model bundles (exclude checkpoints and _model.h5 weight files)
    bundle_files = list(MODELS_DIR.glob("**/*.h5"))
    bundle_files = [
        f for f in bundle_files
        if "checkpoint" not in str(f).lower()
        and not f.stem.endswith("_model")  # Exclude weight files like foo_model.h5
    ]

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


def select_genre(generator: Union[MusicGenerator, TransformerMusicGenerator]) -> str:
    """Select a genre from available genres."""
    genres = generator.list_genres()
    print("Available genres:")
    print("-" * 40)
    for i, genre in enumerate(genres, 1):
        print(f"  [{i}] {genre}")
    print()

    genre_choice = get_choice(len(genres), "Select genre: ")
    return genres[genre_choice - 1]


def select_instrument(generator: Union[MusicGenerator, TransformerMusicGenerator], genre: Optional[str] = None) -> str:
    """Select an instrument from available instruments."""
    # Get instruments for genre if available, otherwise all active instruments
    # Both generators now support these methods
    if genre:
        instruments = generator.get_instruments_for_genre(genre)
        if not instruments:
            instruments = generator.list_active_instruments()
            if not instruments:
                instruments = generator.list_instruments()[:20]  # Limit to first 20
    else:
        instruments = generator.list_active_instruments()
        if not instruments:
            instruments = generator.list_instruments()[:20]

    print("Available instruments:")
    print("-" * 40)
    for i, inst in enumerate(instruments, 1):
        print(f"  [{i}] {inst}")
    print()

    inst_choice = get_choice(len(instruments), "Select instrument: ")
    return instruments[inst_choice - 1]


def select_multiple_instruments(generator: Union[MusicGenerator, TransformerMusicGenerator], genre: Optional[str] = None) -> List[str]:
    """Select multiple instruments for multi-track generation."""
    # Get instruments for genre if available
    # Both generators now support these methods
    if genre:
        instruments = generator.get_instruments_for_genre(genre)
        if not instruments:
            instruments = generator.list_active_instruments()
            if not instruments:
                instruments = generator.list_instruments()[:20]
    else:
        instruments = generator.list_active_instruments()
        if not instruments:
            instruments = generator.list_instruments()[:20]

    # Always include Drums as an option
    if "Drums" not in instruments:
        instruments = ["Drums"] + list(instruments)

    print("Available instruments (select multiple):")
    print("-" * 40)
    for i, inst in enumerate(instruments, 1):
        print(f"  [{i}] {inst}")
    print()

    print("Enter instrument numbers separated by commas (e.g., 1,3,5)")
    print("Or press Enter for default selection")

    selection = get_input("Instruments", default="")

    if not selection:
        # Default: Drums + first 2-3 instruments
        selected = ["Drums"]
        for inst in instruments:
            if inst != "Drums" and len(selected) < 4:
                selected.append(inst)
        return selected

    # Parse selection
    try:
        indices = [int(x.strip()) for x in selection.split(",")]
        selected = [instruments[i - 1] for i in indices if 1 <= i <= len(instruments)]
        return selected if selected else ["Drums", instruments[1] if len(instruments) > 1 else instruments[0]]
    except (ValueError, IndexError):
        print("Invalid selection, using defaults")
        return ["Drums"] + instruments[1:3] if len(instruments) > 2 else instruments


def run_single_generation(generator: Union[MusicGenerator, TransformerMusicGenerator]):
    """Generate a single instrument MIDI file."""
    print_header("Generate Single Instrument Track")

    # Select genre
    genre = select_genre(generator)

    # Select instrument
    print(f"\nInstruments for '{genre}':")
    instrument = select_instrument(generator, genre)

    # Generation parameters
    print("\nGeneration parameters:")
    print("-" * 40)
    temperature = get_float("Temperature (higher = more random)", default=1.0, min_val=0.1, max_val=2.0)
    tempo = get_float("Tempo (BPM)", default=120.0, min_val=40.0, max_val=240.0)

    # Threshold only applies to LSTM generator
    if isinstance(generator, MusicGenerator):
        threshold = get_float("Threshold (for binarizing notes)", default=0.5, min_val=0.0, max_val=1.0)
    else:
        # Transformer uses top-k/top-p sampling instead
        min_length = get_int("Min sequence length (prevents early EOS)", default=200, min_val=50, max_val=1000)
        top_k = get_int("Top-K sampling (0=disabled)", default=50, min_val=0, max_val=200)
        top_p = get_float("Top-P nucleus sampling (0=disabled)", default=0.9, min_val=0.0, max_val=1.0)

    # Output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inst_clean = instrument.replace(" ", "_").replace("(", "").replace(")", "")
    default_output = str(OUTPUT_DIR / f"generated_{genre}_{inst_clean}.mid")
    output_path = get_path("Output path", default=default_output)

    # Generate
    print(f"\nGenerating {instrument} track for {genre}...")
    try:
        if isinstance(generator, MusicGenerator):
            music = generator.generate_midi(
                genre=genre,
                output_path=output_path,
                instrument=instrument,
                temperature=temperature,
                threshold=threshold,
                tempo=tempo,
            )
        else:
            # TransformerMusicGenerator
            music = generator.generate_midi(
                genre=genre,
                output_path=output_path,
                instrument=instrument,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                tempo=tempo,
            )
        print(f"\nGenerated MIDI saved to: {output_path}")
        print(f"  Instrument: {instrument}")
        print(f"  Tracks: {len(music.tracks)}")
        if music.tracks:
            print(f"  Notes: {sum(len(t.notes) for t in music.tracks)}")

    except Exception as e:
        print(f"\nError during generation: {e}")
        raise

    input("\nPress Enter to continue...")


def run_batch_generation(generator: Union[MusicGenerator, TransformerMusicGenerator]):
    """Generate multiple MIDI files for a single instrument."""
    print_header("Generate Multiple Tracks (Single Instrument)")

    # Check if transformer - batch generation is only available for LSTM
    if isinstance(generator, TransformerMusicGenerator):
        print("Batch generation is only available for LSTM models.")
        print("For transformer models, use single track generation multiple times.")
        input("\nPress Enter to continue...")
        return

    # Select genre
    genre = select_genre(generator)

    # Select instrument
    print(f"\nInstruments for '{genre}':")
    instrument = select_instrument(generator, genre)

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
    print(f"\nGenerating {count} {instrument} tracks...")
    try:
        paths = generator.generate_batch_midi(
            genre=genre,
            output_dir=output_dir,
            instrument=instrument,
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


def run_generate_song(generator: Union[MusicGenerator, TransformerMusicGenerator]):
    """Generate a complete song with drums and multiple instruments."""
    print_header("Generate Complete Song")

    print("This will generate a complete multi-instrument song.")
    print("Drums are generated first, then other instruments are aligned to them.\n")

    # Select genre
    genre = select_genre(generator)

    # Generation parameters
    print("\nGeneration parameters:")
    print("-" * 40)
    temperature = get_float("Temperature (higher = more random)", default=1.0, min_val=0.1, max_val=2.0)
    tempo = get_float("Tempo (BPM)", default=120.0, min_val=40.0, max_val=240.0)

    if isinstance(generator, MusicGenerator):
        num_segments = get_int("Number of segments (more = longer song)", default=1, min_val=1, max_val=16)
        threshold = get_float("Threshold (for binarizing notes)", default=0.5, min_val=0.0, max_val=1.0)
    else:
        # Transformer uses top-k/top-p sampling
        min_length = get_int("Min sequence length (prevents early EOS)", default=200, min_val=50, max_val=1000)
        top_k = get_int("Top-K sampling (0=disabled)", default=50, min_val=0, max_val=200)
        top_p = get_float("Top-P nucleus sampling (0=disabled)", default=0.9, min_val=0.0, max_val=1.0)

    # Output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    default_output = str(OUTPUT_DIR / f"song_{genre}.mid")
    output_path = get_path("Output path", default=default_output)

    # Generate
    print(f"\nGenerating complete song for '{genre}'...")
    try:
        if isinstance(generator, MusicGenerator):
            if num_segments > 1:
                music = generator.generate_extended_song_midi(
                    genre=genre,
                    output_path=output_path,
                    num_segments=num_segments,
                    temperature=temperature,
                    threshold=threshold,
                    tempo=tempo,
                )
            else:
                music = generator.generate_song_midi(
                    genre=genre,
                    output_path=output_path,
                    temperature=temperature,
                    threshold=threshold,
                    tempo=tempo,
                )
        else:
            # TransformerMusicGenerator
            music = generator.generate_song_midi(
                genre=genre,
                output_path=output_path,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                tempo=tempo,
            )

        print(f"\nGenerated song saved to: {output_path}")
        print(f"  Tracks: {len(music.tracks)}")
        for track in music.tracks:
            track_type = "Drums" if track.is_drum else track.name or f"Program {track.program}"
            print(f"    - {track_type}: {len(track.notes)} notes")

    except Exception as e:
        print(f"\nError during generation: {e}")
        raise

    input("\nPress Enter to continue...")


def run_multi_instrument_generation(generator: Union[MusicGenerator, TransformerMusicGenerator]):
    """Generate a multi-instrument track with custom instrument selection."""
    print_header("Generate Multi-Instrument Track (Custom)")

    # Transformer doesn't support custom instrument selection for multi-instrument
    if isinstance(generator, TransformerMusicGenerator):
        print("Custom instrument selection is only available for LSTM models.")
        print("For transformer models, use 'Generate complete song' which auto-selects instruments.")
        input("\nPress Enter to continue...")
        return

    print("Select specific instruments to include in the generation.\n")

    # Select genre
    genre = select_genre(generator)

    # Select instruments
    print(f"\nSelect instruments for '{genre}':")
    instruments = select_multiple_instruments(generator, genre)
    print(f"\nSelected instruments: {', '.join(instruments)}")

    # Generation parameters
    print("\nGeneration parameters:")
    print("-" * 40)
    temperature = get_float("Temperature (higher = more random)", default=1.0, min_val=0.1, max_val=2.0)
    threshold = get_float("Threshold (for binarizing notes)", default=0.5, min_val=0.0, max_val=1.0)
    tempo = get_float("Tempo (BPM)", default=120.0, min_val=40.0, max_val=240.0)

    # Output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    default_output = str(OUTPUT_DIR / f"multi_{genre}.mid")
    output_path = get_path("Output path", default=default_output)

    # Generate
    print(f"\nGenerating multi-instrument track...")
    print(f"Instruments: {', '.join(instruments)}")
    try:
        music = generator.generate_multi_instrument_midi(
            genre=genre,
            output_path=output_path,
            instruments=instruments,
            temperature=temperature,
            threshold=threshold,
            tempo=tempo,
            generate_drums_first="Drums" in instruments,
        )

        print(f"\nGenerated MIDI saved to: {output_path}")
        print(f"  Tracks: {len(music.tracks)}")
        for track in music.tracks:
            track_type = "Drums" if track.is_drum else track.name or f"Program {track.program}"
            print(f"    - {track_type}: {len(track.notes)} notes")

    except Exception as e:
        print(f"\nError during generation: {e}")
        raise

    input("\nPress Enter to continue...")


def show_model_info(generator: Union[MusicGenerator, TransformerMusicGenerator]):
    """Display information about the loaded model."""
    print_header("Model Information")

    print(generator.summary())

    print("\n" + "=" * 60)
    print("Available Genres:")
    print("-" * 40)
    for genre in generator.list_genres():
        # Show top instruments for each genre
        top_instruments = generator.get_top_instruments_for_genre(genre, top_n=3, exclude_drums=True)
        if top_instruments:
            print(f"  {genre}: {', '.join(top_instruments)}")
        else:
            print(f"  {genre}")

    print("\n" + "=" * 60)
    print("Active Instruments:")
    print("-" * 40)
    active = generator.list_active_instruments()
    for i, inst in enumerate(active[:15], 1):
        print(f"  {i}. {inst}")
    if len(active) > 15:
        print(f"  ... and {len(active) - 15} more")

    input("\nPress Enter to continue...")


def run_generate_music():
    """Main entry point for generation menu."""
    print_header("Music Generation")

    # Select model
    bundle_path = select_model_bundle()
    if bundle_path is None:
        return

    # Load generator using factory function that auto-detects model type
    print(f"\nLoading model...")
    try:
        generator = create_generator(bundle_path)
        model_type = "Transformer" if isinstance(generator, TransformerMusicGenerator) else "LSTM"
        print(f"Loaded {model_type} model")
        print(generator.summary())
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to continue...")
        return

    while True:
        print_header("Generation Options")

        # Show model type
        model_type = "Transformer" if isinstance(generator, TransformerMusicGenerator) else "LSTM"
        print(f"Current model type: {model_type}\n")

        options = [
            "Generate complete song (auto instruments)",
            "Generate multi-instrument track (custom)" + (" [LSTM only]" if isinstance(generator, TransformerMusicGenerator) else ""),
            "Generate single instrument track",
            "Generate batch (single instrument)" + (" [LSTM only]" if isinstance(generator, TransformerMusicGenerator) else ""),
            "Show model info",
            "Select different model",
            "Back to main menu",
        ]
        print_menu(options)
        choice = get_choice(len(options))

        if choice == 1:
            run_generate_song(generator)
        elif choice == 2:
            run_multi_instrument_generation(generator)
        elif choice == 3:
            run_single_generation(generator)
        elif choice == 4:
            run_batch_generation(generator)
        elif choice == 5:
            show_model_info(generator)
        elif choice == 6:
            new_bundle = select_model_bundle()
            if new_bundle:
                print(f"\nLoading model...")
                try:
                    generator = create_generator(new_bundle)
                    model_type = "Transformer" if isinstance(generator, TransformerMusicGenerator) else "LSTM"
                    print(f"Loaded {model_type} model")
                    print(generator.summary())
                except Exception as e:
                    print(f"Error loading model: {e}")
        elif choice == 7:
            break
