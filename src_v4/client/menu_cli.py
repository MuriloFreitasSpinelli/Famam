"""
Menu-driven CLI for the music generation pipeline.

Provides an interactive menu interface for:
    - Dataset management
    - Model training
    - Music generation with multiple modes
    - Instrument selection
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from .pipeline import MusicPipeline, PipelineConfig


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str, width: int = 60):
    """Print a styled header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_menu(title: str, options: List[Tuple[str, str]], width: int = 60):
    """Print a menu with numbered options."""
    print_header(title, width)
    for i, (key, desc) in enumerate(options, 1):
        print(f"  [{i}] {desc}")
    print("-" * width)
    print("  [0] Back / Exit")
    print("=" * width)


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default."""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def get_int_input(prompt: str, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    """Get integer input with validation."""
    while True:
        try:
            value = get_input(prompt, str(default))
            value = int(value)
            if min_val is not None and value < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("  Please enter a valid number")


def get_float_input(prompt: str, default: float = 0.0, min_val: float = None, max_val: float = None) -> float:
    """Get float input with validation."""
    while True:
        try:
            value = get_input(prompt, str(default))
            value = float(value)
            if min_val is not None and value < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("  Please enter a valid number")


def get_choice(prompt: str, max_choice: int) -> int:
    """Get menu choice."""
    while True:
        try:
            choice = int(input(f"\n{prompt}: "))
            if 0 <= choice <= max_choice:
                return choice
            print(f"  Please enter 0-{max_choice}")
        except ValueError:
            print("  Please enter a number")


class MenuCLI:
    """Interactive menu-driven CLI for music generation."""

    def __init__(self):
        self.pipeline: Optional[MusicPipeline] = None
        self.config = PipelineConfig()
        self.running = True

        # Generation settings
        self.gen_settings = {
            'temperature': 0.9,
            'top_k': 50,
            'top_p': 0.92,
            'max_length': 512,
            'min_notes': 10,
            'min_bars': 4,
            'max_retries': 3,
        }

    def run(self):
        """Run the menu CLI."""
        clear_screen()
        print_header("Music Generation Pipeline")
        print("  Welcome! Use the menus to navigate.")

        while self.running:
            self.main_menu()

        print("\nGoodbye!")

    # =========================================================================
    # Main Menu
    # =========================================================================

    def main_menu(self):
        """Show main menu."""
        options = [
            ("dataset", "Dataset Management"),
            ("train", "Model Training"),
            ("generate", "Generate Music"),
            ("settings", "Generation Settings"),
            ("info", "View Info"),
        ]

        print_menu("Main Menu", options)

        # Show current status
        if self.pipeline and self.pipeline.dataset:
            print(f"  Dataset: {len(self.pipeline.dataset)} entries")
        if self.pipeline and self.pipeline.model:
            print(f"  Model: Loaded")
        print()

        choice = get_choice("Select option", len(options))

        if choice == 0:
            self.running = False
        elif choice == 1:
            self.dataset_menu()
        elif choice == 2:
            self.training_menu()
        elif choice == 3:
            self.generation_menu()
        elif choice == 4:
            self.settings_menu()
        elif choice == 5:
            self.info_menu()

    # =========================================================================
    # Dataset Menu
    # =========================================================================

    def dataset_menu(self):
        """Dataset management menu."""
        while True:
            options = [
                ("build", "Build Dataset from MIDI"),
                ("load", "Load Existing Dataset"),
                ("info", "Dataset Info"),
                ("instruments", "View Instrument Stats"),
            ]

            print_menu("Dataset Management", options)
            choice = get_choice("Select option", len(options))

            if choice == 0:
                return
            elif choice == 1:
                self.build_dataset()
            elif choice == 2:
                self.load_dataset()
            elif choice == 3:
                self.show_dataset_info()
            elif choice == 4:
                self.show_instrument_stats()

    def build_dataset(self):
        """Build dataset from MIDI files."""
        print_header("Build Dataset")

        midi_dir = get_input("MIDI directory path", "data/midi")
        output_path = get_input("Output file path", "output/dataset.h5")

        if not Path(midi_dir).exists():
            print(f"\n  Error: Directory '{midi_dir}' not found")
            return

        print("\n  Building dataset...")

        try:
            self.pipeline = MusicPipeline(self.config)
            self.pipeline.build_dataset(midi_dir=midi_dir, output_path=output_path)
            print(f"\n  Dataset built: {len(self.pipeline.dataset)} entries")
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"\n  Error: {e}")

        input("\nPress Enter to continue...")

    def load_dataset(self):
        """Load existing dataset."""
        print_header("Load Dataset")

        # List available datasets
        datasets = list(Path(".").rglob("*.h5"))
        if datasets:
            print("\nAvailable datasets:")
            for i, ds in enumerate(datasets[:10], 1):
                print(f"  [{i}] {ds}")
            print()

        filepath = get_input("Dataset file path")

        if not filepath:
            return

        if not Path(filepath).exists():
            print(f"\n  Error: File '{filepath}' not found")
            input("\nPress Enter to continue...")
            return

        print("\n  Loading dataset...")

        try:
            if self.pipeline is None:
                self.pipeline = MusicPipeline(self.config)
            self.pipeline.load_dataset(filepath)
            print(f"\n  Loaded: {len(self.pipeline.dataset)} entries")
        except Exception as e:
            print(f"\n  Error: {e}")

        input("\nPress Enter to continue...")

    def show_dataset_info(self):
        """Show dataset information."""
        print_header("Dataset Info")

        if not self.pipeline or not self.pipeline.dataset:
            print("\n  No dataset loaded.")
            input("\nPress Enter to continue...")
            return

        stats = self.pipeline.dataset.get_stats()
        print(f"\n  Entries: {stats['num_entries']}")
        print(f"  Tracks: {stats['num_tracks']}")
        print(f"  Genres: {stats['num_genres']}")
        print(f"  Active Instruments: {stats['num_active_instruments']}")
        print(f"  Total Notes: {stats['total_notes']}")
        print(f"\n  Genres: {', '.join(stats['genres'][:5])}")
        if len(stats['genres']) > 5:
            print(f"          ... and {len(stats['genres']) - 5} more")

        input("\nPress Enter to continue...")

    def show_instrument_stats(self):
        """Show instrument usage statistics."""
        print_header("Instrument Statistics")

        if not self.pipeline or not self.pipeline.dataset:
            print("\n  No dataset loaded.")
            input("\nPress Enter to continue...")
            return

        top_n = get_int_input("Show top N instruments", 20, 5, 100)
        self.pipeline.print_instrument_stats(top_n)

        input("\nPress Enter to continue...")

    # =========================================================================
    # Training Menu
    # =========================================================================

    def training_menu(self):
        """Model training menu."""
        while True:
            options = [
                ("train", "Train New Model"),
                ("load", "Load Existing Model"),
                ("save", "Save Current Model"),
            ]

            print_menu("Model Training", options)

            if self.pipeline and self.pipeline.model:
                print("  Model: Loaded")
            print()

            choice = get_choice("Select option", len(options))

            if choice == 0:
                return
            elif choice == 1:
                self.train_model()
            elif choice == 2:
                self.load_model()
            elif choice == 3:
                self.save_model()

    def train_model(self):
        """Train a new model."""
        print_header("Train Model")

        if not self.pipeline or not self.pipeline.dataset:
            print("\n  No dataset loaded. Please load a dataset first.")
            input("\nPress Enter to continue...")
            return

        # Get training parameters
        print("\n  Current config:")
        print(f"    Model type: {self.config.model_type}")
        print(f"    d_model: {self.config.d_model}")
        print(f"    Layers: {self.config.num_layers}")
        print(f"    Epochs: {self.config.epochs}")
        print(f"    Multi-track: {self.config.multitrack}")

        print("\n  Enter new values (or press Enter to keep current):")

        # Ask about multi-track mode
        multitrack_str = "y" if self.config.multitrack else "n"
        multitrack_input = get_input("Use multi-track mode (all instruments together)? (y/n)", multitrack_str)
        self.config.multitrack = multitrack_input.lower() == 'y'

        if self.config.multitrack:
            print("\n  MULTI-TRACK MODE: All instruments will be encoded together.")
            print("  The model will learn relationships between all instruments!")
            self.config.encoder_type = "multitrack"

        self.config.epochs = get_int_input("Epochs", self.config.epochs, 1, 1000)
        self.config.batch_size = get_int_input("Batch size", self.config.batch_size, 1, 256)

        confirm = get_input("\n  Start training? (y/n)", "y")
        if confirm.lower() != 'y':
            return

        print("\n  Training model...")
        try:
            # Update pipeline config
            self.pipeline.config = self.config

            # Recreate encoder if multitrack mode changed
            num_genres = self.pipeline.vocabulary.num_genres if self.pipeline.vocabulary else 10
            self.pipeline.create_encoder(num_genres=num_genres)

            self.pipeline.train()
            print("\n  Training complete!")
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()

        input("\nPress Enter to continue...")

    def load_model(self):
        """Load existing model."""
        print_header("Load Model")

        # List available models
        models = list(Path(".").rglob("*_model"))
        h5_models = list(Path(".").rglob("model_bundle.h5"))
        if models or h5_models:
            print("\nAvailable models:")
            all_models = models[:5] + h5_models[:5]
            for i, m in enumerate(all_models, 1):
                print(f"  [{i}] {m}")
            print()

        filepath = get_input("Model file path (without .h5)")

        if not filepath:
            return

        print("\n  Loading model...")
        try:
            if self.pipeline is None:
                self.pipeline = MusicPipeline(self.config)
            self.pipeline.load_model(filepath)
            print("\n  Model loaded successfully!")
        except Exception as e:
            print(f"\n  Error: {e}")

        input("\nPress Enter to continue...")

    def save_model(self):
        """Save current model."""
        print_header("Save Model")

        if not self.pipeline or not self.pipeline.model:
            print("\n  No model to save.")
            input("\nPress Enter to continue...")
            return

        filepath = get_input("Save path", "models/my_model")

        try:
            path = self.pipeline.save_model(filepath)
            print(f"\n  Model saved to: {path}")
        except Exception as e:
            print(f"\n  Error: {e}")

        input("\nPress Enter to continue...")

    # =========================================================================
    # Generation Menu
    # =========================================================================

    def generation_menu(self):
        """Music generation menu."""
        while True:
            # Check if multi-track mode
            is_multitrack = self.pipeline and self.pipeline.is_multitrack_mode() if self.pipeline else False

            if is_multitrack:
                options = [
                    ("multitrack", "Multi-Track (All Instruments Together)"),
                    ("drums", "Drums Only (Rule-Based)"),
                ]
            else:
                options = [
                    ("full", "Full Song (Drums + Top 2 Instruments)"),
                    ("single", "Single Instrument"),
                    ("multi", "Multi Instrument (Custom Selection)"),
                    ("drums", "Drums Only (Rule-Based)"),
                ]

            print_menu("Generate Music", options)

            if not self.pipeline or not self.pipeline.model:
                print("  Warning: No model loaded!")
            else:
                mode_str = "MULTI-TRACK" if is_multitrack else "Single-Track"
                print(f"  Mode: {mode_str}")
            if self.pipeline and self.pipeline.vocabulary:
                print(f"  Available genres: {self.pipeline.vocabulary.num_genres}")
                print(f"  Available instruments: {self.pipeline.vocabulary.num_active_instruments}")
            print()

            choice = get_choice("Select option", len(options))

            if choice == 0:
                return

            if is_multitrack:
                if choice == 1:
                    self.generate_multitrack_song()
                elif choice == 2:
                    self.generate_drums_only()
            else:
                if choice == 1:
                    self.generate_full_song()
                elif choice == 2:
                    self.generate_single_instrument()
                elif choice == 3:
                    self.generate_multi_instrument()
                elif choice == 4:
                    self.generate_drums_only()

    def _show_vocabulary(self):
        """Display available genres and instruments."""
        if not self.pipeline or not self.pipeline.vocabulary:
            print("\n  No vocabulary available. Load a dataset or model first.")
            return False

        vocab = self.pipeline.vocabulary

        # Show genres
        print("\n  Available Genres:")
        genres = list(vocab.genre_to_id.items())
        for i, (name, gid) in enumerate(genres[:10]):
            print(f"    [{gid}] {name}")
        if len(genres) > 10:
            print(f"    ... and {len(genres) - 10} more")

        # Show top instruments
        print("\n  Most Used Instruments:")
        inst_stats = self.pipeline.dataset.get_instrument_stats(10)
        for name, count in inst_stats:
            inst_id = vocab.get_instrument_id(name)
            print(f"    [{inst_id:3d}] {name} ({count} songs)")

        return True

    def _get_genre_selection(self) -> int:
        """Get genre selection from user."""
        if not self.pipeline or not self.pipeline.vocabulary:
            return 0

        genres = list(self.pipeline.vocabulary.genre_to_id.items())
        print("\n  Select Genre:")
        for name, gid in genres[:10]:
            print(f"    [{gid}] {name}")

        return get_int_input("Genre ID", 0, 0, max(g[1] for g in genres))

    def _get_generation_params(self) -> Dict[str, Any]:
        """Get common generation parameters."""
        print("\n  Generation Parameters:")
        print(f"    Temperature: {self.gen_settings['temperature']}")
        print(f"    Min Notes: {self.gen_settings['min_notes']}")
        print(f"    Min Bars: {self.gen_settings['min_bars']}")

        modify = get_input("Modify parameters? (y/n)", "n")
        if modify.lower() == 'y':
            self.gen_settings['temperature'] = get_float_input(
                "Temperature", self.gen_settings['temperature'], 0.1, 2.0
            )
            self.gen_settings['min_notes'] = get_int_input(
                "Min notes per track", self.gen_settings['min_notes'], 1, 1000
            )
            self.gen_settings['min_bars'] = get_int_input(
                "Min bars", self.gen_settings['min_bars'], 1, 64
            )

        return self.gen_settings.copy()

    def generate_full_song(self):
        """Generate full song with drums and top instruments."""
        print_header("Generate Full Song")

        if not self.pipeline or not self.pipeline.model:
            print("\n  No model loaded. Please load a model first.")
            input("\nPress Enter to continue...")
            return

        if not self._show_vocabulary():
            input("\nPress Enter to continue...")
            return

        # Get genre
        genre_id = self._get_genre_selection()

        # Get top 2 instruments for this genre
        vocab = self.pipeline.vocabulary
        top_instruments = vocab.get_top_instruments_for_genre(
            vocab.get_genre_name(genre_id) or "",
            top_n=2,
            exclude_drums=True
        )

        if not top_instruments:
            # Fallback to overall top instruments
            inst_stats = self.pipeline.dataset.get_instrument_stats(3)
            top_instruments = [
                vocab.get_instrument_id(name)
                for name, _ in inst_stats
                if vocab.get_instrument_id(name) != 128
            ][:2]

        print(f"\n  Selected instruments: {[vocab.get_instrument_name(i) for i in top_instruments]}")

        # Get output path
        output_path = get_input("Output MIDI file", "output/full_song.mid")

        # Get generation params
        params = self._get_generation_params()

        print("\n  Generating full song...")
        print("  - Drums track")
        for inst_id in top_instruments:
            print(f"  - {vocab.get_instrument_name(inst_id)}")

        try:
            from ..music_generation import DrumPatternGenerator
            import muspy

            tracks = []

            # Generate drums using rule-based generator
            print("\n  Generating drums (rule-based)...")
            drum_gen = DrumPatternGenerator(
                resolution=self.config.resolution,
                humanize=True,
            )
            drum_track = drum_gen.generate_track(
                num_bars=params['min_bars'] * 2,
                pattern_name="alt_groove",
                add_fills=True,
                add_crashes=True,
            )
            tracks.append(drum_track)
            print(f"    Drums: {len(drum_track.notes)} notes")

            # Generate melodic instruments
            generator = self.pipeline._ensure_generator()

            for inst_id in top_instruments:
                inst_name = vocab.get_instrument_name(inst_id)
                print(f"\n  Generating {inst_name}...")

                track = generator.generate_track(
                    genre_id=genre_id,
                    instrument_id=inst_id,
                    program=inst_id if inst_id < 128 else 0,
                    is_drum=False,
                    name=inst_name,
                    min_notes=params['min_notes'],
                    min_bars=params['min_bars'],
                    max_retries=params['max_retries'],
                    temperature=params['temperature'],
                    max_length=params['max_length'],
                )
                tracks.append(track)

                validation = generator.validate_track(track)
                print(f"    {inst_name}: {validation['note_count']} notes, {validation['bars']} bars")

            # Create music
            music = muspy.Music(
                resolution=self.config.resolution,
                tempos=[muspy.Tempo(time=0, qpm=120)],
                tracks=tracks,
            )

            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            muspy.write_midi(output_path, music)

            print(f"\n  Saved to: {output_path}")
            print(f"  Total tracks: {len(tracks)}")
            print(f"  Total notes: {sum(len(t.notes) for t in tracks)}")

        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()

        input("\nPress Enter to continue...")

    def generate_single_instrument(self):
        """Generate single instrument track."""
        print_header("Generate Single Instrument")

        if not self.pipeline or not self.pipeline.model:
            print("\n  No model loaded. Please load a model first.")
            input("\nPress Enter to continue...")
            return

        if not self._show_vocabulary():
            input("\nPress Enter to continue...")
            return

        # Get selections
        genre_id = self._get_genre_selection()

        # Show instruments and get selection
        print("\n  Select Instrument:")
        vocab = self.pipeline.vocabulary
        inst_stats = self.pipeline.dataset.get_instrument_stats(20)
        for name, count in inst_stats:
            inst_id = vocab.get_instrument_id(name)
            print(f"    [{inst_id:3d}] {name} ({count} songs)")

        instrument_id = get_int_input("Instrument ID", 0, 0, 128)
        inst_name = vocab.get_instrument_name(instrument_id)

        # Output path
        output_path = get_input("Output MIDI file", f"output/{inst_name.replace(' ', '_')}.mid")

        # Generation params
        params = self._get_generation_params()

        print(f"\n  Generating {inst_name}...")

        try:
            import muspy

            generator = self.pipeline._ensure_generator()
            track = generator.generate_track(
                genre_id=genre_id,
                instrument_id=instrument_id,
                program=instrument_id if instrument_id < 128 else 0,
                is_drum=(instrument_id == 128),
                name=inst_name,
                min_notes=params['min_notes'],
                min_bars=params['min_bars'],
                max_retries=params['max_retries'],
                temperature=params['temperature'],
                max_length=params['max_length'],
            )

            validation = generator.validate_track(track)
            print(f"\n    Notes: {validation['note_count']}")
            print(f"    Bars: {validation['bars']}")
            print(f"    Meets min notes: {validation['meets_min_notes']}")
            print(f"    Meets min bars: {validation['meets_min_bars']}")

            # Create music and save
            music = muspy.Music(
                resolution=self.config.resolution,
                tempos=[muspy.Tempo(time=0, qpm=120)],
                tracks=[track],
            )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            muspy.write_midi(output_path, music)
            print(f"\n  Saved to: {output_path}")

        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()

        input("\nPress Enter to continue...")

    def generate_multi_instrument(self):
        """Generate multiple instruments of user's choice."""
        print_header("Generate Multi Instrument")

        if not self.pipeline or not self.pipeline.model:
            print("\n  No model loaded. Please load a model first.")
            input("\nPress Enter to continue...")
            return

        if not self._show_vocabulary():
            input("\nPress Enter to continue...")
            return

        # Get genre
        genre_id = self._get_genre_selection()

        # Get instrument selection
        vocab = self.pipeline.vocabulary
        print("\n  Available Instruments:")
        inst_stats = self.pipeline.dataset.get_instrument_stats(30)
        for name, count in inst_stats:
            inst_id = vocab.get_instrument_id(name)
            print(f"    [{inst_id:3d}] {name} ({count} songs)")

        print("\n  Enter instrument IDs separated by commas (e.g., 0,33,128):")
        inst_input = get_input("Instrument IDs")

        try:
            instrument_ids = [int(x.strip()) for x in inst_input.split(",")]
        except ValueError:
            print("  Invalid input. Please enter comma-separated numbers.")
            input("\nPress Enter to continue...")
            return

        # Validate instruments
        valid_instruments = []
        for inst_id in instrument_ids:
            if 0 <= inst_id <= 128:
                valid_instruments.append(inst_id)
                print(f"    + {vocab.get_instrument_name(inst_id)}")
            else:
                print(f"    - Invalid ID: {inst_id}")

        if not valid_instruments:
            print("\n  No valid instruments selected.")
            input("\nPress Enter to continue...")
            return

        # Include drums?
        include_drums = get_input("Include rule-based drums? (y/n)", "y").lower() == 'y'

        # Output path
        output_path = get_input("Output MIDI file", "output/multi_instrument.mid")

        # Generation params
        params = self._get_generation_params()

        print("\n  Generating tracks...")

        try:
            import muspy
            from ..music_generation import DrumPatternGenerator

            tracks = []
            generator = self.pipeline._ensure_generator()

            # Add rule-based drums if requested
            if include_drums:
                print("\n  Generating drums (rule-based)...")
                drum_gen = DrumPatternGenerator(
                    resolution=self.config.resolution,
                    humanize=True,
                )
                drum_track = drum_gen.generate_track(
                    num_bars=params['min_bars'] * 2,
                    pattern_name="alt_groove",
                )
                tracks.append(drum_track)
                print(f"    Drums: {len(drum_track.notes)} notes")

            # Generate each instrument
            for inst_id in valid_instruments:
                if inst_id == 128 and include_drums:
                    continue  # Skip if already added rule-based drums

                inst_name = vocab.get_instrument_name(inst_id)
                print(f"\n  Generating {inst_name}...")

                track = generator.generate_track(
                    genre_id=genre_id,
                    instrument_id=inst_id,
                    program=inst_id if inst_id < 128 else 0,
                    is_drum=(inst_id == 128),
                    name=inst_name,
                    min_notes=params['min_notes'],
                    min_bars=params['min_bars'],
                    max_retries=params['max_retries'],
                    temperature=params['temperature'],
                    max_length=params['max_length'],
                )
                tracks.append(track)

                validation = generator.validate_track(track)
                print(f"    {inst_name}: {validation['note_count']} notes, {validation['bars']} bars")

            # Create music and save
            music = muspy.Music(
                resolution=self.config.resolution,
                tempos=[muspy.Tempo(time=0, qpm=120)],
                tracks=tracks,
            )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            muspy.write_midi(output_path, music)

            print(f"\n  Saved to: {output_path}")
            print(f"  Total tracks: {len(tracks)}")
            print(f"  Total notes: {sum(len(t.notes) for t in tracks)}")

        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()

        input("\nPress Enter to continue...")

    def generate_drums_only(self):
        """Generate drums using rule-based generator."""
        print_header("Generate Drums (Rule-Based)")

        try:
            from ..music_generation import DrumPatternGenerator
            import muspy

            # Show available patterns
            drum_gen = DrumPatternGenerator(resolution=self.config.resolution)
            patterns = drum_gen.list_patterns()

            print("\n  Available Patterns:")
            for i, name in enumerate(patterns, 1):
                info = drum_gen.get_pattern_info(name)
                print(f"    [{i}] {name}: {info['name']}")

            pattern_idx = get_int_input("Pattern number", 1, 1, len(patterns)) - 1
            pattern_name = patterns[pattern_idx]

            num_bars = get_int_input("Number of bars", 8, 1, 64)
            add_fills = get_input("Add fills? (y/n)", "y").lower() == 'y'
            add_crashes = get_input("Add crashes? (y/n)", "y").lower() == 'y'

            output_path = get_input("Output MIDI file", "output/drums.mid")

            print(f"\n  Generating {num_bars} bars of {pattern_name}...")

            drum_track = drum_gen.generate_track(
                num_bars=num_bars,
                pattern_name=pattern_name,
                add_fills=add_fills,
                add_crashes=add_crashes,
            )

            music = muspy.Music(
                resolution=self.config.resolution,
                tempos=[muspy.Tempo(time=0, qpm=120)],
                tracks=[drum_track],
            )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            muspy.write_midi(output_path, music)

            print(f"\n  Generated: {len(drum_track.notes)} drum notes")
            print(f"  Saved to: {output_path}")

        except Exception as e:
            print(f"\n  Error: {e}")

        input("\nPress Enter to continue...")

    def generate_multitrack_song(self):
        """Generate song with all instruments knowing each other."""
        print_header("Multi-Track Generation")
        print("\n  All instruments will be generated together!")
        print("  They will be aware of each other's parts.")

        if not self.pipeline or not self.pipeline.model:
            print("\n  No model loaded. Please load a model first.")
            input("\nPress Enter to continue...")
            return

        if not self._show_vocabulary():
            input("\nPress Enter to continue...")
            return

        # Get parameters
        genre_id = self._get_genre_selection()
        num_bars = get_int_input("Number of bars", 8, 1, 32)
        output_path = get_input("Output MIDI file", "output/multitrack_song.mid")

        # Get generation params
        params = self._get_generation_params()

        print("\n  Generating multi-track music...")
        print("  All instruments are being generated together, aware of each other.")

        try:
            music = self.pipeline.generate_multitrack(
                output_path=output_path,
                genre_id=genre_id,
                num_bars=num_bars,
                temperature=params['temperature'],
            )

            print(f"\n  Saved to: {output_path}")

        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()

        input("\nPress Enter to continue...")

    # =========================================================================
    # Settings Menu
    # =========================================================================

    def settings_menu(self):
        """Generation settings menu."""
        while True:
            print_header("Generation Settings")

            print("\n  Current Settings:")
            print(f"    [1] Temperature: {self.gen_settings['temperature']}")
            print(f"    [2] Top-K: {self.gen_settings['top_k']}")
            print(f"    [3] Top-P: {self.gen_settings['top_p']}")
            print(f"    [4] Max Length: {self.gen_settings['max_length']}")
            print(f"    [5] Min Notes: {self.gen_settings['min_notes']}")
            print(f"    [6] Min Bars: {self.gen_settings['min_bars']}")
            print(f"    [7] Max Retries: {self.gen_settings['max_retries']}")
            print("-" * 60)
            print("    [0] Back")
            print("=" * 60)

            choice = get_choice("Select setting to change", 7)

            if choice == 0:
                return
            elif choice == 1:
                self.gen_settings['temperature'] = get_float_input(
                    "Temperature", self.gen_settings['temperature'], 0.1, 2.0
                )
            elif choice == 2:
                self.gen_settings['top_k'] = get_int_input(
                    "Top-K", self.gen_settings['top_k'], 1, 500
                )
            elif choice == 3:
                self.gen_settings['top_p'] = get_float_input(
                    "Top-P", self.gen_settings['top_p'], 0.1, 1.0
                )
            elif choice == 4:
                self.gen_settings['max_length'] = get_int_input(
                    "Max Length", self.gen_settings['max_length'], 64, 2048
                )
            elif choice == 5:
                self.gen_settings['min_notes'] = get_int_input(
                    "Min Notes", self.gen_settings['min_notes'], 1, 1000
                )
            elif choice == 6:
                self.gen_settings['min_bars'] = get_int_input(
                    "Min Bars", self.gen_settings['min_bars'], 1, 64
                )
            elif choice == 7:
                self.gen_settings['max_retries'] = get_int_input(
                    "Max Retries", self.gen_settings['max_retries'], 1, 10
                )

    # =========================================================================
    # Info Menu
    # =========================================================================

    def info_menu(self):
        """Information menu."""
        while True:
            options = [
                ("pipeline", "Pipeline Status"),
                ("model", "Model Info"),
                ("config", "Current Config"),
            ]

            print_menu("Information", options)
            choice = get_choice("Select option", len(options))

            if choice == 0:
                return
            elif choice == 1:
                self.show_pipeline_status()
            elif choice == 2:
                self.show_model_info()
            elif choice == 3:
                self.show_config()

    def show_pipeline_status(self):
        """Show pipeline status."""
        print_header("Pipeline Status")

        if self.pipeline:
            print(self.pipeline.summary())
        else:
            print("\n  No pipeline initialized.")

        input("\nPress Enter to continue...")

    def show_model_info(self):
        """Show model information."""
        print_header("Model Info")

        if self.pipeline and self.pipeline.bundle:
            print(self.pipeline.bundle.summary())
        elif self.pipeline and self.pipeline.model:
            print("\n  Model loaded but no bundle info available.")
        else:
            print("\n  No model loaded.")

        input("\nPress Enter to continue...")

    def show_config(self):
        """Show current configuration."""
        print_header("Current Configuration")

        import json
        print(json.dumps(self.config.__dict__, indent=2))

        input("\nPress Enter to continue...")


def main():
    """Main entry point for menu CLI."""
    cli = MenuCLI()
    cli.run()


if __name__ == "__main__":
    main()
