"""
Generation CLI - Menu-based interface for music generation.

Features:
    - Load model bundles and view details
    - Display vocabulary info (genres, instruments)
    - Interactive generation with all parameters
    - Song continuation/concatenation for longer pieces
    - Generate to specific durations

Author: Murilo de Freitas Spinelli
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import numpy as np

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if TYPE_CHECKING:
    from ..models import ModelBundle


# =============================================================================
# UI Helpers
# =============================================================================

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


def wait_for_enter():
    """Wait for user to press Enter."""
    input("\nPress Enter to continue...")


# =============================================================================
# Generation CLI Class
# =============================================================================

class GenerationCLI:
    """Menu-driven CLI for music generation."""

    def __init__(self):
        self.running = True
        self.bundle: Optional["ModelBundle"] = None

        # Default generation settings
        self.settings = {
            'temperature': 0.9,
            'top_k': 50,
            'top_p': 0.92,
            'min_length': 256,
        }

    # =========================================================================
    # Instrument Helpers
    # =========================================================================

    def _get_top_instruments(self, top_n: int = 20) -> List[tuple]:
        """Return the top N most-used instruments as (program_id, name, count) tuples."""
        from ..data.vocabulary import INSTRUMENT_NAME_TO_ID
        vocab = self.bundle.vocabulary
        stats = vocab.get_instrument_stats()
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        result = []
        for name, count in sorted_stats[:top_n]:
            prog_id = INSTRUMENT_NAME_TO_ID.get(name, -1)
            if prog_id >= 0:
                result.append((prog_id, name, count))
        return result

    def _select_instruments(self, genre_id: int) -> Optional[List[int]]:
        """
        Interactive instrument selection backed by vocabulary frequency data.

        Returns a list of allowed program IDs, or None (no filtering).
        """
        if not self.bundle or not self.bundle.vocabulary:
            return None

        from ..data.vocabulary import GENERAL_MIDI_INSTRUMENTS

        top_instruments = self._get_top_instruments(20)
        if not top_instruments:
            return None

        print("\n  Most used instruments in training data:")
        print(f"  {'Rank':<5} {'ID':<5} {'Instrument':<32} {'Songs'}")
        print("  " + "-" * 55)
        for rank, (prog_id, name, count) in enumerate(top_instruments, 1):
            tag = " [DRUMS]" if prog_id == 128 else ""
            print(f"  {rank:<5} {prog_id:<5} {name:<32}{count}{tag}")

        print()
        print("  [1] Use top 5  (recommended)")
        print("  [2] Use top 10")
        print("  [3] Enter custom IDs")
        print("  [0] No filtering (model decides freely)")
        print("-" * 60)

        choice = get_choice("Instrument selection", 3)

        if choice == 0:
            return None

        if choice in (1, 2):
            n = 5 if choice == 1 else 10
            selected = [prog_id for prog_id, _, _ in top_instruments[:n]]
        else:
            print("  Enter space-separated MIDI program IDs (0-127, or 128 for drums).")
            print("  Example: 0 33 25 128  → Piano, Electric Bass, Steel Guitar, Drums")
            raw = get_input("Instrument IDs")
            if not raw:
                return None
            try:
                ids = [int(x) for x in raw.split()]
                selected = [i for i in ids if 0 <= i <= 128]
            except ValueError:
                print("  Invalid input — no filtering applied.")
                return None

        if selected:
            names = [GENERAL_MIDI_INSTRUMENTS.get(p, "Drums") for p in selected]
            print(f"\n  Using instruments: {names}")
        return selected or None

    def _remap_instruments(self, music, allowed_programs: List[int]):
        """
        Remap track programs to the nearest allowed instrument, then merge tracks
        that share the same program after remapping.
        """
        import muspy
        from ..data.vocabulary import GENERAL_MIDI_INSTRUMENTS

        allowed_melodic = [p for p in allowed_programs if p != 128]
        allow_drums = 128 in allowed_programs

        merged: Dict[int, Any] = {}  # program_key -> muspy.Track

        for track in music.tracks:
            if track.is_drum:
                if not allow_drums:
                    continue
                key = 128
            else:
                if not allowed_melodic:
                    continue
                prog = track.program
                if prog in allowed_melodic:
                    key = prog
                else:
                    key = min(allowed_melodic, key=lambda x: abs(x - prog))

            if key not in merged:
                is_drum = key == 128
                merged[key] = muspy.Track(
                    program=0 if is_drum else key,
                    is_drum=is_drum,
                    name="Drums" if is_drum else GENERAL_MIDI_INSTRUMENTS.get(key, f"Program {key}"),
                    notes=[],
                )
            merged[key].notes.extend(track.notes)

        for track in merged.values():
            track.notes.sort(key=lambda n: n.time)

        music.tracks = list(merged.values())
        return music

    def run(self):
        """Run the generation CLI."""
        clear_screen()
        print_header("Music Generation CLI")
        print("  Generate music from trained models.\n")

        while self.running:
            self.main_menu()

        print("\nGoodbye!")

    # =========================================================================
    # Main Menu
    # =========================================================================

    def main_menu(self):
        """Show main menu."""
        options = [
            "Load Model",
            "View Model Info",
            "View Genres",
            "View Instruments",
            "Generate Music",
            "Generation Settings",
        ]
        print_menu("Main Menu", options)

        # Show current status
        if self.bundle:
            print(f"  Model: {self.bundle.model_name}")
            if self.bundle.vocabulary:
                print(f"  Genres: {self.bundle.vocabulary.num_genres}")
                print(f"  Instruments: {self.bundle.vocabulary.num_active_instruments}")
        else:
            print("  No model loaded")
        print()

        choice = get_choice("Select option", len(options))

        if choice == 0:
            self.running = False
        elif choice == 1:
            self.load_model()
        elif choice == 2:
            self.view_model_info()
        elif choice == 3:
            self.view_genres()
        elif choice == 4:
            self.view_instruments()
        elif choice == 5:
            self.generate_music()
        elif choice == 6:
            self.settings_menu()

    # =========================================================================
    # Load Model
    # =========================================================================

    def load_model(self):
        """Load a model bundle."""
        from ..models import load_model_bundle

        print_header("Load Model")

        # Find model bundles in models directory only
        model_bundles = []
        models_dir = Path("models")
        if models_dir.exists():
            model_bundles = list(models_dir.rglob("*.h5"))

        if model_bundles:
            print("\nFound model bundles:")
            for i, b in enumerate(model_bundles[:10], 1):
                print(f"  [{i}] {b}")
            if len(model_bundles) > 10:
                print(f"  ... and {len(model_bundles) - 10} more")
            print()

        bundle_path = get_input("Model bundle path (.h5)")
        if not bundle_path:
            return

        # Handle numeric selection
        if bundle_path.isdigit():
            idx = int(bundle_path) - 1
            if 0 <= idx < len(model_bundles):
                bundle_path = str(model_bundles[idx])

        if not Path(bundle_path).exists():
            print(f"\nFile not found: {bundle_path}")
            wait_for_enter()
            return

        print(f"\nLoading model from: {bundle_path}")
        try:
            self.bundle = load_model_bundle(bundle_path)
            print(f"\nModel loaded successfully!")
            print(f"  Name: {self.bundle.model_name}")
            print(f"  Type: {self.bundle.model_type}")
            print(f"  Vocab size: {self.bundle.vocab_size}")
            print(f"  Max sequence length: {self.bundle.max_seq_length}")
            if self.bundle.vocabulary:
                print(f"  Genres: {self.bundle.vocabulary.num_genres}")
                print(f"  Active instruments: {self.bundle.vocabulary.num_active_instruments}")
        except Exception as e:
            print(f"\nError loading model: {e}")
            import traceback
            traceback.print_exc()

        wait_for_enter()

    # =========================================================================
    # View Model Info
    # =========================================================================

    def view_model_info(self):
        """View detailed model information."""
        print_header("Model Information")

        if not self.bundle:
            print("\n  No model loaded. Use 'Load Model' first.")
            wait_for_enter()
            return

        print(self.bundle.summary())
        wait_for_enter()

    # =========================================================================
    # View Genres
    # =========================================================================

    def view_genres(self):
        """View available genres from the model's vocabulary."""
        print_header("Available Genres")

        if not self.bundle:
            print("\n  No model loaded.")
            wait_for_enter()
            return

        if not self.bundle.vocabulary:
            print("\n  No vocabulary in model bundle.")
            print("  (Model may have been trained without vocabulary info)")
            wait_for_enter()
            return

        vocab = self.bundle.vocabulary
        genres = list(vocab.genre_to_id.items())

        if not genres:
            print("\n  No genres found in vocabulary.")
            wait_for_enter()
            return

        print(f"\n  Total genres: {len(genres)}")
        print("\n  ID  | Genre Name")
        print("  " + "-" * 40)

        for name, genre_id in sorted(genres, key=lambda x: x[1]):
            print(f"  {genre_id:3d} | {name}")

        wait_for_enter()

    # =========================================================================
    # View Instruments
    # =========================================================================

    def view_instruments(self):
        """View instrument statistics from the model's vocabulary."""
        print_header("Instrument Statistics")

        if not self.bundle:
            print("\n  No model loaded.")
            wait_for_enter()
            return

        if not self.bundle.vocabulary:
            print("\n  No vocabulary in model bundle.")
            wait_for_enter()
            return

        vocab = self.bundle.vocabulary
        instrument_stats = vocab.get_instrument_stats()

        if not instrument_stats:
            print("\n  No instrument usage data available.")
            wait_for_enter()
            return

        # Sort by usage count
        sorted_stats = sorted(instrument_stats.items(), key=lambda x: x[1], reverse=True)

        top_n = get_int("Show top N instruments", 20, min_val=1, max_val=len(sorted_stats))

        print(f"\n  Top {top_n} Most Used Instruments")
        print("\n  Rank | ID  | Instrument                    | Songs")
        print("  " + "-" * 55)

        from ..data.vocabulary import INSTRUMENT_NAME_TO_ID

        for rank, (name, count) in enumerate(sorted_stats[:top_n], 1):
            inst_id = INSTRUMENT_NAME_TO_ID.get(name, -1)
            print(f"  {rank:4d} | {inst_id:3d} | {name:<29} | {count}")

        total = vocab.num_active_instruments
        if total > top_n:
            print(f"\n  ... and {total - top_n} more instruments")

        wait_for_enter()

    # =========================================================================
    # Generate Music
    # =========================================================================

    def _create_poly_generator(self, max_length: Optional[int] = None):
        """Create a PolyGenerator from the loaded bundle."""
        from ..generation.poly_generator import PolyGenerator, PolyGeneratorConfig

        config = PolyGeneratorConfig(
            max_length=max_length or self.bundle.max_seq_length,
            temperature=self.settings['temperature'],
            top_k=self.settings['top_k'],
            top_p=self.settings['top_p'],
            resolution=self.bundle.encoder.resolution,
        )
        return PolyGenerator(
            model=self.bundle.model,
            encoder=self.bundle.encoder,
            config=config,
        )

    def generate_music(self):
        """Interactive music generation."""
        import muspy
        from ..data import MultiTrackEncoder

        print_header("Generate Music")

        if not self.bundle:
            print("\n  No model loaded. Use 'Load Model' first.")
            wait_for_enter()
            return

        is_multitrack = isinstance(self.bundle.encoder, MultiTrackEncoder)

        # Show model capabilities
        print(f"\n  Model: {self.bundle.model_name}")
        print(f"  Max sequence length: {self.bundle.max_seq_length}")
        if is_multitrack:
            print(f"  Mode: Multi-track (PolyGenerator)")

        # Show available genres
        if self.bundle.vocabulary:
            vocab = self.bundle.vocabulary
            genres = list(vocab.genre_to_id.items())
            if genres:
                print(f"\n  Available Genres ({len(genres)}):")
                for name, gid in sorted(genres, key=lambda x: x[1])[:10]:
                    print(f"    [{gid}] {name}")
                if len(genres) > 10:
                    print(f"    ... and {len(genres) - 10} more")

        # Get generation parameters
        print("\n--- Generation Parameters ---")

        genre_id = get_int("Genre ID (see list above)", 0, min_val=0)

        # Instrument selection (multitrack only)
        allowed_instruments = None
        if is_multitrack:
            allowed_instruments = self._select_instruments(genre_id)

        print(f"\n  Current settings:")
        print(f"    Temperature: {self.settings['temperature']}")
        print(f"    Top-k: {self.settings['top_k']}")
        print(f"    Top-p: {self.settings['top_p']}")

        modify_settings = get_bool("Modify settings", False)
        if modify_settings:
            self.settings['temperature'] = get_float("Temperature (0.1-2.0)", self.settings['temperature'], 0.1, 2.0)
            self.settings['top_k'] = get_int("Top-k (0=disabled)", self.settings['top_k'], 0, 500)
            self.settings['top_p'] = get_float("Top-p (0.0-1.0)", self.settings['top_p'], 0.0, 1.0)

        if is_multitrack:
            num_bars = get_int("Number of bars to generate", 16, min_val=1, max_val=64)
        else:
            num_bars = None

        max_length = get_int(f"Max sequence length (model trained on: {self.bundle.max_seq_length})",
                              self.bundle.max_seq_length, 64, 8192)

        num_songs = get_int("Number of songs to generate", 1, min_val=1, max_val=100)

        exclude_drums = get_bool("Exclude drums from output", False)
        output_path = get_input("Output MIDI file", "output/generated.mid")

        # Confirm
        print(f"\n  Will generate with:")
        print(f"    Genre ID: {genre_id}")
        print(f"    Temperature: {self.settings['temperature']}")
        if num_bars:
            print(f"    Bars: {num_bars}")
        print(f"    Max length: {max_length}")
        print(f"    Songs: {num_songs}")
        if allowed_instruments:
            from ..data.vocabulary import GENERAL_MIDI_INSTRUMENTS
            names = [GENERAL_MIDI_INSTRUMENTS.get(p, "Drums") for p in allowed_instruments]
            print(f"    Instruments: {names}")

        confirm = get_bool("\nProceed with generation", True)
        if not confirm:
            return

        # Build output path template
        out_base = Path(output_path)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        use_numbered = num_songs > 1
        stem = out_base.stem
        suffix = out_base.suffix or ".mid"

        # Create generator once (reused across all songs)
        if is_multitrack:
            generator = self._create_poly_generator(max_length=max_length)

        for song_idx in range(num_songs):
            if use_numbered:
                out_path = out_base.parent / f"{stem}_{song_idx + 1:02d}{suffix}"
                print(f"\n[{song_idx + 1}/{num_songs}] Generating...")
            else:
                out_path = out_base
                print("\nGenerating...")

            try:
                if is_multitrack:
                    music = generator.generate_music(
                        genre_id=genre_id,
                        instruments=allowed_instruments,
                        num_bars=num_bars,
                        temperature=self.settings['temperature'],
                    )
                    if allowed_instruments:
                        music = self._remap_instruments(music, allowed_instruments)
                else:
                    tokens = self.bundle.generate(
                        genre_id=genre_id,
                        max_length=max_length,
                        temperature=self.settings['temperature'],
                        top_k=self.settings['top_k'],
                        top_p=self.settings['top_p'],
                    )
                    print(f"  Generated {len(tokens)} tokens")
                    music = self.bundle.encoder.decode_to_music(tokens)

                # Exclude drums if requested
                if exclude_drums:
                    music.tracks = [t for t in music.tracks if not t.is_drum]

                muspy.write_midi(str(out_path), music)

                total_notes = sum(len(t.notes) for t in music.tracks)
                print(f"  Saved to: {out_path}")
                print(f"  Tracks: {len(music.tracks)}, Total notes: {total_notes}")
                for i, track in enumerate(music.tracks):
                    name = "Drums" if track.is_drum else f"Program {track.program}"
                    print(f"    [{i}] {name}: {len(track.notes)} notes")

            except Exception as e:
                print(f"\nError during generation: {e}")
                import traceback
                traceback.print_exc()

        wait_for_enter()

    # =========================================================================
    # Settings Menu
    # =========================================================================

    def settings_menu(self):
        """Generation settings menu."""
        while True:
            print_header("Generation Settings")

            print(f"\n  Current settings:")
            print(f"    [1] Temperature: {self.settings['temperature']}")
            print(f"    [2] Top-k: {self.settings['top_k']}")
            print(f"    [3] Top-p: {self.settings['top_p']}")
            print(f"    [4] Min length: {self.settings['min_length']}")
            print("-" * 60)
            print("    [0] Back")
            print("=" * 60)

            choice = get_choice("Select setting to change", 4)

            if choice == 0:
                return
            elif choice == 1:
                self.settings['temperature'] = get_float("Temperature (0.1-2.0)",
                                                          self.settings['temperature'], 0.1, 2.0)
            elif choice == 2:
                self.settings['top_k'] = get_int("Top-k (0=disabled)",
                                                  self.settings['top_k'], 0, 500)
            elif choice == 3:
                self.settings['top_p'] = get_float("Top-p (0.0-1.0)",
                                                    self.settings['top_p'], 0.0, 1.0)
            elif choice == 4:
                self.settings['min_length'] = get_int("Min length before EOS",
                                                       self.settings['min_length'], 0, 4096)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    cli = GenerationCLI()
    cli.run()


if __name__ == "__main__":
    main()
