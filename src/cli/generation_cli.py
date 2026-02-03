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
            "Generate Extended (Concatenation)",
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
            self.generate_extended()
        elif choice == 7:
            self.settings_menu()

    # =========================================================================
    # Load Model
    # =========================================================================

    def load_model(self):
        """Load a model bundle."""
        from ..models import load_model_bundle

        print_header("Load Model")

        # Find model bundles
        bundles = list(Path(".").rglob("*.h5"))
        model_bundles = [b for b in bundles if "model_bundle" in b.name or b.parent.name in ("checkpoints", "models")]

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

    def generate_music(self):
        """Interactive music generation."""
        import muspy
        from ..data import MultiTrackEncoder

        print_header("Generate Music")

        if not self.bundle:
            print("\n  No model loaded. Use 'Load Model' first.")
            wait_for_enter()
            return

        # Show model capabilities
        print(f"\n  Model: {self.bundle.model_name}")
        print(f"  Max sequence length: {self.bundle.max_seq_length}")

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

        print(f"\n  Current settings:")
        print(f"    Temperature: {self.settings['temperature']}")
        print(f"    Top-k: {self.settings['top_k']}")
        print(f"    Top-p: {self.settings['top_p']}")

        modify_settings = get_bool("Modify settings", False)
        if modify_settings:
            self.settings['temperature'] = get_float("Temperature (0.1-2.0)", self.settings['temperature'], 0.1, 2.0)
            self.settings['top_k'] = get_int("Top-k (0=disabled)", self.settings['top_k'], 0, 500)
            self.settings['top_p'] = get_float("Top-p (0.0-1.0)", self.settings['top_p'], 0.0, 1.0)

        max_length = get_int(f"Max sequence length (max: {self.bundle.max_seq_length})",
                              self.bundle.max_seq_length, 64, self.bundle.max_seq_length)

        min_length = get_int("Min length before EOS (0=none)", self.settings['min_length'], 0, max_length)
        self.settings['min_length'] = min_length

        ignore_eos = get_bool("Ignore EOS (generate full length)", False)
        exclude_drums = get_bool("Exclude drums from output", False)

        output_path = get_input("Output MIDI file", "output/generated.mid")

        # Confirm
        print(f"\n  Will generate with:")
        print(f"    Genre ID: {genre_id}")
        print(f"    Temperature: {self.settings['temperature']}")
        print(f"    Max length: {max_length}")
        if min_length > 0:
            print(f"    Min length: {min_length}")
        if ignore_eos:
            print(f"    Ignoring EOS (full {max_length} tokens)")

        confirm = get_bool("\nProceed with generation", True)
        if not confirm:
            return

        # Generate
        print("\nGenerating...")
        try:
            tokens = self.bundle.generate(
                genre_id=genre_id,
                max_length=max_length,
                min_length=min_length if min_length > 0 else None,
                temperature=self.settings['temperature'],
                top_k=self.settings['top_k'],
                top_p=self.settings['top_p'],
                ignore_eos=ignore_eos,
            )

            print(f"  Generated {len(tokens)} tokens")

            # Decode to music
            if isinstance(self.bundle.encoder, MultiTrackEncoder):
                music = self.bundle.encoder.decode_to_music(tokens)
            else:
                music = self.bundle.encoder.decode_to_music(tokens)

            # Exclude drums if requested
            if exclude_drums:
                music.tracks = [t for t in music.tracks if not t.is_drum]

            # Save
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            muspy.write_midi(str(out_path), music)

            # Print results
            print(f"\n  Saved to: {output_path}")
            print(f"  Tracks: {len(music.tracks)}")
            for i, track in enumerate(music.tracks):
                name = "Drums" if track.is_drum else f"Program {track.program}"
                print(f"    [{i}] {name}: {len(track.notes)} notes")

        except Exception as e:
            print(f"\nError during generation: {e}")
            import traceback
            traceback.print_exc()

        wait_for_enter()

    # =========================================================================
    # Generate Extended (Concatenation)
    # =========================================================================

    def generate_extended(self):
        """Generate longer music by concatenating multiple generations."""
        import muspy
        from ..data import MultiTrackEncoder

        print_header("Generate Extended Music")
        print("  Generate longer pieces by concatenating multiple generations.")

        if not self.bundle:
            print("\n  No model loaded. Use 'Load Model' first.")
            wait_for_enter()
            return

        # Show model info
        print(f"\n  Model: {self.bundle.model_name}")
        print(f"  Max sequence length per generation: {self.bundle.max_seq_length}")

        # Show genres
        if self.bundle.vocabulary:
            vocab = self.bundle.vocabulary
            genres = list(vocab.genre_to_id.items())
            if genres:
                print(f"\n  Available Genres:")
                for name, gid in sorted(genres, key=lambda x: x[1])[:10]:
                    print(f"    [{gid}] {name}")

        # Get parameters
        print("\n--- Extended Generation Parameters ---")

        genre_id = get_int("Genre ID", 0, min_val=0)

        # Duration options
        print("\n  Specify desired length:")
        print("    [1] By number of segments to concatenate")
        print("    [2] By approximate duration in bars")

        length_method = get_int("Choice", 1, min_val=1, max_val=2)

        num_segments = 1
        if length_method == 1:
            num_segments = get_int("Number of segments to generate", 3, min_val=1, max_val=10)
        else:
            approx_bars = get_int("Approximate bars (each segment ~8-16 bars)", 32, min_val=8)
            bars_per_segment = 12  # Rough estimate
            num_segments = max(1, approx_bars // bars_per_segment)
            print(f"  Will generate ~{num_segments} segments")

        segment_length = get_int(f"Tokens per segment (max: {self.bundle.max_seq_length})",
                                  self.bundle.max_seq_length, 256, self.bundle.max_seq_length)

        print(f"\n  Generation settings:")
        print(f"    Temperature: {self.settings['temperature']}")
        print(f"    Top-k: {self.settings['top_k']}")
        print(f"    Top-p: {self.settings['top_p']}")

        modify = get_bool("Modify settings", False)
        if modify:
            self.settings['temperature'] = get_float("Temperature", self.settings['temperature'], 0.1, 2.0)
            self.settings['top_k'] = get_int("Top-k", self.settings['top_k'], 0, 500)
            self.settings['top_p'] = get_float("Top-p", self.settings['top_p'], 0.0, 1.0)

        exclude_drums = get_bool("Exclude drums from output", False)
        output_path = get_input("Output MIDI file", "output/extended.mid")

        print(f"\n  Will generate {num_segments} segments and concatenate.")

        confirm = get_bool("\nProceed", True)
        if not confirm:
            return

        # Generate segments
        print("\nGenerating segments...")
        all_tracks: Dict[int, List[muspy.Note]] = {}  # program -> notes

        total_time_offset = 0

        for seg_idx in range(num_segments):
            print(f"  Segment {seg_idx + 1}/{num_segments}...")

            try:
                tokens = self.bundle.generate(
                    genre_id=genre_id,
                    max_length=segment_length,
                    min_length=segment_length // 2,
                    temperature=self.settings['temperature'],
                    top_k=self.settings['top_k'],
                    top_p=self.settings['top_p'],
                    ignore_eos=True,  # Always generate full segment for concatenation
                )

                # Decode
                if isinstance(self.bundle.encoder, MultiTrackEncoder):
                    music = self.bundle.encoder.decode_to_music(tokens)
                else:
                    music = self.bundle.encoder.decode_to_music(tokens)

                # Get segment duration
                segment_duration = 0
                for track in music.tracks:
                    if track.notes:
                        track_end = max(n.time + n.duration for n in track.notes)
                        segment_duration = max(segment_duration, track_end)

                # Add notes to combined tracks with time offset
                for track in music.tracks:
                    if exclude_drums and track.is_drum:
                        continue

                    program = 128 if track.is_drum else track.program

                    if program not in all_tracks:
                        all_tracks[program] = []

                    for note in track.notes:
                        offset_note = muspy.Note(
                            time=note.time + total_time_offset,
                            pitch=note.pitch,
                            duration=note.duration,
                            velocity=note.velocity,
                        )
                        all_tracks[program].append(offset_note)

                total_time_offset += segment_duration
                print(f"    Generated {len(tokens)} tokens, duration {segment_duration} ticks")

            except Exception as e:
                print(f"    Error in segment {seg_idx + 1}: {e}")
                continue

        # Build final music object
        print("\nBuilding combined music...")

        resolution = getattr(self.bundle.encoder, 'resolution', 24)
        final_music = muspy.Music(
            resolution=resolution,
            tempos=[muspy.Tempo(time=0, qpm=120)],
            tracks=[],
        )

        for program, notes in all_tracks.items():
            is_drum = (program == 128)
            track = muspy.Track(
                program=0 if is_drum else program,
                is_drum=is_drum,
                name="Drums" if is_drum else f"Program {program}",
                notes=sorted(notes, key=lambda n: n.time),
            )
            final_music.tracks.append(track)

        # Save
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        muspy.write_midi(str(out_path), final_music)

        # Print results
        total_notes = sum(len(t.notes) for t in final_music.tracks)
        ticks_per_bar = resolution * 4
        total_bars = total_time_offset / ticks_per_bar if ticks_per_bar > 0 else 0

        print(f"\n  Extended generation complete!")
        print(f"  Saved to: {output_path}")
        print(f"  Total segments: {num_segments}")
        print(f"  Total duration: ~{total_bars:.1f} bars")
        print(f"  Total tracks: {len(final_music.tracks)}")
        print(f"  Total notes: {total_notes}")
        print(f"\n  Tracks:")
        for i, track in enumerate(final_music.tracks):
            name = "Drums" if track.is_drum else f"Program {track.program}"
            print(f"    [{i}] {name}: {len(track.notes)} notes")

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
