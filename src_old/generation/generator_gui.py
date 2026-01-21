"""
Simple GUI for Music Generation using trained Famam models.

Usage:
    python src/generation/generator_gui.py [--model-path path/to/model_bundle.h5]
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path
import sys
import os
import traceback

# Add project root and src to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Suppress TF warnings before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import muspy


class MusicGeneratorGUI:
    """Simple GUI for generating music with trained models."""

    def __init__(self, root, model_path=None):
        self.root = root
        self.root.title("Famam Music Generator")
        self.root.geometry("550x500")
        self.root.resizable(True, True)

        self.saved_model = None
        self.model_path = model_path
        self.output_dir = str(project_root / "output")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Create UI
        self._create_widgets()

        # Auto-load model if path provided
        if model_path:
            self.root.after(100, lambda: self._load_model(model_path))

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # === Model Selection ===
        ttk.Label(main_frame, text="Model:", font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(0, 5)
        )
        row += 1

        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        model_frame.columnconfigure(0, weight=1)

        self.model_var = tk.StringVar(value="No model loaded")
        self.model_label = ttk.Label(model_frame, textvariable=self.model_var, width=40)
        self.model_label.grid(row=0, column=0, sticky="ew")

        self.load_btn = ttk.Button(model_frame, text="Browse...", command=self._browse_model)
        self.load_btn.grid(row=0, column=1, padx=(5, 0))

        row += 1

        # === Output Directory ===
        ttk.Label(main_frame, text="Output Folder:", font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(0, 5)
        )
        row += 1

        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)

        self.output_var = tk.StringVar(value=self.output_dir)
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, sticky="ew")

        self.output_btn = ttk.Button(output_frame, text="Browse...", command=self._browse_output)
        self.output_btn.grid(row=0, column=1, padx=(5, 0))

        row += 1

        # === Genre Selection ===
        ttk.Label(main_frame, text="Genre:", font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(0, 5)
        )
        row += 1

        self.genre_var = tk.StringVar()
        self.genre_combo = ttk.Combobox(
            main_frame, textvariable=self.genre_var, state="readonly", width=30
        )
        self.genre_combo.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.genre_combo["values"] = ["Load a model first..."]
        self.genre_combo.current(0)
        row += 1

        # === Generation Settings ===
        ttk.Label(main_frame, text="Settings:", font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(0, 5)
        )
        row += 1

        settings_frame = ttk.Frame(main_frame)
        settings_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Filename
        ttk.Label(settings_frame, text="Filename:").grid(row=0, column=0, sticky="w")
        self.filename_var = tk.StringVar(value="generated")
        self.filename_entry = ttk.Entry(settings_frame, textvariable=self.filename_var, width=20)
        self.filename_entry.grid(row=0, column=1, padx=(10, 5))
        ttk.Label(settings_frame, text=".mid").grid(row=0, column=2, sticky="w")

        row += 1

        settings_frame2 = ttk.Frame(main_frame)
        settings_frame2.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Temperature (randomness)
        ttk.Label(settings_frame2, text="Temperature:").grid(row=0, column=0, sticky="w")
        self.temp_var = tk.DoubleVar(value=1.0)
        self.temp_spin = ttk.Spinbox(
            settings_frame2, from_=0.1, to=2.0, increment=0.1,
            textvariable=self.temp_var, width=8
        )
        self.temp_spin.grid(row=0, column=1, padx=(10, 20))

        # Threshold
        ttk.Label(settings_frame2, text="Threshold:").grid(row=0, column=2, sticky="w")
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_spin = ttk.Spinbox(
            settings_frame2, from_=0.1, to=0.9, increment=0.1,
            textvariable=self.threshold_var, width=8
        )
        self.threshold_spin.grid(row=0, column=3, padx=(10, 0))

        row += 1

        # Instrument selection
        ttk.Label(settings_frame2, text="Instrument:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.instrument_var = tk.StringVar(value="Piano")
        self.instrument_combo = ttk.Combobox(
            settings_frame2, textvariable=self.instrument_var, state="readonly", width=20
        )
        self.instrument_combo["values"] = [
            "Piano", "Electric Piano", "Organ", "Guitar (Acoustic)",
            "Guitar (Electric Clean)", "Guitar (Electric Distorted)",
            "Bass", "Strings", "Synth Lead", "Synth Pad"
        ]
        self.instrument_combo.current(0)
        self.instrument_combo.grid(row=1, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=(5, 0))

        row += 1

        # Seed option
        seed_frame = ttk.LabelFrame(main_frame, text="Seed Type", padding="5")
        seed_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.seed_type_var = tk.StringVar(value="noise")
        ttk.Radiobutton(
            seed_frame, text="Noise (recommended)", variable=self.seed_type_var, value="noise"
        ).grid(row=0, column=0, padx=(0, 15))
        ttk.Radiobutton(
            seed_frame, text="Random", variable=self.seed_type_var, value="random"
        ).grid(row=0, column=1, padx=(0, 15))
        ttk.Radiobutton(
            seed_frame, text="Zero", variable=self.seed_type_var, value="zero"
        ).grid(row=0, column=2)

        row += 1

        # === Generate Button ===
        self.generate_btn = ttk.Button(
            main_frame, text="Generate Music", command=self._generate,
            state="disabled"
        )
        self.generate_btn.grid(row=row, column=0, columnspan=2, pady=(10, 10), sticky="ew")
        row += 1

        # === Progress ===
        self.progress_var = tk.StringVar(value="Ready - Load a model to start")
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        # === Output Info ===
        self.output_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        self.output_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        main_frame.rowconfigure(row, weight=1)

        self.output_text = tk.Text(self.output_frame, height=8, width=50, state="disabled")
        scrollbar = ttk.Scrollbar(self.output_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _browse_model(self):
        """Open file dialog to select model."""
        filepath = filedialog.askopenfilename(
            title="Select Model Bundle",
            filetypes=[("H5 Files", "*.h5"), ("All Files", "*.*")],
            initialdir=str(project_root / "models")
        )
        if filepath:
            self._load_model(filepath)

    def _browse_output(self):
        """Open folder dialog to select output directory."""
        folder = filedialog.askdirectory(
            title="Select Output Folder",
            initialdir=self.output_dir
        )
        if folder:
            self.output_dir = folder
            self.output_var.set(folder)

    def _load_model(self, filepath):
        """Load the saved model."""
        self.progress_var.set("Loading model...")
        self.root.update()

        try:
            self._log(f"Loading model: {Path(filepath).name}")

            from training.saved_model import SavedModel
            self.saved_model = SavedModel.load(filepath)
            self.model_path = filepath

            # Update UI
            self.model_var.set(Path(filepath).name)

            # Populate genres
            genres = self.saved_model.list_genres()
            if genres:
                self.genre_combo["values"] = genres
                self.genre_combo.current(0)
                self._log(f"  Available genres: {len(genres)}")
            else:
                self.genre_combo["values"] = ["No genres available"]
                self.genre_combo.current(0)

            # Enable generate button
            self.generate_btn["state"] = "normal"

            self._log(f"  Music shape: {self.saved_model.dataset_info.music_shape}")
            self._log("Model loaded successfully!")

            self.progress_var.set("Ready - Select a genre and click Generate")

        except Exception as e:
            error_msg = f"Failed to load model:\n{str(e)}"
            self._log(f"ERROR: {error_msg}")
            self._log(traceback.format_exc())
            messagebox.showerror("Error", error_msg)
            self.progress_var.set("Failed to load model")

    def _generate(self):
        """Generate music."""
        if self.saved_model is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        # Disable button during generation
        self.generate_btn["state"] = "disabled"
        self.progress_var.set("Generating...")
        self.root.update()

        try:
            genre = self.genre_var.get()
            temperature = self.temp_var.get()
            threshold = self.threshold_var.get()
            seed_type = self.seed_type_var.get()
            instrument = self.instrument_var.get()
            filename = self.filename_var.get() or "generated"

            # Map instrument names to MIDI program numbers
            instrument_programs = {
                "Piano": 0,
                "Electric Piano": 4,
                "Organ": 19,
                "Guitar (Acoustic)": 25,
                "Guitar (Electric Clean)": 27,
                "Guitar (Electric Distorted)": 29,
                "Bass": 33,
                "Strings": 48,
                "Synth Lead": 80,
                "Synth Pad": 88,
            }
            midi_program = instrument_programs.get(instrument, 0)

            self._log(f"\n--- Generating Music ---")
            self._log(f"  Genre: {genre}")
            self._log(f"  Instrument: {instrument} (program {midi_program})")
            self._log(f"  Temperature: {temperature}")
            self._log(f"  Threshold: {threshold}")
            self._log(f"  Seed type: {seed_type}")

            # Get model input shape
            music_shape = self.saved_model.dataset_info.music_shape
            pitches, time_steps = music_shape
            self._log(f"  Input shape: {music_shape}")

            # Create seed input based on selection
            self.progress_var.set("Creating seed...")
            self.root.update()

            if seed_type == "random":
                np.random.seed(None)
                seed = np.random.rand(pitches, time_steps).astype(np.float32)
            elif seed_type == "zero":
                seed = np.zeros((pitches, time_steps), dtype=np.float32)
            else:  # noise
                seed = np.random.randn(pitches, time_steps).astype(np.float32) * 0.1

            # Generate
            self.progress_var.set("Running model prediction...")
            self.root.update()

            self._log("  Running prediction...")
            output = self.saved_model.predict(seed, genre=genre)
            self._log(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

            # Apply temperature
            if temperature != 1.0:
                output = output * temperature

            # Threshold to binary piano roll (must be bool for muspy)
            output_binary = (output > threshold)
            active_notes = np.sum(output_binary)
            self._log(f"  Active values after threshold: {active_notes}")

            # Convert to muspy Music object
            self.progress_var.set("Converting to MIDI...")
            self.root.update()

            # Piano roll format: (time_steps, pitches) for muspy
            piano_roll = output_binary.T

            # Create muspy Music from piano roll
            music = muspy.from_pianoroll_representation(
                piano_roll,
                resolution=24,
                encode_velocity=False
            )

            # Set instrument program on the track
            if music.tracks:
                music.tracks[0].program = midi_program
                music.tracks[0].name = instrument

            # Set tempo
            music.tempos = [muspy.Tempo(time=0, qpm=120)]

            # Count notes
            total_notes = sum(len(t.notes) for t in music.tracks)
            self._log(f"  Generated notes: {total_notes}")

            if total_notes == 0:
                self._log("  WARNING: No notes generated! Try lower threshold or different seed.")
                messagebox.showwarning("Warning", "No notes were generated.\nTry lowering the threshold or using a different seed type.")
            else:
                # Save to file
                self.progress_var.set("Saving MIDI file...")
                self.root.update()

                # Create filename with genre
                safe_genre = "".join(c if c.isalnum() else "_" for c in genre)
                output_path = Path(self.output_dir) / f"{filename}_{safe_genre}.mid"

                # Ensure directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save
                music.write_midi(str(output_path))
                self._log(f"  Saved to: {output_path}")

                self.progress_var.set(f"Done! Saved to {output_path.name}")
                messagebox.showinfo("Success", f"Music generated!\n\nSaved to:\n{output_path}")

        except Exception as e:
            error_msg = f"Generation failed:\n{str(e)}"
            self._log(f"ERROR: {error_msg}")
            self._log(traceback.format_exc())
            messagebox.showerror("Error", error_msg)
            self.progress_var.set("Generation failed")

        finally:
            self.generate_btn["state"] = "normal"

    def _log(self, message):
        """Add message to output text."""
        self.output_text["state"] = "normal"
        self.output_text.insert("end", message + "\n")
        self.output_text.see("end")
        self.output_text["state"] = "disabled"
        self.root.update()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Famam Music Generator GUI")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model_bundle.h5 file (optional, can browse in GUI)"
    )
    args = parser.parse_args()

    # Check for default model if not specified
    default_model = project_root / "models" / "test_gui" / "model_bundle.h5"
    model_path = args.model_path
    if model_path is None and default_model.exists():
        model_path = str(default_model)

    # Create and run GUI
    root = tk.Tk()
    app = MusicGeneratorGUI(root, model_path=model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
