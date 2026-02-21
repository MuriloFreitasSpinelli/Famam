"""
Music Generator GUI

Graphical interface for music generation. Mirrors generation_cli.py exactly:
uses the same model loading, instrument selection, generator creation, and
generation logic.

Author: Radu Cristea
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
from typing import Optional, List
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class MusicGeneratorGUI:
    """GUI application for music generation."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Music Generator")
        self.root.geometry("860x820")
        self.root.resizable(True, True)

        # State
        self.bundle = None
        self.is_multitrack = False
        self.is_generating = False
        self.selected_instruments = None   # None = no filtering; list[int] = prog IDs
        self._top_instruments: List[tuple] = []

        self._create_ui()
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    # UI Construction
    
    def _create_ui(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)

        ttk.Label(main, text="Music Generator", font=("Helvetica", 18, "bold")).grid(
            row=0, column=0, pady=(0, 12))

        # Model Loading
        mf = ttk.LabelFrame(main, text="1. Load Model", padding="10")
        mf.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        mf.grid_columnconfigure(1, weight=1)

        ttk.Label(mf, text="Model Bundle (.h5):").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.model_path_var = tk.StringVar(value="No model loaded")
        ttk.Label(mf, textvariable=self.model_path_var, foreground="gray").grid(
            row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(mf, text="Browse...", command=self._load_model).grid(row=0, column=2)

        self.model_info_var = tk.StringVar(value="")
        ttk.Label(mf, textvariable=self.model_info_var, foreground="blue",
                  font=("Helvetica", 9)).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # Generation Parameters
        pf = ttk.LabelFrame(main, text="2. Generation Parameters", padding="10")
        pf.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        pf.grid_columnconfigure(1, weight=1)

        r = 0

        # Genre
        ttk.Label(pf, text="Genre:").grid(row=r, column=0, sticky="w", pady=4)
        self.genre_var = tk.StringVar()
        self.genre_combo = ttk.Combobox(pf, textvariable=self.genre_var, state="disabled", width=38)
        self.genre_combo.grid(row=r, column=1, sticky="ew", padx=(10, 0), pady=4)
        r += 1

        # Instruments
        ttk.Label(pf, text="Instruments:").grid(row=r, column=0, sticky="w", pady=4)
        inst_row = ttk.Frame(pf)
        inst_row.grid(row=r, column=1, sticky="ew", padx=(10, 0), pady=4)
        inst_row.grid_columnconfigure(0, weight=1)
        self.inst_display_var = tk.StringVar(value="No filtering (model decides freely)")
        ttk.Label(inst_row, textvariable=self.inst_display_var,
                  foreground="gray", font=("Helvetica", 9)).grid(row=0, column=0, sticky="w")
        self.inst_select_btn = ttk.Button(inst_row, text="Select...",
                                          command=self._open_instrument_dialog, state="disabled")
        self.inst_select_btn.grid(row=0, column=1, padx=(12, 0))
        r += 1

        # Temperature
        ttk.Label(pf, text="Temperature:").grid(row=r, column=0, sticky="w", pady=4)
        tf = ttk.Frame(pf)
        tf.grid(row=r, column=1, sticky="ew", padx=(10, 0), pady=4)
        tf.grid_columnconfigure(0, weight=1)
        self.temperature_var = tk.DoubleVar(value=0.9)
        self.temp_slider = ttk.Scale(tf, from_=0.1, to=2.0, variable=self.temperature_var,
                                     orient="horizontal", command=self._on_temp_changed,
                                     state="disabled")
        self.temp_slider.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.temp_label = ttk.Label(tf, text="0.90", width=5)
        self.temp_label.grid(row=0, column=1)
        r += 1
        ttk.Label(pf, text="Lower = conservative  |  Higher = creative",
                  foreground="gray", font=("Helvetica", 8)).grid(
            row=r, column=1, sticky="w", padx=(10, 0), pady=(0, 4))
        r += 1

        # Top-k
        ttk.Label(pf, text="Top-k:").grid(row=r, column=0, sticky="w", pady=4)
        tk_row = ttk.Frame(pf)
        tk_row.grid(row=r, column=1, sticky="w", padx=(10, 0), pady=4)
        self.top_k_var = tk.IntVar(value=50)
        self.top_k_spin = ttk.Spinbox(tk_row, from_=0, to=500, textvariable=self.top_k_var,
                                      width=8, state="disabled")
        self.top_k_spin.grid(row=0, column=0)
        ttk.Label(tk_row, text="  (0 = disabled)", foreground="gray",
                  font=("Helvetica", 8)).grid(row=0, column=1)
        r += 1

        # Top-p
        ttk.Label(pf, text="Nucleus (top-p):").grid(row=r, column=0, sticky="w", pady=4)
        tp_row = ttk.Frame(pf)
        tp_row.grid(row=r, column=1, sticky="w", padx=(10, 0), pady=4)
        self.top_p_var = tk.DoubleVar(value=0.92)
        self.top_p_spin = ttk.Spinbox(tp_row, from_=0.0, to=1.0, increment=0.01,
                                      textvariable=self.top_p_var, width=8,
                                      format="%.2f", state="disabled")
        self.top_p_spin.grid(row=0, column=0)
        ttk.Label(tp_row, text="  (0.0 – 1.0)", foreground="gray",
                  font=("Helvetica", 8)).grid(row=0, column=1)
        r += 1

        # Bars to generate 
        ttk.Label(pf, text="Bars to Generate:").grid(row=r, column=0, sticky="w", pady=4)
        bars_row = ttk.Frame(pf)
        bars_row.grid(row=r, column=1, sticky="w", padx=(10, 0), pady=4)
        self.num_bars_var = tk.IntVar(value=16)
        self.num_bars_spin = ttk.Spinbox(bars_row, from_=1, to=64, textvariable=self.num_bars_var,
                                         width=8, state="disabled")
        self.num_bars_spin.grid(row=0, column=0)
        ttk.Label(bars_row, text="  (multitrack only)", foreground="gray",
                  font=("Helvetica", 8)).grid(row=0, column=1)
        r += 1

        # Max sequence length
        ttk.Label(pf, text="Max Sequence Length:").grid(row=r, column=0, sticky="w", pady=4)
        ml_row = ttk.Frame(pf)
        ml_row.grid(row=r, column=1, sticky="w", padx=(10, 0), pady=4)
        self.max_length_var = tk.IntVar(value=2048)
        self.max_length_spin = ttk.Spinbox(ml_row, from_=64, to=8192, increment=64,
                                           textvariable=self.max_length_var, width=8,
                                           state="disabled")
        self.max_length_spin.grid(row=0, column=0)
        self.max_length_hint = ttk.Label(ml_row, text="  (model trained on: –)",
                                         foreground="gray", font=("Helvetica", 8))
        self.max_length_hint.grid(row=0, column=1)
        r += 1

        # Number of songs
        ttk.Label(pf, text="Number of Songs:").grid(row=r, column=0, sticky="w", pady=4)
        self.num_songs_var = tk.IntVar(value=1)
        self.num_songs_spin = ttk.Spinbox(pf, from_=1, to=100, textvariable=self.num_songs_var,
                                          width=8, state="disabled")
        self.num_songs_spin.grid(row=r, column=1, sticky="w", padx=(10, 0), pady=4)
        r += 1

        # Exclude drums
        ttk.Label(pf, text="Exclude Drums:").grid(row=r, column=0, sticky="w", pady=4)
        self.exclude_drums_var = tk.BooleanVar(value=False)
        self.exclude_drums_check = ttk.Checkbutton(pf, variable=self.exclude_drums_var,
                                                    state="disabled")
        self.exclude_drums_check.grid(row=r, column=1, sticky="w", padx=(10, 0), pady=4)

        # Output
        of = ttk.LabelFrame(main, text="3. Output", padding="10")
        of.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        of.grid_columnconfigure(1, weight=1)

        ttk.Label(of, text="Output Directory:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.output_path_var = tk.StringVar(value="output")
        ttk.Entry(of, textvariable=self.output_path_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(of, text="Browse...", command=self._select_output_dir).grid(row=0, column=2)

        ttk.Label(of, text="Filename Prefix:").grid(row=1, column=0, sticky="w",
                                                     padx=(0, 8), pady=(6, 0))
        self.prefix_var = tk.StringVar(value="generated")
        ttk.Entry(of, textvariable=self.prefix_var).grid(
            row=1, column=1, sticky="ew", padx=(0, 8), pady=(6, 0))

        # Buttons
        bf = ttk.Frame(main)
        bf.grid(row=4, column=0, pady=8)
        self.generate_btn = ttk.Button(bf, text="Generate Music", command=self._start_generation,
                                       state="disabled", style="Accent.TButton")
        self.generate_btn.pack(side="left", padx=5)
        self.cancel_btn = ttk.Button(bf, text="Cancel", command=self._cancel_generation,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=5)

        lf = ttk.LabelFrame(main, text="Log", padding="10")
        lf.grid(row=5, column=0, sticky="nsew", pady=(0, 8))
        lf.grid_columnconfigure(0, weight=1)
        lf.grid_rowconfigure(1, weight=1)
        main.grid_rowconfigure(5, weight=1)

        self.progress_bar = ttk.Progressbar(lf, mode="indeterminate")
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.log_text = scrolledtext.ScrolledText(lf, height=10, state="disabled",
                                                  wrap="word", font=("Courier", 9))
        self.log_text.grid(row=1, column=0, sticky="nsew")

        # Status bar   
        self.status_var = tk.StringVar(value="Ready. Load a model to begin.")
        ttk.Label(main, textvariable=self.status_var, relief="sunken", anchor="w").grid(
            row=6, column=0, sticky="ew")

    # Model Loading

    def _load_model(self):
        filepath = filedialog.askopenfilename(
            title="Select Model Bundle (.h5)",
            filetypes=[("Model Bundle", "*.h5"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        self.status_var.set("Loading model…")
        self._log(f"Loading: {filepath}")
        self._set_controls("disabled")
        threading.Thread(target=self._load_model_worker, args=(filepath,), daemon=True).start()

    def _load_model_worker(self, filepath: str):
        try:
            from src.models.model_bundle import load_model_bundle
            from src.data.encoders.multitrack_encoder import MultiTrackEncoder
            from src.data.vocabulary import INSTRUMENT_NAME_TO_ID

            bundle = load_model_bundle(filepath)
            is_multitrack = isinstance(bundle.encoder, MultiTrackEncoder)

            top_instruments = []
            if bundle.vocabulary:
                stats = bundle.vocabulary.get_instrument_stats()
                sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
                for name, count in sorted_stats[:20]:
                    prog_id = INSTRUMENT_NAME_TO_ID.get(name, -1)
                    if prog_id >= 0:
                        top_instruments.append((prog_id, name, count))

            self.root.after(0, lambda: self._on_model_loaded(
                bundle, is_multitrack, top_instruments, filepath))

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            err = str(e)
            self.root.after(0, lambda: self._on_model_load_error(err, tb))

    def _on_model_loaded(self, bundle, is_multitrack: bool,
                         top_instruments: list, filepath: str):
        self.bundle = bundle
        self.is_multitrack = is_multitrack
        self._top_instruments = top_instruments
        self.selected_instruments = None

        genres = []
        if bundle.vocabulary:
            genres = [name for name, _ in
                      sorted(bundle.vocabulary.genre_to_id.items(), key=lambda x: x[1])]

        model_type = "Multi-track" if is_multitrack else "Single-track"
        info = (f"{bundle.model_type.upper()}  |  {model_type}  |  "
                f"Vocab: {bundle.vocab_size}  |  Max seq: {bundle.max_seq_length}")
        if bundle.vocabulary:
            info += (f"  |  Genres: {bundle.vocabulary.num_genres}"
                     f"  |  Instruments: {bundle.vocabulary.num_active_instruments}")
        self.model_info_var.set(info)
        self.model_path_var.set(Path(filepath).name)

        self.genre_combo['values'] = genres
        if genres:
            self.genre_combo.set(genres[0])

        self.max_length_var.set(bundle.max_seq_length)
        self.max_length_hint.config(text=f"  (trained on: {bundle.max_seq_length})")
        self.inst_display_var.set("No filtering (model decides freely)")

        self._set_controls("normal")

        self._log(f"Loaded: {bundle.model_name}  [{model_type}]")
        self._log(f"  Vocab size: {bundle.vocab_size}  |  Max seq: {bundle.max_seq_length}")
        if bundle.vocabulary:
            self._log(f"  Genres: {bundle.vocabulary.num_genres}"
                      f"  |  Active instruments: {bundle.vocabulary.num_active_instruments}")
        self.status_var.set("Model loaded. Ready to generate.")

    def _on_model_load_error(self, error: str, tb: str):
        self._log(f"Error loading model:\n{tb}")
        messagebox.showerror("Load Error", f"Failed to load model:\n{error}")
        self.status_var.set("Error loading model.")
        self._set_controls("disabled")

    def _set_controls(self, state: str):
        """Enable or disable all generation-related widgets."""
        for w in (self.temp_slider, self.top_k_spin, self.top_p_spin,
                  self.num_bars_spin, self.max_length_spin, self.num_songs_spin,
                  self.exclude_drums_check, self.generate_btn):
            try:
                w['state'] = state
            except Exception:
                pass

        self.genre_combo['state'] = "disabled" if state == "disabled" else "readonly"

        self.inst_select_btn['state'] = (
            "normal" if (state == "normal" and self.is_multitrack) else "disabled"
        )

    # Instrument Selection Dialog

    def _open_instrument_dialog(self):
        if not self._top_instruments:
            messagebox.showinfo("No Data",
                                "No instrument statistics found in this model's vocabulary.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Instruments")
        dialog.geometry("540x500")
        dialog.resizable(False, False)
        dialog.grab_set()

        ttk.Label(dialog, text="Most used instruments in training data:",
                  font=("Helvetica", 10, "bold")).pack(anchor="w", padx=12, pady=(12, 4))

        # Instrument table
        tree_frame = ttk.Frame(dialog)
        tree_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        cols = ("rank", "id", "instrument", "songs")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                             height=13, selectmode="none")
        tree.heading("rank", text="Rank")
        tree.heading("id", text="ID")
        tree.heading("instrument", text="Instrument")
        tree.heading("songs", text="Songs")
        tree.column("rank", width=55, anchor="center")
        tree.column("id", width=55, anchor="center")
        tree.column("instrument", width=230)
        tree.column("songs", width=80, anchor="center")

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        for rank, (prog_id, name, count) in enumerate(self._top_instruments, 1):
            tag = " [DRUMS]" if prog_id == 128 else ""
            tree.insert("", "end", values=(rank, prog_id, name + tag, count))

        custom_frame = ttk.Frame(dialog)
        custom_frame.pack(fill="x", padx=12, pady=(0, 4))
        ttk.Label(custom_frame,
                  text="Custom IDs (space-separated, 0-127, 128=drums):").pack(anchor="w")
        custom_entry = ttk.Entry(custom_frame)
        custom_entry.pack(fill="x", pady=(2, 0))

        result = [self.selected_instruments]  

        def pick(n):
            result[0] = [prog_id for prog_id, _, _ in self._top_instruments[:n]]
            dialog.destroy()

        def pick_none():
            result[0] = None
            dialog.destroy()

        def pick_custom():
            raw = custom_entry.get().strip()
            if not raw:
                pick_none()
                return
            try:
                ids = [int(x) for x in raw.split() if 0 <= int(x) <= 128]
                result[0] = ids if ids else None
            except ValueError:
                messagebox.showwarning("Invalid Input",
                                       "Enter space-separated integers 0-128.", parent=dialog)
                return
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="Top 5 (Recommended)",
                   command=lambda: pick(5)).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Top 10",
                   command=lambda: pick(10)).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Use Custom IDs",
                   command=pick_custom).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="No Filtering",
                   command=pick_none).pack(side="left", padx=4)

        dialog.wait_window()

        # Apply result
        self.selected_instruments = result[0]
        if self.selected_instruments is None:
            self.inst_display_var.set("No filtering (model decides freely)")
        else:
            from src.data.vocabulary import GENERAL_MIDI_INSTRUMENTS
            names = [GENERAL_MIDI_INSTRUMENTS.get(p, "Drums") for p in self.selected_instruments]
            self.inst_display_var.set(", ".join(names))

    # Output helpers

    def _select_output_dir(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.output_path_var.set(d)

    def _on_temp_changed(self, val):
        self.temp_label.config(text=f"{float(val):.2f}")

    # Generation

    def _start_generation(self):
        if not self.bundle:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        if not self.genre_var.get():
            messagebox.showwarning("No Genre", "Select a genre.")
            return

        self.is_generating = True
        self.generate_btn['state'] = "disabled"
        self.cancel_btn['state'] = "normal"
        self.progress_bar.start()

        threading.Thread(target=self._generation_worker, daemon=True).start()

    def _generation_worker(self):
        try:
            import muspy
            from src.generation.poly_generator import PolyGenerator, PolyGeneratorConfig
            from src.data.vocabulary import GENERAL_MIDI_INSTRUMENTS

            # Snapshot all parameters from the UI
            genre_name        = self.genre_var.get()
            genre_id          = self.bundle.vocabulary.genre_to_id[genre_name]
            temperature       = self.temperature_var.get()
            top_k             = self.top_k_var.get()
            top_p             = self.top_p_var.get()
            num_bars          = self.num_bars_var.get()
            max_length        = self.max_length_var.get()
            num_songs         = self.num_songs_var.get()
            exclude_drums     = self.exclude_drums_var.get()
            allowed_instruments = self.selected_instruments 

            output_dir = Path(self.output_path_var.get())
            prefix     = self.prefix_var.get() or "generated"
            output_dir.mkdir(parents=True, exist_ok=True)

            self._log(f"\n{'='*55}")
            self._log("Starting generation")
            self._log(f"  Genre:        {genre_name} (ID: {genre_id})")
            self._log(f"  Temperature:  {temperature:.2f}")
            self._log(f"  Top-k:        {top_k}   Top-p: {top_p:.2f}")
            self._log(f"  Max length:   {max_length}")
            if self.is_multitrack:
                self._log(f"  Bars:         {num_bars}")
            if allowed_instruments:
                names = [GENERAL_MIDI_INSTRUMENTS.get(p, "Drums")
                         for p in allowed_instruments]
                self._log(f"  Instruments:  {names}")
            else:
                self._log("  Instruments:  No filtering")
            self._log(f"  Songs:        {num_songs}")
            self._log(f"{'='*55}\n")

            if self.is_multitrack:
                config = PolyGeneratorConfig(
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    resolution=self.bundle.encoder.resolution,
                )
                generator = PolyGenerator(
                    model=self.bundle.model,
                    encoder=self.bundle.encoder,
                    config=config,
                )

            use_numbered = num_songs > 1
            generated_files = []

            for song_idx in range(num_songs):
                if not self.is_generating:
                    self._log("Cancelled by user.")
                    break

                out_path = (
                    output_dir / f"{prefix}_{song_idx + 1:02d}.mid"
                    if use_numbered
                    else output_dir / f"{prefix}.mid"
                )

                label = f"[{song_idx + 1}/{num_songs}]" if use_numbered else ""
                self._log(f"{label} Generating…")
                self.root.after(0, lambda i=song_idx: self.status_var.set(
                    f"Generating song {i + 1} of {num_songs}…"))

                if self.is_multitrack:
                    music = generator.generate_music(
                        genre_id=genre_id,
                        instruments=allowed_instruments,
                        num_bars=num_bars,
                        temperature=temperature,
                    )
                    if allowed_instruments:
                        music = self._remap_instruments(music, allowed_instruments)
                else:
                    tokens = self.bundle.generate(
                        genre_id=genre_id,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    self._log(f"  Generated {len(tokens)} tokens")
                    music = self.bundle.encoder.decode_to_music(tokens)

                if exclude_drums:
                    music.tracks = [t for t in music.tracks if not t.is_drum]

                muspy.write_midi(str(out_path), music)

                total_notes = sum(len(t.notes) for t in music.tracks)
                self._log(f"  Saved: {out_path.name}")
                self._log(f"  Tracks: {len(music.tracks)}  |  Notes: {total_notes}")
                for i, track in enumerate(music.tracks):
                    tname = "Drums" if track.is_drum else f"Program {track.program}"
                    self._log(f"    [{i}] {tname}: {len(track.notes)} notes")

                generated_files.append(str(out_path))

            if self.is_generating and generated_files:
                self._log(f"\n{'='*55}")
                self._log(f"Done!  {len(generated_files)} file(s) saved to: {output_dir}")
                self._log(f"{'='*55}\n")
                self.root.after(0, lambda: self.status_var.set(
                    f"Done! {len(generated_files)} song(s) saved."))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Generation Complete",
                    f"Generated {len(generated_files)} song(s).\n\nSaved to:\n{output_dir}"
                ))

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            err = str(e)
            self._log(f"\nError: {err}\n{tb}")
            self.root.after(0, lambda: self.status_var.set("Generation failed."))
            self.root.after(0, lambda: messagebox.showerror("Generation Error", err))

        finally:
            self.root.after(0, self._reset_after_generation)

    def _remap_instruments(self, music, allowed_programs: List[int]):
        """Remap tracks to nearest allowed program. Mirrors CLI._remap_instruments()."""
        import muspy
        from src.data.vocabulary import GENERAL_MIDI_INSTRUMENTS

        allowed_melodic = [p for p in allowed_programs if p != 128]
        allow_drums = 128 in allowed_programs
        merged = {}

        for track in music.tracks:
            if track.is_drum:
                if not allow_drums:
                    continue
                key = 128
            else:
                if not allowed_melodic:
                    continue
                prog = track.program
                key = (prog if prog in allowed_melodic
                       else min(allowed_melodic, key=lambda x: abs(x - prog)))

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

    def _cancel_generation(self):
        if messagebox.askyesno("Cancel", "Cancel generation?"):
            self.is_generating = False
            self._log("Cancelling…")
            self.status_var.set("Cancelling…")

    def _reset_after_generation(self):
        self.is_generating = False
        self.generate_btn['state'] = "normal"
        self.cancel_btn['state'] = "disabled"
        self.progress_bar.stop()

    # Logging

    def _log(self, message: str):
        def _append():
            self.log_text['state'] = "normal"
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text['state'] = "disabled"

        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.root.after(0, _append)

# Entry Point

def main():
    """Run the GUI."""
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass
    style.configure("Accent.TButton", font=("Helvetica", 11, "bold"), padding=10)
    MusicGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
