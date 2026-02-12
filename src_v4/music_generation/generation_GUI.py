"""
Music Generator GUI

A graphical user interface for generating music using trained models.
Works with MusicGenerator and MultiTrackGenerator classes.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
from typing import Optional, List
import sys

# Add project root to path for imports (ensure `src` package is importable)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.model_bundle import load_model_bundle
from src.generation import (
    MonoGenerator, MonoGeneratorConfig,
    PolyGenerator, PolyGeneratorConfig,
    # Backwards compatibility names
    MusicGenerator, GenerationConfig,
    MultiTrackGenerator, MultiTrackConfig
)


class MusicGeneratorGUI:
    """GUI application for music generation."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Music Generator")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        
        # Generator state
        self.bundle = None
        self.generator = None
        self.vocabulary = None
        self.is_multitrack = False
        self.is_generating = False
        
        # Available genres (will be populated from bundle vocabulary)
        self.genres = []
        # Available instruments for current genre (populated dynamically)
        self.available_instruments = {}
        self.available_instruments_ids = {}
        
        # Create UI
        self._create_ui()
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def _create_ui(self):
        """Create all UI elements."""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Music Generator", 
            font=("Helvetica", 20, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # === Model Loading Section ===
        model_frame = ttk.LabelFrame(main_frame, text="1. Load Model Bundle", padding="10")
        model_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        model_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model Bundle:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.model_path_var = tk.StringVar(value="No model loaded")
        ttk.Label(
            model_frame, 
            textvariable=self.model_path_var, 
            foreground="gray"
        ).grid(row=0, column=1, sticky="ew", padx=(0, 10))
        
        ttk.Button(
            model_frame, 
            text="Browse...", 
            command=self._load_model
        ).grid(row=0, column=2)
        
        # Model info
        self.model_info_var = tk.StringVar(value="")
        ttk.Label(
            model_frame, 
            textvariable=self.model_info_var, 
            foreground="blue",
            font=("Helvetica", 9)
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))
        
        # === Generation Settings Section ===
        settings_frame = ttk.LabelFrame(main_frame, text="2. Generation Settings", padding="10")
        settings_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        settings_frame.grid_columnconfigure(1, weight=1)
        
        # Genre selection
        ttk.Label(settings_frame, text="Genre:").grid(row=0, column=0, sticky="w", pady=5)
        
        self.genre_var = tk.StringVar()
        self.genre_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.genre_var,
            state="disabled",
            width=30
        )
        self.genre_combo.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=5)
        self.genre_combo.bind("<<ComboboxSelected>>", lambda e: self._on_genre_selected())
        
        # Instruments (for multitrack)
        ttk.Label(settings_frame, text="Instruments:").grid(row=1, column=0, sticky="w", pady=5)
        
        self.inst_container = ttk.Frame(settings_frame)
        self.inst_container.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)
        self.inst_container.grid_columnconfigure(0, weight=1)
        
        self.instruments_text = tk.Text(
            self.inst_container,
            height=3,
            state="disabled",
            wrap="word",
            font=("Helvetica", 9)
        )
        self.instruments_text.grid(row=0, column=0, sticky="ew")
        
        self.inst_select_button = ttk.Button(
            self.inst_container,
            text="Select...",
            command=self._select_instruments,
            state="disabled"
        )
        self.inst_select_button.grid(row=0, column=1, padx=(5, 0))
        
        self.selected_instruments = []
        
        # Temperature slider
        ttk.Label(settings_frame, text="Temperature:").grid(row=2, column=0, sticky="w", pady=5)
        
        temp_container = ttk.Frame(settings_frame)
        temp_container.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=5)
        temp_container.grid_columnconfigure(0, weight=1)
        
        self.temperature_var = tk.DoubleVar(value=0.9)
        self.temperature_slider = ttk.Scale(
            temp_container,
            from_=0.5,
            to=1.5,
            variable=self.temperature_var,
            orient="horizontal",
            command=self._update_temperature_label,
            state="disabled"
        )
        self.temperature_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        self.temperature_label = ttk.Label(temp_container, text="0.9")
        self.temperature_label.grid(row=0, column=1)
        
        # Temperature explanation
        temp_info = ttk.Label(
            settings_frame,
            text="Lower (0.5-0.8) = Conservative | Higher (1.0-1.5) = Creative",
            font=("Helvetica", 8),
            foreground="gray"
        )
        temp_info.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        # Number of bars
        ttk.Label(settings_frame, text="Number of Bars:").grid(row=4, column=0, sticky="w", pady=5)
        
        bars_container = ttk.Frame(settings_frame)
        bars_container.grid(row=4, column=1, sticky="ew", padx=(10, 0), pady=5)
        
        self.num_bars_var = tk.IntVar(value=8)
        self.num_bars_spinbox = ttk.Spinbox(
            bars_container,
            from_=4,
            to=32,
            textvariable=self.num_bars_var,
            width=10,
            state="disabled"
        )
        self.num_bars_spinbox.grid(row=0, column=0, sticky="w")
        
        ttk.Label(
            bars_container,
            text="(4 bars = ~8 seconds at 120 BPM)",
            font=("Helvetica", 8),
            foreground="gray"
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        # Number of variations
        ttk.Label(settings_frame, text="Variations:").grid(row=5, column=0, sticky="w", pady=5)
        
        variations_container = ttk.Frame(settings_frame)
        variations_container.grid(row=5, column=1, sticky="ew", padx=(10, 0), pady=5)
        
        self.num_variations_var = tk.IntVar(value=1)
        self.num_variations_spinbox = ttk.Spinbox(
            variations_container,
            from_=1,
            to=10,
            textvariable=self.num_variations_var,
            width=10,
            state="disabled"
        )
        self.num_variations_spinbox.grid(row=0, column=0, sticky="w")
        
        ttk.Label(
            variations_container,
            text="(Generate multiple songs with same settings)",
            font=("Helvetica", 8),
            foreground="gray"
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        # === Output Settings Section ===
        output_frame = ttk.LabelFrame(main_frame, text="3. Output Settings", padding="10")
        output_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        output_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.output_path_var = tk.StringVar(value="./generated_music")
        ttk.Entry(
            output_frame,
            textvariable=self.output_path_var,
            state="readonly"
        ).grid(row=0, column=1, sticky="ew", padx=(0, 10))
        
        ttk.Button(
            output_frame,
            text="Browse...",
            command=self._select_output_dir
        ).grid(row=0, column=2)
        
        # Filename prefix
        ttk.Label(output_frame, text="Filename Prefix:").grid(row=1, column=0, sticky="w", pady=(5, 0), padx=(0, 10))
        
        self.prefix_var = tk.StringVar(value="generated")
        self.prefix_entry = ttk.Entry(
            output_frame,
            textvariable=self.prefix_var,
            state="disabled"
        )
        self.prefix_entry.grid(row=1, column=1, sticky="ew", pady=(5, 0), padx=(0, 10))
        
        # Tempo
        ttk.Label(output_frame, text="Tempo (BPM):").grid(row=2, column=0, sticky="w", pady=(5, 0), padx=(0, 10))
        
        self.tempo_var = tk.DoubleVar(value=120.0)
        ttk.Spinbox(
            output_frame,
            from_=60,
            to=180,
            textvariable=self.tempo_var,
            width=10,
            state="disabled"
        ).grid(row=2, column=1, sticky="w", pady=(5, 0), padx=(0, 10))
        
        # === Action Buttons ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, pady=(0, 10))
        
        self.generate_button = ttk.Button(
            button_frame,
            text="🎹 Generate Music",
            command=self._generate_music,
            state="disabled",
            style="Accent.TButton"
        )
        self.generate_button.pack(side="left", padx=5)
        
        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel_generation,
            state="disabled"
        )
        self.cancel_button.pack(side="left", padx=5)
        
        # === Progress Section ===
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=5, column=0, sticky="nsew", pady=(0, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        progress_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_rowconfigure(5, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="indeterminate",
            variable=self.progress_var
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Log output
        self.log_text = scrolledtext.ScrolledText(
            progress_frame,
            height=10,
            state="disabled",
            wrap="word",
            font=("Courier", 9)
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load a model to begin.")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w"
        )
        status_bar.grid(row=6, column=0, sticky="ew")
    
    def _load_model(self):
        """Load a model bundle file."""
        filepath = filedialog.askopenfilename(
            title="Select Model Bundle",
            filetypes=[
                ("Model Bundle", "*.h5"),
                ("All Files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            self._log("Loading model bundle...")
            self.status_var.set("Loading model...")
            
            # Load bundle
            self.bundle = load_model_bundle(filepath)
            
            # Detect if multitrack encoder
            encoder = self.bundle.encoder
            self.is_multitrack = 'multitrack' in encoder.__class__.__name__.lower()
            
            # Create generator
            if self.is_multitrack:
                config = PolyGeneratorConfig()
                self.generator = PolyGenerator.from_bundle(self.bundle, config)
                model_type = "MultiTrack (Polyphonic)"
            else:
                config = MonoGeneratorConfig()
                self.generator = MonoGenerator.from_bundle(self.bundle, config)
                model_type = "Single Track (Monophonic)"
            
            # Get available genres from vocabulary
            if hasattr(self.bundle, 'vocabulary'):
                self.vocabulary = self.bundle.vocabulary
                if hasattr(self.vocabulary, 'genre_to_id'):
                    self.genres = list(self.vocabulary.genre_to_id.keys())
            
            # Update UI
            self.model_path_var.set(Path(filepath).name)
            
            self.genre_combo['values'] = self.genres
            if self.genres:
                self.genre_combo.set(self.genres[0])
            
            self.model_info_var.set(f"✓ Model loaded: {model_type} | {len(self.genres)} genres available")
            
            # Enable controls
            self.genre_combo['state'] = "readonly"
            self.temperature_slider['state'] = "normal"
            self.num_bars_spinbox['state'] = "normal"
            self.num_variations_spinbox['state'] = "normal"
            # enable prefix entry for editing
            self.prefix_entry['state'] = "normal"
            self.generate_button['state'] = "normal"
            
            # Auto-select first genre and update instruments
            if self.genres:
                self._on_genre_selected()
            
            self._log(f"Model loaded successfully!")
            self._log(f"  Type: {model_type}")
            self._log(f"  Genres: {', '.join(self.genres[:5])}{'...' if len(self.genres) > 5 else ''}")
            if self.is_multitrack:
                self._log(f"  Instruments: Model will use genre-specific instruments")
            
            self.status_var.set("Model loaded. Ready to generate.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self._log(f"Error: {str(e)}")
            self.status_var.set("Error loading model.")
    
    def _on_genre_selected(self):
        """Called when genre selection changes. Updates available instruments."""
        if not self.vocabulary or not self.is_multitrack:
            return
        
        genre = self.genre_var.get()
        if not genre:
            return
        
        # Get instruments that were actually used for this genre during training
        instrument_ids = self.vocabulary.get_instruments_for_genre(genre)
        
        if not instrument_ids:
            # Fallback: try top instruments if no specific ones found
            instrument_ids = self.vocabulary.get_top_instruments_for_genre(genre, top_n=5)
        
        # Build mapping from instrument ID to name
        self.available_instruments = {}
        self.available_instruments_ids = {}
        for inst_id in sorted(instrument_ids):
            inst_name = self.vocabulary.get_instrument_name(inst_id)
            self.available_instruments[inst_id] = inst_name
            self.available_instruments_ids[inst_name] = inst_id
        
        # Enable instrument selection if we have instruments for this genre
        if self.available_instruments:
            self.inst_select_button['state'] = "normal"
        else:
            self.inst_select_button['state'] = "disabled"
            self.selected_instruments = []
            self._update_instruments_display()
    
    def _select_instruments(self):
        """Open dialog to select instruments for multitrack generation."""
        if not self.available_instruments:
            messagebox.showwarning("No Instruments", f"No instruments available for genre '{self.genre_var.get()}'")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Instruments")
        dialog.geometry("400x500")
        
        genre = self.genre_var.get()
        ttk.Label(
            dialog,
            text=f"Select instruments for '{genre}':\n(showing trained instruments)",
            font=("Helvetica", 11, "bold")
        ).pack(pady=10)
        
        # Create listbox with checkboxes
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        listbox = tk.Listbox(
            list_frame,
            selectmode="multiple",
            yscrollcommand=scrollbar.set
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Show only instruments available for this genre
        inst_names = list(self.available_instruments.values())
        for inst in inst_names:
            listbox.insert("end", inst)
        
        # Select previously selected instruments
        for i, inst in enumerate(inst_names):
            if inst in self.selected_instruments:
                listbox.selection_set(i)
        
        def on_ok():
            selected_indices = listbox.curselection()
            inst_names = list(self.available_instruments.values())
            self.selected_instruments = [inst_names[i] for i in selected_indices]
            self._update_instruments_display()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="left", padx=5)
    
    def _update_instruments_display(self):
        """Update the instruments text display."""
        self.instruments_text['state'] = "normal"
        self.instruments_text.delete("1.0", "end")
        if self.selected_instruments:
            text = ", ".join(self.selected_instruments)
        else:
            text = "None selected (will use defaults)"
        self.instruments_text.insert("1.0", text)
        self.instruments_text['state'] = "disabled"
    
    def _select_output_dir(self):
        """Select output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_path_var.set(directory)
    
    def _update_temperature_label(self, value):
        """Update temperature label."""
        temp = float(value)
        self.temperature_label.config(text=f"{temp:.2f}")
    
    def _generate_music(self):
        """Start music generation."""
        if not self.generator:
            messagebox.showwarning("No Model", "Please load a model first.")
            return
        
        if not self.genre_var.get():
            messagebox.showwarning("No Genre", "Please select a genre.")
            return
        
        # Disable UI
        self.is_generating = True
        self.generate_button['state'] = "disabled"
        self.cancel_button['state'] = "normal"
        self.progress_bar.start()
        
        # Start generation thread
        thread = threading.Thread(target=self._generate_worker, daemon=True)
        thread.start()
    
    def _generate_worker(self):
        """Worker thread for generation."""
        try:
            genre = self.genre_var.get()
            temperature = self.temperature_var.get()
            num_bars = self.num_bars_var.get()
            num_variations = self.num_variations_var.get()
            output_dir = Path(self.output_path_var.get())
            prefix = self.prefix_var.get()
            tempo = self.tempo_var.get()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get genre ID
            genre_id = self.vocabulary.get_genre_id(genre)
            
            # Get instrument IDs for multitrack
            instrument_ids = None
            if self.is_multitrack and self.selected_instruments:
                instrument_ids = [
                    self.vocabulary.get_instrument_id(inst)
                    for inst in self.selected_instruments
                ]
            
            self._log(f"\n{'='*60}")
            self._log(f"Starting generation...")
            self._log(f"  Genre: {genre} (ID: {genre_id})")
            self._log(f"  Temperature: {temperature:.2f}")
            self._log(f"  Bars: {num_bars}")
            self._log(f"  Variations: {num_variations}")
            self._log(f"  Tempo: {tempo} BPM")
            if self.is_multitrack and instrument_ids:
                self._log(f"  Instruments: {self.selected_instruments}")
            self._log(f"{'='*60}\n")
            
            self.status_var.set(f"Generating {num_variations} song(s)...")
            
            generated_files = []
            
            for i in range(num_variations):
                if not self.is_generating:
                    self._log("\nCancelled by user")
                    break
                
                self._log(f"Generating variation {i+1}/{num_variations}...")
                
                filename = f"{prefix}_{genre}_temp{temperature:.2f}_{i+1:03d}.mid"
                output_path = output_dir / filename
                
                if self.is_multitrack:
                    # MultiTrack generation (PolyGenerator)
                    self.generator.generate_midi(
                        output_path=str(output_path),
                        genre_id=genre_id,
                        instruments=instrument_ids,
                        num_bars=num_bars,
                        temperature=temperature,
                    )
                else:
                    # Single track generation (MonoGenerator)
                    self.generator.generate_midi(
                        output_path=str(output_path),
                        genre_id=genre_id,
                        instrument_ids=instrument_ids,
                        include_drums=True,
                        temperature=temperature,
                    )
                
                generated_files.append(str(output_path))
                self._log(f"  Saved: {filename}")
            
            if self.is_generating:
                self._log(f"\n{'='*60}")
                self._log(f" Generation complete!")
                self._log(f"  Generated: {len(generated_files)} file(s)")
                self._log(f"  Location: {output_dir}")
                self._log(f"{'='*60}\n")
                
                self.status_var.set(f"Complete! Generated {len(generated_files)} song(s).")
                
                self.root.after(0, lambda: messagebox.showinfo(
                    "Complete",
                    f"Generated {len(generated_files)} song(s)!\n\nSaved to: {output_dir}"
                ))
        
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            self._log(f"\n✗ Error: {error_msg}")
            self.status_var.set("Generation failed.")
            
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            self.root.after(0, self._reset_ui_after_generation)
    
    def _cancel_generation(self):
        """Cancel generation."""
        if messagebox.askyesno("Cancel", "Cancel generation?"):
            self.is_generating = False
            self._log("\nCancelling...")
            self.status_var.set("Cancelling...")
    
    def _reset_ui_after_generation(self):
        """Reset UI after generation."""
        self.is_generating = False
        self.generate_button['state'] = "normal"
        self.cancel_button['state'] = "disabled"
        self.progress_bar.stop()
    
    def _log(self, message: str):
        """Add message to log."""
        def _append():
            self.log_text['state'] = "normal"
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text['state'] = "disabled"
        
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.root.after(0, _append)


def main():
    """Run the GUI."""
    root = tk.Tk()
    
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    
    style.configure("Accent.TButton", font=("Helvetica", 11, "bold"), padding=10)
    
    app = MusicGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()