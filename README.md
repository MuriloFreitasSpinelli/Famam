# Famam — Music Generation

Famam is a deep-learning framework for generating MIDI music. It supports both polyphonic multi-track generation (all instruments simultaneously) and monophonic single-track generation, across multiple model architectures and encoding schemes.

## Features

- **Polyphonic & monophonic generation** — polyphonic mode generates all instruments in a single interleaved sequence so every instrument is aware of all others; monophonic mode generates single-track melodies
- **Multiple encodings** — interleaved multi-track, event-based, and REMI encodings; choose the representation that best suits your data and use case
- **Transformer & LSTM backbones** — Transformer with relative positional attention (based on [Music Transformer, Huang et al. 2018](https://arxiv.org/abs/1809.04281)) or stacked LSTM; both configurable in depth and width
- **Genre & instrument conditioning** — steer generation toward specific genres and instrument sets drawn from the training vocabulary
- **Full training pipeline** — dataset building from raw MIDI, configurable training, checkpoint management, model bundling
- **GUI and CLI** — graphical interface for quick generation, menu-driven CLI for full control

---

## Quick Start

### 1. Install dependencies
You need conda for for use in the cluster and is recommended in general.
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

```bash
conda create -n famam python=3.9 anaconda
source activate famam
```

```bash
conda install -n yourenvname [package]
```

### 2. Download the pre-trained model

Download both files from the [v1.0 release](https://github.com/MuriloFreitasSpinelli/Famam/releases/tag/v1.0) and place them in `models/transformer_large/`:

| File | Size | Description |
|------|------|-------------|
| `transformer_large.h5` | 14 MB | Metadata, encoder config, vocabulary |
| `model_bundle_model.keras` | 331 MB | Model weights |

### 3. Generate music

**Graphical interface:**
```bash
python run_gui.py
```

**Command-line interface:**
```bash
python run_cli.py generate
```

Load `models/transformer_large/transformer_large.h5` when prompted, then select a genre and generate.

---

## Pre-trained Model

The `transformer_large` model was trained on rock genres from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/):

- **Genres:** Alternative Rock, Rock, Hard Rock, Pop Rock, Classic Rock, Progressive Rock, Blues Rock, Soft Rock
- **Architecture:** 12-layer Transformer, 768 d_model, 12 attention heads, relative positional attention
- **Encoding:** interleaved multi-track (all instruments in one sequence)
- **Training:** 16-bar segments, max sequence length 2048

---

## Project Structure

```
Famam/
├── run_cli.py                   # CLI entry point
├── run_gui.py                   # GUI entry point
├── requirements.txt
├── configs/
│   ├── model_training/          # Training hyperparameter configs
│   │   ├── transformer_large.json
│   │   ├── transformer_small.json
│   │   ├── lstm_large.json
│   │   └── lstm_small.json
│   └── music_dataset/           # Dataset building configs
│       ├── multitrack_rock_large.json
│       └── ...
├── models/                      # Model bundles (not tracked in git)
│   └── transformer_large/
│       ├── transformer_large.h5
│       └── model_bundle_model.keras
├── data/
│   ├── midi/                    # Raw MIDI input files
│   └── datasets/                # Built HDF5 datasets
└── src/
    ├── cli/
    │   ├── __main__.py          # CLI menu router
    │   ├── experiment_cli.py    # Dataset & training CLI
    │   ├── generation_cli.py    # Generation CLI
    │   └── cluster_train.py     # Non-interactive cluster training
    ├── config/
    │   ├── training_config.py
    │   └── music_dataset_config.py
    ├── data/
    │   ├── vocabulary.py        # Genre & instrument vocabulary
    │   ├── music_dataset.py     # HDF5 dataset class
    │   ├── encoders/
    │   │   ├── multitrack_encoder.py  # Interleaved multi-track encoding
    │   │   ├── event_encoder.py
    │   │   └── remi_encoder.py
    │   └── preprocessing/
    │       └── dataset_builder.py
    ├── models/
    │   ├── base_model.py        # Autoregressive generation loop
    │   ├── transformer_model.py # Relative attention Transformer
    │   ├── lstm_model.py
    │   └── model_bundle.py      # Inference packaging
    ├── generation/
    │   ├── poly_generator.py    # Multi-track generation
    │   └── mono_generator.py    # Single-track generation
    ├── training/
    │   └── trainer.py
    └── gui/
        └── generation_GUI.py    # Tkinter GUI
```

---

## Usage

### Generation

**GUI** — browse for a model bundle, pick a genre, adjust parameters, click Generate:
```bash
python run_gui.py
```

**CLI** — interactive menus:
```bash
python run_cli.py generate
```

**Generation parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temperature | 0.9 | Higher = more creative, lower = more conservative |
| Top-k | 50 | Sample from top-k most likely tokens (0 = disabled) |
| Nucleus (top-p) | 0.92 | Sample from the smallest set whose cumulative probability ≥ p |
| Bars | 16 | Number of bars to generate |
| Max sequence length | 2048 | Token budget (can exceed training length) |
| Number of songs | 1 | Generate multiple variations in one run |

---

### Training your own model

**Step 1 — Build a dataset**

Place MIDI files under `data/midi/` and optionally provide a `genre.tsv` that maps filenames to genre labels. Then run:

```bash
python run_cli.py experiment
# → Build Dataset from config
```

Example dataset config:
```json
{
  "input_dirs": ["./data/midi/"],
  "output_path": "data/datasets/my_dataset.h5",
  "encoder_type": "multitrack",
  "segment_bars": 16,
  "max_seq_length": 2048,
  "min_tracks": 2,
  "enable_transposition": true,
  "transposition_semitones": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
}
```

**Step 2 — Train**

```bash
python run_cli.py experiment
# → Train Model
```

Key training config options (Transformer example):
```json
{
  "model_type": "transformer",
  "d_model": 768,
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072,
  "use_relative_attention": true,
  "max_seq_length": 2048,
  "batch_size": 4,
  "learning_rate": 0.0001,
  "warmup_steps": 4000,
  "epochs": 100
}
```

For an LSTM, set `"model_type": "lstm"` and replace the transformer-specific fields with `"lstm_units"` (list of units per layer, e.g. `[512, 512]`). Ready-made configs for both architectures at small and large scales are in `configs/model_training/`.

**Step 3 — Bundle for inference**

After training, package the best checkpoint into a model bundle:
```bash
python run_cli.py experiment
# → Create Model Bundle from Checkpoint
```

This saves a `.h5` metadata file and a `_model.keras` weights file that can be loaded by the CLI and GUI.

**Cluster / non-interactive training:**
```bash
python -m src.cli.cluster_train \
  --dataset data/datasets/my_dataset.h5 \
  --config configs/model_training/transformer_large.json \
  --output models/my_model/
```

---

## Architecture

### Encodings

Three encoding schemes are supported, selectable per dataset via `encoder_type` in the dataset config:

**`multitrack` — Interleaved multi-track (polyphonic)**
All instruments are merged into a single time-sorted token stream. The model generates every instrument simultaneously, attending to all others at each step:

```
BOS  genre  bar_0
  pos_0  inst_drums   pitch_36  dur_2  vel_high
  pos_0  inst_bass    pitch_40  dur_4  vel_mid
  pos_2  inst_guitar  pitch_64  dur_8  vel_mid
  pos_4  inst_drums   pitch_38  dur_2  vel_high
bar_1  ...  EOS
```

**`event` — Event-based (monophonic / single-track)**
A sequential stream of note-on, pitch, velocity, duration, and note-off events. Simpler and faster to train, suited for single-instrument or melody generation.

**`remi` — REMI (monophonic / single-track)**
REMI-style encoding with explicit bar and position markers, following [Pop Music Transformer (Huang & Yang, 2020)](https://arxiv.org/abs/2002.00212).

### Model backbones

**Transformer** — uses relative multi-head attention where position scores are computed as:

```
score = (Q @ K^T + Q @ E_rel^T) / sqrt(d_k)
```

`E_rel` is a learnable matrix indexed by relative token distance, clipped to `[-max_relative_position, max_relative_position]`. This lets the model learn structural patterns — *this phrase repeats 4 bars later* — rather than absolute positions. KV-caching is used during inference for efficient token-by-token generation.

**LSTM** — stacked LSTM layers with optional bidirectional encoding. Lighter to train and faster to run, at the cost of shorter effective context.

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- muspy
- h5py
- numpy

See `requirements.txt` for the full list.

---

## Acknowledgements

- [Music Transformer](https://arxiv.org/abs/1809.04281) — Huang et al., 2018
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) — Colin Raffel
- [muspy](https://github.com/salu133445/muspy) — music processing library
