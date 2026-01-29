# Music Generation Pipeline CLI

A command-line interface for building datasets, training models, and generating music using the `src_v4` music generation pipeline.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install tensorflow muspy numpy h5py tqdm
```

## Quick Start

```bash
# Build a dataset from MIDI files
python -m src_v4.client.cli dataset build ./data/midi -o dataset.h5

# Train a model
python -m src_v4.client.cli train dataset.h5 -o my_model

# Generate music
python -m src_v4.client.cli generate my_model.h5 -o output.mid
```

## Usage

```
python -m src_v4.client.cli <command> [options]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `config` | Create, view, or edit configuration files |
| `dataset` | Build or inspect datasets |
| `train` | Train a model |
| `generate` | Generate music |
| `info` | Show information about models or datasets |
| `interactive` | Run interactive terminal mode |

---

## Commands Reference

### `config` - Configuration Management

#### Create a new config file

```bash
python -m src_v4.client.cli config create -o my_config.json \
    --encoder remi \
    --model transformer \
    --name my_model \
    --d-model 256 \
    --layers 4 \
    --heads 8 \
    --epochs 100 \
    --batch-size 32 \
    --seq-length 1024
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `config.json` | Output file path |
| `--encoder` | `event` | Encoder type: `event` or `remi` |
| `--model` | `transformer` | Model type: `transformer` or `lstm` |
| `--name` | `music_model` | Model name |
| `--d-model` | `256` | Model dimension |
| `--layers` | `4` | Number of layers |
| `--heads` | `8` | Number of attention heads |
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--seq-length` | `1024` | Maximum sequence length |

#### Show config contents

```bash
python -m src_v4.client.cli config show my_config.json
```

#### Edit a config value

```bash
python -m src_v4.client.cli config edit my_config.json epochs 50
python -m src_v4.client.cli config edit my_config.json learning_rate 0.0001
```

---

### `dataset` - Dataset Management

#### Build dataset from MIDI files

```bash
python -m src_v4.client.cli dataset build ./data/midi/clean_midi \
    -o ./data/datasets/my_dataset.h5 \
    --encoder remi \
    --resolution 24 \
    --seq-length 1024
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `midi_dir` | Yes | Directory containing MIDI files |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | Required | Output .h5 file path |
| `-c, --config` | None | Config file to use |
| `--encoder` | `event` | Encoder type: `event` or `remi` |
| `--resolution` | `24` | Ticks per beat |
| `--seq-length` | `1024` | Maximum sequence length |

#### Show dataset info

```bash
python -m src_v4.client.cli dataset info my_dataset.h5
```

**Output example:**
```
Dataset: my_dataset.h5
  Entries: 1542
  Genres: 15
  Active instruments: 89

Genres: ['Rock', 'Pop', 'Jazz', ...]
```

---

### `train` - Model Training

```bash
python -m src_v4.client.cli train dataset.h5 \
    -c my_config.json \
    -o ./models/my_model \
    --epochs 50 \
    --batch-size 16
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `dataset` | Yes | Dataset .h5 file |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --config` | None | Config file (uses defaults if not provided) |
| `-o, --output` | Auto | Output model path |
| `--epochs` | Config value | Override number of epochs |
| `--batch-size` | Config value | Override batch size |
| `--resume` | None | Resume from checkpoint |

---

### `generate` - Music Generation

```bash
python -m src_v4.client.cli generate my_model.h5 \
    -o output.mid \
    --genre 0 \
    --instruments 0 33 128 \
    --temperature 1.0 \
    --top-k 50 \
    --top-p 0.9
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `model` | Yes | Model bundle .h5 file |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | Required | Output MIDI file path |
| `--genre` | `0` | Genre ID for conditioning |
| `--instruments` | Auto | Instrument IDs (space-separated) |
| `--no-drums` | False | Exclude drum track |
| `--temperature` | `1.0` | Sampling temperature (higher = more random) |
| `--top-k` | `50` | Top-k sampling parameter |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--length` | Config | Maximum sequence length |

**Temperature Guide:**
- `0.5-0.7`: Conservative, repetitive patterns
- `0.8-1.0`: Balanced creativity
- `1.1-1.5`: More experimental
- `> 1.5`: Very random, may be chaotic

---

### `info` - Information Display

Show information about any model, dataset, or config file:

```bash
# Model info
python -m src_v4.client.cli info my_model.h5

# Dataset info
python -m src_v4.client.cli info my_dataset.h5

# Config info
python -m src_v4.client.cli info my_config.json
```

---

### `interactive` - Interactive Mode

Launch an interactive terminal session:

```bash
python -m src_v4.client.cli interactive
python -m src_v4.client.cli interactive -c my_config.json
```

#### Interactive Commands

Once in interactive mode, you have access to:

```
============================================================
  Music Generation Pipeline - Interactive Mode
============================================================
Type 'help' for available commands, 'quit' to exit.

music>
```

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `status` | Show current pipeline status |
| `config` | Show current configuration |
| `set <key> <value>` | Set a config value |
| `save config <file>` | Save config to file |
| `save model <file>` | Save trained model |
| `load config <file>` | Load config from file |
| `load dataset <file>` | Load dataset from file |
| `load model <file>` | Load model from file |
| `dataset build <dir> [output]` | Build dataset from MIDI directory |
| `dataset info` | Show loaded dataset info |
| `train [epochs]` | Train the model |
| `generate <output> [options]` | Generate music |
| `genres` | List available genres |
| `instruments` | List available instruments |
| `quit` / `exit` | Exit interactive mode |

#### Interactive Generation Options

```
music> generate output.mid --genre 0 --temp 1.0 --no-drums
music> generate jazz_song.mid --genre 2 --instruments 0 33 40
```

---

## Complete Workflow Examples

### Example 1: Train a Transformer Model

```bash
# 1. Create a config
python -m src_v4.client.cli config create -o rock_model.json \
    --encoder remi \
    --model transformer \
    --name rock_transformer \
    --d-model 512 \
    --layers 6 \
    --heads 8 \
    --epochs 100

# 2. Build dataset
python -m src_v4.client.cli dataset build ./data/midi/rock \
    -o ./data/datasets/rock.h5 \
    --encoder remi

# 3. Train
python -m src_v4.client.cli train ./data/datasets/rock.h5 \
    -c rock_model.json \
    -o ./models/rock_transformer

# 4. Generate
python -m src_v4.client.cli generate ./models/rock_transformer.h5 \
    -o rock_song.mid \
    --temperature 0.9
```

### Example 2: Quick Test with LSTM

```bash
# Use defaults with overrides
python -m src_v4.client.cli config create -o test.json \
    --model lstm \
    --epochs 10

python -m src_v4.client.cli dataset build ./data/midi \
    -o test_data.h5 \
    --seq-length 512

python -m src_v4.client.cli train test_data.h5 \
    -c test.json \
    --batch-size 8
```

### Example 3: Interactive Session

```bash
python -m src_v4.client.cli interactive

music> load dataset ./data/datasets/my_dataset.h5
Loaded dataset: 1542 entries

music> set epochs 50
Set epochs = 50

music> train
Training transformer for 50 epochs...
...

music> generate my_song.mid --genre 0 --temp 0.9
Generated: my_song.mid

music> quit
```

---

## Encoder Types

### Event Encoder
- Simpler token vocabulary
- Events: note-on, note-off, time-shift, velocity
- Good for: Faster training, simpler patterns

### REMI Encoder
- Richer musical representation
- Events: bar, position, pitch, duration, velocity, tempo
- Good for: Better musical structure, longer compositions

---

## Model Types

### Transformer
- Uses self-attention for long-range dependencies
- Better for capturing musical structure
- Parameters: `d_model`, `num_layers`, `num_heads`, `d_ff`

### LSTM
- Recurrent architecture
- Faster training, smaller memory footprint
- Parameters: `lstm_units`

---

## Configuration Parameters

### Dataset Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | 24 | Ticks per quarter note |
| `max_seq_length` | 1024 | Maximum token sequence length |
| `encoder_type` | "event" | "event" or "remi" |
| `train_split` | 0.8 | Training data proportion |
| `val_split` | 0.1 | Validation data proportion |

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | "transformer" | "transformer" or "lstm" |
| `d_model` | 256 | Model/embedding dimension |
| `num_layers` | 4 | Number of transformer layers |
| `num_heads` | 8 | Number of attention heads |
| `d_ff` | 1024 | Feed-forward dimension |
| `dropout` | 0.1 | Dropout rate |
| `lstm_units` | 512 | LSTM hidden units (for LSTM model) |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `epochs` | 100 | Number of training epochs |
| `learning_rate` | 1e-4 | Initial learning rate |
| `warmup_steps` | 4000 | LR warmup steps |

### Generation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `top_p` | 0.9 | Nucleus sampling threshold |

---

## Tips

1. **Start small**: Use `--max-samples 100` when building datasets for testing
2. **Monitor training**: Check TensorBoard logs in `./logs/`
3. **Temperature tuning**: Start at 1.0, adjust based on output quality
4. **Genre conditioning**: Use `genres` command to see available genre IDs
5. **Instrument selection**: Use `instruments` command to see available instrument IDs

---

## File Formats

| Extension | Description |
|-----------|-------------|
| `.h5` | HDF5 format for datasets and model bundles |
| `.json` | JSON format for configuration files |
| `.mid` | Standard MIDI format for generated music |

---

## Troubleshooting

### "No MIDI files found"
- Check that the directory contains `.mid` or `.midi` files
- Ensure files are not in deeply nested subdirectories

### "Out of memory"
- Reduce `batch_size`
- Reduce `max_seq_length`
- Use a smaller model (`d_model`, `num_layers`)

### "Poor generation quality"
- Train for more epochs
- Try different temperature values
- Use more training data
- Check that the encoder type matches training

### "Module not found"
- Ensure you're running from the project root directory
- Check that `src_v4` is in your Python path

---

## API Usage

For programmatic access, use the `MusicPipeline` class directly:

```python
from src_v4.client import MusicPipeline, PipelineConfig

# Create pipeline
config = PipelineConfig(
    encoder_type="remi",
    model_type="transformer",
    epochs=100,
)
pipeline = MusicPipeline(config)

# Build dataset
pipeline.build_dataset("./data/midi", output_path="dataset.h5")

# Train
pipeline.train()

# Generate
pipeline.generate_midi("output.mid", genre_id=0, temperature=0.9)
```

See `src_v4/client/pipeline.py` for full API documentation.

---

## Common Instrument IDs

| ID | Instrument |
|----|------------|
| 0 | Acoustic Grand Piano |
| 24 | Acoustic Guitar (nylon) |
| 25 | Acoustic Guitar (steel) |
| 32 | Acoustic Bass |
| 33 | Electric Bass (finger) |
| 40 | Violin |
| 48 | String Ensemble |
| 56 | Trumpet |
| 65 | Alto Sax |
| 73 | Flute |
| 128 | Drums |

Use `python -m src_v4.client.cli interactive` then `instruments` to see all available instruments.

---

## License

See the main project LICENSE file.
