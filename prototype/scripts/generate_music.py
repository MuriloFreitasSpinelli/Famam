from pathlib import Path
import numpy as np
import pickle
import tensorflow as tf
import muspy

def load_trained_model():
    """Load the trained model and associated data"""
    model_dir = Path("models")
    
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_dir / "music_lstm.keras")
    
    with open(model_dir / "vocabulary.pkl", 'rb') as f:
        vocab = pickle.load(f)
    
    with open(model_dir / "model_config.pkl", 'rb') as f:
        config = pickle.load(f)
    
    print(f"✓ Model loaded")
    print(f"✓ Vocabulary size: {vocab['vocab_size']}")
    print(f"✓ Sequence length: {config['sequence_length']}")
    
    return model, vocab, config

def sample_with_temperature(predictions, temperature=1.0):
    """
    Sample from probability distribution with temperature
    
    Args:
        predictions: Model output probabilities
        temperature: Higher = more random, lower = more conservative
                    1.0 = unchanged, 0.5 = more conservative, 1.5 = more random
    """
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    
    # Sample from the distribution
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

def generate_sequence(model, vocab, config, seed_sequence=None, 
                     num_tokens=200, temperature=1.0):
    """
    Generate a sequence of tokens using the trained model
    
    Args:
        model: Trained LSTM model
        vocab: Vocabulary dictionary
        config: Model configuration
        seed_sequence: Starting sequence (if None, random seed)
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature
    """
    sequence_length = config['sequence_length']
    idx_to_token = vocab['idx_to_token']
    token_to_idx = vocab['token_to_idx']
    
    # Initialize seed sequence
    if seed_sequence is None:
        # Random seed from vocabulary
        seed_idx = np.random.randint(0, vocab['vocab_size'], size=sequence_length)
    else:
        # Convert seed tokens to indices
        seed_idx = np.array([token_to_idx.get(int(token), 0) for token in seed_sequence])
        if len(seed_idx) < sequence_length:
            # Pad if too short
            seed_idx = np.pad(seed_idx, (sequence_length - len(seed_idx), 0), 
                            mode='constant', constant_values=0)
        elif len(seed_idx) > sequence_length:
            # Truncate if too long
            seed_idx = seed_idx[-sequence_length:]
    
    # Current sequence
    current_sequence = seed_idx.copy()
    generated_tokens = []
    
    print(f"Generating {num_tokens} tokens with temperature={temperature}...")
    
    for i in range(num_tokens):
        # Prepare input
        x = current_sequence.reshape(1, sequence_length)
        
        # Predict next token
        predictions = model.predict(x, verbose=0)
        
        # Get prediction for last position
        next_token_probs = predictions[0, -1, :]
        
        # Sample next token
        next_token_idx = sample_with_temperature(next_token_probs, temperature)
        
        # Convert index to token
        next_token = idx_to_token[next_token_idx]
        generated_tokens.append(int(next_token))
        
        # Update sequence (slide window)
        current_sequence = np.append(current_sequence[1:], next_token_idx)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_tokens} tokens")
    
    print(f"✓ Generation complete!")
    
    return np.array(generated_tokens)

def tokens_to_midi(tokens, output_path, resolution=24):
    """
    Convert generated tokens (MIDI pitches) to MIDI file
    
    Args:
        tokens: Array of MIDI pitch numbers
        output_path: Where to save MIDI file
        resolution: Ticks per quarter note
    """
    # Create MusPy Music object
    music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(0, 120)])
    
    # Create a track
    track = muspy.Track(program=0, is_drum=False, name="Generated")
    
    # Convert tokens to notes
    # Simple approach: each token is a note with fixed duration
    current_time = 0
    note_duration = resolution // 4  # Quarter note duration
    
    for pitch in tokens:
        # Skip invalid pitches
        if pitch < 0 or pitch > 127:
            continue
        
        note = muspy.Note(
            time=current_time,
            pitch=int(pitch),
            duration=note_duration,
            velocity=80
        )
        track.notes.append(note)
        current_time += note_duration
    
    music.tracks.append(track)
    
    # Write MIDI file
    muspy.write(output_path, music)
    print(f"✓ MIDI saved to: {output_path}")

def generate_multiple_samples(model, vocab, config, 
                              num_samples=5, 
                              num_tokens=200,
                              temperatures=[0.5, 0.8, 1.0, 1.2, 1.5]):
    """
    Generate multiple samples with different temperatures
    """
    output_dir = Path("outputs/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING MUSIC SAMPLES")
    print("="*80)
    
    for i, temp in enumerate(temperatures, 1):
        print(f"\n[{i}/{len(temperatures)}] Temperature: {temp}")
        print("-"*80)
        
        # Generate sequence
        tokens = generate_sequence(
            model, vocab, config,
            num_tokens=num_tokens,
            temperature=temp
        )
        
        # Convert to MIDI
        output_path = output_dir / f"generated_temp_{temp:.1f}.mid"
        tokens_to_midi(tokens, output_path)
        
        # Basic statistics
        unique_pitches = len(np.unique(tokens))
        pitch_range = tokens.max() - tokens.min()
        print(f"  Unique pitches: {unique_pitches}")
        print(f"  Pitch range: {pitch_range} semitones")
        print(f"  Average pitch: {tokens.mean():.1f}")
    
    print("\n✅ All samples generated!")
    print(f"Location: {output_dir}")

def generate_with_seed(model, vocab, config, seed_file=None):
    """
    Generate music using a seed from an existing MIDI file
    """
    if seed_file is None:
        print("No seed file provided, using random seed")
        return generate_sequence(model, vocab, config)
    
    print(f"Loading seed from: {seed_file}")
    
    # Load seed MIDI
    music = muspy.read(seed_file)
    
    # Extract pitches from first track
    if len(music.tracks) == 0:
        print("No tracks in seed file, using random seed")
        return generate_sequence(model, vocab, config)
    
    seed_pitches = [note.pitch for note in music.tracks[0].notes[:config['sequence_length']]]
    
    print(f"✓ Loaded {len(seed_pitches)} notes from seed")
    
    # Generate continuation
    tokens = generate_sequence(
        model, vocab, config,
        seed_sequence=seed_pitches,
        num_tokens=200,
        temperature=1.0
    )
    
    return tokens

if __name__ == "__main__":
    # Load model
    model, vocab, config = load_trained_model()
    
    print("\n" + "="*80)
    print("MUSIC GENERATION")
    print("="*80)
    
    # Option 1: Generate multiple samples with different temperatures
    generate_multiple_samples(
        model, vocab, config,
        num_samples=5,
        num_tokens=300,  # Generate 300 notes
        temperatures=[0.5, 0.8, 1.0, 1.2, 1.5]
    )
    
    # Option 2: Generate with seed (uncomment to use)
    # seed_file = Path("data/processed/monophonic_filtered/some_file.mid")
    # if seed_file.exists():
    #     tokens = generate_with_seed(model, vocab, config, seed_file)
    #     output_path = Path("outputs/generated/generated_with_seed.mid")
    #     tokens_to_midi(tokens, output_path)
    
    print("\n" + "="*80)
    print("✅ GENERATION COMPLETE!")
    print("="*80)
    print("Generated MIDI files are in: outputs/generated/")
    print("\nNext steps:")
    print("1. Listen to the generated MIDI files")
    print("2. Try different temperatures (lower=conservative, higher=creative)")
    print("3. Experiment with longer sequences")
    print("4. Use seed sequences from your favorite melodies")