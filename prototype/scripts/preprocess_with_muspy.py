from pathlib import Path
import muspy
import numpy as np
import pickle
from collections import Counter

def preprocess_midi_file(midi_path, target_resolution=24):
    """
    Preprocess a single MIDI file with MusPy
    """
    try:
        # Read MIDI file
        music = muspy.read(midi_path)
        
        # Adjust resolution (ticks per quarter note)
        music = muspy.adjust_resolution(music, target_resolution)
        
        # Note: quantize is not available in newer MusPy versions
        # The adjust_resolution already handles most timing issues
        
        return music
        
    except Exception as e:
        print(f"  ✗ Error processing {midi_path.name}: {e}")
        return None

def music_to_tokens(music):
    """
    Convert MusPy Music object to simple pitch sequence
    """
    try:
        # Extract notes from all tracks
        all_notes = []
        
        for track in music.tracks:
            for note in track.notes:
                all_notes.append({
                    'time': note.time,
                    'pitch': note.pitch,
                    'duration': note.duration,
                    'velocity': note.velocity
                })
        
        # Sort by time
        all_notes.sort(key=lambda x: x['time'])
        
        # Simple pitch sequence (just MIDI pitch numbers)
        tokens = [note['pitch'] for note in all_notes]
        
        return np.array(tokens, dtype=np.int32)
        
    except Exception as e:
        print(f"  ✗ Error encoding: {e}")
        return None

def preprocess_dataset(input_dir, output_dir, resolution=24):
    """
    Preprocess all MIDI files in directory
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    midi_files = list(input_dir.glob("*.mid"))
    
    print(f"Preprocessing {len(midi_files)} MIDI files with MusPy")
    print(f"Target resolution: {resolution} ticks per quarter note")
    print("="*80)
    
    all_tokens = []
    processed_files = []
    failed_files = []
    
    for i, midi_path in enumerate(midi_files, 1):
        print(f"[{i}/{len(midi_files)}] {midi_path.name}", end='')
        
        # Preprocess with MusPy
        music = preprocess_midi_file(midi_path, resolution)
        
        if music is None:
            failed_files.append(midi_path.name)
            print(" ✗ FAILED")
            continue
        
        # Convert to tokens
        tokens = music_to_tokens(music)
        
        if tokens is None or len(tokens) == 0:
            failed_files.append(midi_path.name)
            print(" ✗ NO TOKENS")
            continue
        
        # Filter out very short sequences
        if len(tokens) < 20:
            failed_files.append(midi_path.name)
            print(" ✗ TOO SHORT")
            continue
        
        all_tokens.append(tokens)
        processed_files.append({
            'filename': midi_path.name,
            'num_tokens': len(tokens),
            'token_min': int(tokens.min()),
            'token_max': int(tokens.max()),
        })
        
        print(f" ✓ ({len(tokens)} tokens)")
    
    print("\n" + "="*80)
    print(f"✅ Successfully processed: {len(processed_files)}/{len(midi_files)}")
    print(f"❌ Failed: {len(failed_files)}")
    
    if len(processed_files) == 0:
        print("⚠ No files processed successfully!")
        return None, None
    
    # Concatenate all token sequences
    print("\n📊 TOKEN STATISTICS:")
    print("-"*80)
    
    total_tokens = sum(len(t) for t in all_tokens)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per file: {total_tokens/len(all_tokens):.0f}")
    
    # Analyze vocabulary
    all_tokens_flat = np.concatenate(all_tokens)
    unique_tokens = np.unique(all_tokens_flat)
    vocab_size = len(unique_tokens)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Token range: {unique_tokens.min()} to {unique_tokens.max()}")
    
    # Token frequency
    token_counts = Counter(all_tokens_flat)
    print(f"\nMost common tokens:")
    for token, count in token_counts.most_common(10):
        # Convert MIDI pitch to note name for readability
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = note_names[token % 12]
        octave = (token // 12) - 1
        print(f"  Token {token} ({note_name}{octave}): {count:,} times ({count/len(all_tokens_flat)*100:.2f}%)")
    
    # Save preprocessed data
    print("\n💾 SAVING PREPROCESSED DATA:")
    print("-"*80)
    
    # Save as pickle for easy loading
    tokens_file = output_dir / "all_tokens.pkl"
    with open(tokens_file, 'wb') as f:
        pickle.dump(all_tokens, f)
    print(f"✓ Saved tokens: {tokens_file}")
    
    # Save vocabulary
    vocab_file = output_dir / "vocabulary.pkl"
    with open(vocab_file, 'wb') as f:
        pickle.dump({
            'vocab_size': vocab_size,
            'unique_tokens': unique_tokens.tolist(),
            'token_to_idx': {int(token): idx for idx, token in enumerate(unique_tokens)},
            'idx_to_token': {idx: int(token) for idx, token in enumerate(unique_tokens)}
        }, f)
    print(f"✓ Saved vocabulary: {vocab_file}")
    
    # Save metadata
    import pandas as pd
    df = pd.DataFrame(processed_files)
    metadata_file = output_dir / "preprocessing_metadata.csv"
    df.to_csv(metadata_file, index=False)
    print(f"✓ Saved metadata: {metadata_file}")
    
    # Save config
    config = {
        'resolution': resolution,
        'vocab_size': vocab_size,
        'total_tokens': total_tokens,
        'num_files': len(processed_files)
    }
    config_file = output_dir / "preprocessing_config.pkl"
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)
    print(f"✓ Saved config: {config_file}")
    
    print("\n✅ Preprocessing complete!")
    
    return all_tokens, config

if __name__ == "__main__":
    input_dir = Path("prototype/data/processed/monophonic_filtered")
    output_dir = Path("prototype/data/processed/tokens")
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("Run extract_monophonic.py and filter_best_parts.py first!")
    else:
        preprocess_dataset(
            input_dir, 
            output_dir, 
            resolution=24
        )