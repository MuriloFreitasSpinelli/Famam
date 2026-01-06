from pathlib import Path
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def create_sequences(tokens, sequence_length=64, step=1):
    """
    Create sliding window sequences from token array
    
    Args:
        tokens: Array of tokens
        sequence_length: Length of each sequence
        step: Step size for sliding window (1 = maximum overlap)
    
    Returns:
        X: Input sequences (all but last token)
        y: Target sequences (all but first token)
    """
    sequences = []
    
    # Use numpy's sliding window view for efficiency
    if len(tokens) < sequence_length + 1:
        return None, None
    
    # Create sliding windows
    for i in range(0, len(tokens) - sequence_length, step):
        seq = tokens[i:i + sequence_length + 1]
        sequences.append(seq)
    
    if len(sequences) == 0:
        return None, None
    
    sequences = np.array(sequences)
    
    # X = input sequences (all but last token)
    # y = target sequences (all but first token) - what we want to predict
    X = sequences[:, :-1]
    y = sequences[:, 1:]
    
    return X, y

def create_dataset(tokens_list, sequence_length=64, step=32):
    """
    Create training dataset from list of token sequences
    
    Args:
        tokens_list: List of token arrays (one per file)
        sequence_length: Length of each training sequence
        step: Step size for sliding window
    """
    all_X = []
    all_y = []
    
    print(f"Creating sequences with length={sequence_length}, step={step}")
    print("="*80)
    
    for i, tokens in enumerate(tokens_list, 1):
        X, y = create_sequences(tokens, sequence_length, step)
        
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            print(f"[{i}/{len(tokens_list)}] Created {len(X)} sequences from {len(tokens)} tokens")
        else:
            print(f"[{i}/{len(tokens_list)}] Skipped (too short)")
    
    # Concatenate all sequences
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print("\n" + "="*80)
    print(f"✅ Total sequences created: {len(X):,}")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    return X, y

def map_to_indices(sequences, token_to_idx):
    """
    Map token values to vocabulary indices
    
    Args:
        sequences: Array of token sequences
        token_to_idx: Dictionary mapping tokens to indices
    """
    mapped = np.zeros_like(sequences)
    
    for i in range(sequences.shape[0]):
        for j in range(sequences.shape[1]):
            token = sequences[i, j]
            mapped[i, j] = token_to_idx.get(token, 0)  # Default to 0 if not found
    
    return mapped

def prepare_training_data(sequence_length=64, step=32, test_size=0.2, val_size=0.1):
    """
    Complete pipeline to prepare training data
    """
    # Load preprocessed tokens
    tokens_dir = Path("prototype/data/processed/tokens")
    
    print("Loading preprocessed data...")
    with open(tokens_dir / "all_tokens.pkl", 'rb') as f:
        tokens_list = pickle.load(f)
    
    with open(tokens_dir / "vocabulary.pkl", 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"✓ Loaded {len(tokens_list)} files")
    print(f"✓ Vocabulary size: {vocab['vocab_size']}")
    
    # Create sequences
    print("\n" + "="*80)
    print("CREATING SEQUENCES")
    print("="*80)
    X, y = create_dataset(tokens_list, sequence_length, step)
    
    # Map to vocabulary indices
    print("\nMapping tokens to vocabulary indices...")
    X_mapped = map_to_indices(X, vocab['token_to_idx'])
    y_mapped = map_to_indices(y, vocab['token_to_idx'])
    
    print("✓ Tokens mapped to indices")
    
    # Split into train/validation/test
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_mapped, y_mapped, 
        test_size=test_size, 
        random_state=42
    )
    
    # Second split: separate validation from train
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=42
    )
    
    print(f"Train set: {len(X_train):,} sequences ({len(X_train)/len(X_mapped)*100:.1f}%)")
    print(f"Validation set: {len(X_val):,} sequences ({len(X_val)/len(X_mapped)*100:.1f}%)")
    print(f"Test set: {len(X_test):,} sequences ({len(X_test)/len(X_mapped)*100:.1f}%)")
    
    # Save processed data
    output_dir = Path("prototype/data/processed/sequences")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING SEQUENCES")
    print("="*80)
    
    # Save as numpy arrays
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    print(f"✓ Saved train data: {output_dir / 'X_train.npy'}")
    print(f"✓ Saved validation data: {output_dir / 'X_val.npy'}")
    print(f"✓ Saved test data: {output_dir / 'X_test.npy'}")
    
    # Save configuration
    config = {
        'sequence_length': sequence_length,
        'step': step,
        'vocab_size': vocab['vocab_size'],
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'total_sequences': len(X_mapped)
    }
    
    with open(output_dir / "sequence_config.pkl", 'wb') as f:
        pickle.dump(config, f)
    
    print(f"✓ Saved config: {output_dir / 'sequence_config.pkl'}")
    
    # Also save vocabulary for easy access during training
    import shutil
    shutil.copy(tokens_dir / "vocabulary.pkl", output_dir / "vocabulary.pkl")
    print(f"✓ Copied vocabulary to sequences folder")
    
    print("\n✅ TRAINING DATA READY!")
    print("="*80)
    print(f"Location: {output_dir}")
    print(f"Sequence length: {sequence_length}")
    print(f"Vocabulary size: {vocab['vocab_size']}")
    print(f"Total sequences: {len(X_mapped):,}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, config

if __name__ == "__main__":
    # Hyperparameters
    SEQUENCE_LENGTH = 64  # How many notes to look at
    STEP = 32  # Step size (32 = 50% overlap, 1 = maximum overlap but slower)
    TEST_SIZE = 0.2  # 20% for testing
    VAL_SIZE = 0.1  # 10% for validation
    
    print("CREATING TRAINING SEQUENCES")
    print("="*80)
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Step size: {STEP}")
    print(f"Train/Val/Test split: {100-TEST_SIZE*100-VAL_SIZE*100:.0f}/{VAL_SIZE*100:.0f}/{TEST_SIZE*100:.0f}")
    print("="*80 + "\n")
    
    prepare_training_data(
        sequence_length=SEQUENCE_LENGTH,
        step=STEP,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE
    )