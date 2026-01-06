from pathlib import Path
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

def create_lstm_model(vocab_size, sequence_length, 
                      embed_dim=256, 
                      hidden_dim=512, 
                      num_layers=2, 
                      dropout=0.3):
    """
    Create LSTM model for music generation
    
    Args:
        vocab_size: Size of vocabulary
        sequence_length: Length of input sequences
        embed_dim: Dimension of embedding layer
        hidden_dim: Number of LSTM units
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """
    model = models.Sequential(name='MusicLSTM')
    
    # Embedding layer - maps token indices to dense vectors
    model.add(layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=sequence_length,
        name='embedding'
    ))
    
    # LSTM layers
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1) or True  # Always return sequences
        model.add(layers.LSTM(
            hidden_dim,
            return_sequences=True,
            dropout=dropout,
            name=f'lstm_{i+1}'
        ))
    
    # Output layer - predict next token
    model.add(layers.TimeDistributed(
        layers.Dense(vocab_size, activation='softmax'),
        name='output'
    ))
    
    return model

def load_training_data():
    """Load preprocessed training data"""
    data_dir = Path("prototype/data/processed/sequences")
    
    print("Loading training data...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    
    with open(data_dir / "vocabulary.pkl", 'rb') as f:
        vocab = pickle.load(f)
    
    with open(data_dir / "sequence_config.pkl", 'rb') as f:
        config = pickle.load(f)
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Validation data: {X_val.shape}")
    print(f"✓ Vocabulary size: {vocab['vocab_size']}")
    
    return X_train, y_train, X_val, y_val, vocab, config

def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=20, batch_size=64, learning_rate=0.001):
    """
    Train the LSTM model
    """
    # Prepare targets for categorical crossentropy
    # y should be one-hot encoded or sparse categorical
    # We'll use sparse_categorical_crossentropy so no need to one-hot encode
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    
    # Setup callbacks
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    callback_list = [
        # Save best model
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # TensorBoard logging (optional)
        callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*80 + "\n")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1
    )
    
    return history

def plot_training_history(history, save_path="outputs/training_history.png"):
    """
    Plot training and validation loss/accuracy
    """
    output_dir = Path(save_path).parent
    output_dir.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training plots saved to: {save_path}")
    

def save_model_and_config(model, config, vocab):
    """
    Save trained model and configuration
    """
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Save model
    model.save(model_dir / "music_lstm.keras")
    print(f"✓ Model saved to: {model_dir / 'music_lstm.keras'}")
    
    # Save config
    with open(model_dir / "model_config.pkl", 'wb') as f:
        pickle.dump(config, f)
    print(f"✓ Config saved to: {model_dir / 'model_config.pkl'}")
    
    # Save vocabulary
    with open(model_dir / "vocabulary.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✓ Vocabulary saved to: {model_dir / 'vocabulary.pkl'}")

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_val, y_val, vocab, config = load_training_data()
    
    # Model hyperparameters
    EMBED_DIM = 256      # Embedding dimension
    HIDDEN_DIM = 512     # LSTM hidden units
    NUM_LAYERS = 2       # Number of LSTM layers
    DROPOUT = 0.3        # Dropout rate
    
    # Training hyperparameters
    EPOCHS = 20          # Number of epochs
    BATCH_SIZE = 64      # Batch size
    LEARNING_RATE = 0.001  # Learning rate
    
    print("\n" + "="*80)
    print("BUILDING LSTM MODEL")
    print("="*80)
    print(f"Vocabulary size: {vocab['vocab_size']}")
    print(f"Sequence length: {config['sequence_length']}")
    print(f"Embedding dim: {EMBED_DIM}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"Num layers: {NUM_LAYERS}")
    print(f"Dropout: {DROPOUT}")
    print("="*80 + "\n")
    
    # Create model
    model = create_lstm_model(
        vocab_size=vocab['vocab_size'],
        sequence_length=config['sequence_length'],
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # Train model
    history = train_model(
        model, 
        X_train, y_train, 
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    model_config = {
        'embed_dim': EMBED_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'sequence_length': config['sequence_length'],
        'vocab_size': vocab['vocab_size']
    }
    save_model_and_config(model, model_config, vocab)
    
    print("\n✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Best model saved to: checkpoints/best_model.keras")
    print(f"Final model saved to: models/music_lstm.keras")
    print("="*80)