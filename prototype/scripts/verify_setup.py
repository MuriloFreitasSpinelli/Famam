"""Verify installation based on project requirements"""

def check_imports():
    print("Checking imports...")
    
    try:
        import music21
        print(f"✓ music21: {music21.__version__}")
    except ImportError:
        print("✗ music21 not installed")
    
    try:
        import muspy
        print(f"✓ muspy: {muspy.__version__}")
    except ImportError:
        print("✗ muspy not installed")
    
    try:
        import tensorflow as tf
        print(f"✓ tensorflow: {tf.__version__}")
    except ImportError:
        print("✗ tensorflow not installed")
    
    try:
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
    except ImportError:
        print("✗ numpy not installed")
    
    try:
        import matplotlib
        print(f"✓ matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib not installed")
    
    try:
        import pandas as pd
        print(f"✓ pandas: {pd.__version__}")
    except ImportError:
        print("✗ pandas not installed")
    
    try:
        import h5py
        print(f"✓ h5py: {h5py.__version__}")
    except ImportError:
        print("✗ h5py not installed")
    
    try:
        import sklearn
        print(f"✓ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn not installed")

def check_data():
    from pathlib import Path
    
    print("\nChecking data...")
    
    lmd_root = Path("data/raw/lmd_clean")
    if lmd_root.exists():
        midi_files = list(lmd_root.rglob("*.mid"))
        print(f"✓ Found {len(midi_files)} MIDI files")
    else:
        print("✗ LMD dataset not found")

if __name__ == "__main__":
    check_imports()
    check_data()