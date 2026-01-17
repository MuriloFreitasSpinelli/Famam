"""
Simple test file for enhanced dataset creation.
Tests the actual usage with real file structure.

Run this file directly: python test_simple_dataset.py
"""

import tempfile
import shutil
from pathlib import Path
import muspy


def create_test_structure():
    """
    Create a test directory structure that mimics the real data:
    
    clean_midi/
      Beatles/
        song1.mid
        song2.mid
      Queen/
        song3.mid
      Coltrane/
        song4.mid
      genre.tsv
    """
    temp_dir = Path(tempfile.mkdtemp())
    clean_midi_dir = temp_dir / "clean_midi"
    
    # Create artist folders
    beatles_dir = clean_midi_dir / "Beatles"
    queen_dir = clean_midi_dir / "Queen"
    coltrane_dir = clean_midi_dir / "Coltrane"
    
    beatles_dir.mkdir(parents=True)
    queen_dir.mkdir(parents=True)
    coltrane_dir.mkdir(parents=True)
    
    # Create a simple MIDI music object
    music = muspy.Music(
        resolution=24,
        tempos=[muspy.Tempo(time=0, qpm=120)],
        tracks=[
            muspy.Track(
                program=0,  # Piano
                is_drum=False,
                notes=[
                    muspy.Note(time=0, pitch=60, duration=24, velocity=64),
                    muspy.Note(time=24, pitch=64, duration=24, velocity=64),
                    muspy.Note(time=48, pitch=67, duration=24, velocity=64),
                    muspy.Note(time=72, pitch=72, duration=48, velocity=64),
                ]
            )
        ]
    )
    
    # Save MIDI files
    muspy.write_midi(beatles_dir / "song1.mid", music)
    muspy.write_midi(beatles_dir / "song2.mid", music)
    muspy.write_midi(queen_dir / "song3.mid", music)
    muspy.write_midi(coltrane_dir / "song4.mid", music)
    
    # Create genre.tsv file
    genre_tsv = clean_midi_dir / "genre.tsv"
    with open(genre_tsv, 'w', encoding='utf-8') as f:
        f.write("Beatles/song1\trock\n")
        f.write("Beatles/song2\trock\n")
        f.write("Queen/song3\trock\n")
        f.write("Coltrane/song4\tjazz\n")
    
    print(f"✓ Created test structure at: {clean_midi_dir}")
    print(f"  - Beatles/song1.mid (genre: rock)")
    print(f"  - Beatles/song2.mid (genre: rock)")
    print(f"  - Queen/song3.mid (genre: rock)")
    print(f"  - Coltrane/song4.mid (genre: jazz)")
    print(f"  - genre.tsv")
    
    return temp_dir, clean_midi_dir


def test_1_basic_usage():
    """Test 1: Basic usage - create dataset from MIDI files with genre TSV."""
    print("\n" + "="*60)
    print("TEST 1: Basic Dataset Creation")
    print("="*60)
    
    temp_dir, clean_midi_dir = create_test_structure()
    
    try:
        from data.configs.enhanced_dataset_config import EnhancedDatasetConfig
        from data.create_enhanced_dataset import create_and_save_dataset
        
        # Create config
        config = EnhancedDatasetConfig(
            dataset_name="test_basic",
            input_dirs=[str(clean_midi_dir)],
            output_path=str(temp_dir / "dataset.h5"),
            genre_tsv_path=str(clean_midi_dir / "genre.tsv"),
            verbose=True,
            min_notes=1
        )
        
        print("\n" + "-"*60)
        print("Creating dataset...")
        print("-"*60)
        
        # Create dataset
        dataset = create_and_save_dataset(config)
        
        print("\n" + "-"*60)
        print("Verifying dataset...")
        print("-"*60)
        
        # Verify
        assert len(dataset) == 4, f"Expected 4 samples, got {len(dataset)}"
        print(f"✓ Dataset has correct number of samples: {len(dataset)}")
        
        assert dataset.vocabulary.num_genres == 2, f"Expected 2 genres, got {dataset.vocabulary.num_genres}"
        print(f"✓ Dataset has correct number of genres: {dataset.vocabulary.num_genres}")
        
        assert dataset.vocabulary.num_artists == 3, f"Expected 3 artists, got {dataset.vocabulary.num_artists}"
        print(f"✓ Dataset has correct number of artists: {dataset.vocabulary.num_artists}")
        
        # Check metadata
        print("\n" + "-"*60)
        print("Checking metadata extraction...")
        print("-"*60)
        
        genres_found = set()
        artists_found = set()
        
        for item in dataset.data:
            if 'genre' in item.metadata:
                genres_found.add(item.metadata['genre'])
            if 'artist' in item.metadata:
                artists_found.add(item.metadata['artist'])
        
        assert 'rock' in genres_found, "Rock genre not found"
        assert 'jazz' in genres_found, "Jazz genre not found"
        print(f"✓ Genres found: {genres_found}")
        
        assert 'Beatles' in artists_found, "Beatles not found"
        assert 'Queen' in artists_found, "Queen not found"
        assert 'Coltrane' in artists_found, "Coltrane not found"
        print(f"✓ Artists found: {artists_found}")
        
        # Verify files exist
        dataset_file = Path(config.output_path)
        config_file = dataset_file.with_suffix('.config.json')
        
        assert dataset_file.exists(), "Dataset file not created"
        assert config_file.exists(), "Config file not created"
        print(f"✓ Dataset saved to: {dataset_file}")
        print(f"✓ Config saved to: {config_file}")
        
        shutil.rmtree(temp_dir)
        print("\n✓✓✓ TEST 1 PASSED ✓✓✓")
        return True
        
    except Exception as e:
        print(f"\n✗✗✗ TEST 1 FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return False


def test_2_load_saved_dataset():
    """Test 2: Load a saved dataset."""
    print("\n" + "="*60)
    print("TEST 2: Load Saved Dataset")
    print("="*60)
    
    temp_dir, clean_midi_dir = create_test_structure()
    
    try:
        from data.configs.enhanced_dataset_config import EnhancedDatasetConfig
        from data.scripts.create_enhanced_dataset import create_and_save_dataset
        from data.enhanced_music_dataset import EnhancedMusicDataset
        
        # Create and save dataset
        config = EnhancedDatasetConfig(
            dataset_name="test_load",
            input_dirs=[str(clean_midi_dir)],
            output_path=str(temp_dir / "dataset.h5"),
            genre_tsv_path=str(clean_midi_dir / "genre.tsv"),
            verbose=False,
            min_notes=1
        )
        
        original_dataset = create_and_save_dataset(config)
        print(f"✓ Created dataset with {len(original_dataset)} samples")
        
        # Load dataset
        loaded_dataset = EnhancedMusicDataset.load(str(temp_dir / "dataset.h5"))
        print(f"✓ Loaded dataset with {len(loaded_dataset)} samples")
        
        # Verify
        assert len(loaded_dataset) == len(original_dataset)
        print("✓ Loaded dataset matches original size")
        
        assert loaded_dataset.vocabulary.num_genres == original_dataset.vocabulary.num_genres
        assert loaded_dataset.vocabulary.num_artists == original_dataset.vocabulary.num_artists
        print("✓ Vocabulary matches original")
        
        shutil.rmtree(temp_dir)
        print("\n✓✓✓ TEST 2 PASSED ✓✓✓")
        return True
        
    except Exception as e:
        print(f"\n✗✗✗ TEST 2 FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return False


def test_3_tensorflow_conversion():
    """Test 3: Convert to TensorFlow dataset."""
    print("\n" + "="*60)
    print("TEST 3: TensorFlow Dataset Conversion")
    print("="*60)
    
    temp_dir, clean_midi_dir = create_test_structure()
    
    try:
        from data.configs.enhanced_dataset_config import EnhancedDatasetConfig
        from data.create_enhanced_dataset import create_and_save_dataset
        
        # Create dataset
        config = EnhancedDatasetConfig(
            dataset_name="test_tf",
            input_dirs=[str(clean_midi_dir)],
            output_path=str(temp_dir / "dataset.h5"),
            genre_tsv_path=str(clean_midi_dir / "genre.tsv"),
            verbose=False,
            min_notes=1
        )
        
        dataset = create_and_save_dataset(config)
        print(f"✓ Created dataset with {len(dataset)} samples")
        
        # Convert to TensorFlow
        print("\nConverting to TensorFlow datasets...")
        tf_datasets = dataset.to_tensorflow_dataset_with_metadata(
            representation='pianoroll',
            splits=[0.5, 0.25, 0.25],
            random_state=42
        )
        
        assert 'train' in tf_datasets
        assert 'validation' in tf_datasets
        assert 'test' in tf_datasets
        print("✓ Created train/validation/test splits")
        
        # Get a sample
        sample = next(iter(tf_datasets['train']))
        
        assert 'music' in sample
        assert 'genre_id' in sample
        assert 'artist_id' in sample
        assert 'instrument_ids' in sample
        
        print(f"✓ Sample structure correct:")
        print(f"  - music shape: {sample['music'].shape}")
        print(f"  - genre_id: {sample['genre_id'].numpy()}")
        print(f"  - artist_id: {sample['artist_id'].numpy()}")
        print(f"  - instrument_ids shape: {sample['instrument_ids'].shape}")
        
        shutil.rmtree(temp_dir)
        print("\n✓✓✓ TEST 3 PASSED ✓✓✓")
        return True
        
    except Exception as e:
        print(f"\n✗✗✗ TEST 3 FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return False


def test_4_filtering():
    """Test 4: Dataset filtering."""
    print("\n" + "="*60)
    print("TEST 4: Dataset Filtering")
    print("="*60)
    
    temp_dir, clean_midi_dir = create_test_structure()
    
    try:
        from data.configs.enhanced_dataset_config import EnhancedDatasetConfig
        from data.create_enhanced_dataset import create_and_save_dataset
        
        print("\nTest 4a: No filtering (all files)")
        config1 = EnhancedDatasetConfig(
            dataset_name="test_no_filter",
            input_dirs=[str(clean_midi_dir)],
            output_path=str(temp_dir / "dataset1.h5"),
            genre_tsv_path=str(clean_midi_dir / "genre.tsv"),
            verbose=False,
            min_notes=1
        )
        dataset1 = create_and_save_dataset(config1)
        print(f"✓ No filtering: {len(dataset1)} samples")
        assert len(dataset1) == 4
        
        print("\nTest 4b: With max_samples limit")
        config2 = EnhancedDatasetConfig(
            dataset_name="test_limited",
            input_dirs=[str(clean_midi_dir)],
            output_path=str(temp_dir / "dataset2.h5"),
            genre_tsv_path=str(clean_midi_dir / "genre.tsv"),
            verbose=False,
            min_notes=1,
            max_samples=2
        )
        dataset2 = create_and_save_dataset(config2)
        print(f"✓ Max samples=2: {len(dataset2)} samples")
        assert len(dataset2) <= 2
        
        print("\nTest 4c: With min_notes filter")
        config3 = EnhancedDatasetConfig(
            dataset_name="test_min_notes",
            input_dirs=[str(clean_midi_dir)],
            output_path=str(temp_dir / "dataset3.h5"),
            genre_tsv_path=str(clean_midi_dir / "genre.tsv"),
            verbose=False,
            min_notes=100  # High threshold - should filter out test files
        )
        dataset3 = create_and_save_dataset(config3)
        print(f"✓ Min notes=100: {len(dataset3)} samples (filtered out)")
        assert len(dataset3) == 0
        
        shutil.rmtree(temp_dir)
        print("\n✓✓✓ TEST 4 PASSED ✓✓✓")
        return True
        
    except Exception as e:
        print(f"\n✗✗✗ TEST 4 FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return False


def test_5_config_save_load():
    """Test 5: Save and load configuration."""
    print("\n" + "="*60)
    print("TEST 5: Config Save/Load")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        from data.configs.enhanced_dataset_config import EnhancedDatasetConfig
        
        # Create config
        config = EnhancedDatasetConfig(
            dataset_name="test_config",
            input_dirs=["./data/midi"],
            output_path="./output.h5",
            genre_tsv_path="./genre.tsv",
            max_samples=50,
            min_notes=20,
            resolution=24
        )
        
        # Save
        config_path = temp_dir / "test_config.json"
        config.save(str(config_path))
        print(f"✓ Config saved to {config_path}")
        
        # Load
        loaded_config = EnhancedDatasetConfig.load(str(config_path))
        print(f"✓ Config loaded")
        
        # Verify
        assert loaded_config.dataset_name == config.dataset_name
        assert loaded_config.max_samples == config.max_samples
        assert loaded_config.min_notes == config.min_notes
        assert loaded_config.genre_tsv_path == config.genre_tsv_path
        print("✓ Loaded config matches original")
        
        shutil.rmtree(temp_dir)
        print("\n✓✓✓ TEST 5 PASSED ✓✓✓")
        return True
        
    except Exception as e:
        print(f"\n✗✗✗ TEST 5 FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return False


def run_all_tests():
    """Run all tests and show summary."""
    print("\n" + "="*60)
    print("ENHANCED DATASET CREATION - USAGE TESTS")
    print("="*60)
    print("\nThis will test the actual usage of the dataset creation system.")
    print("It creates temporary test files automatically.\n")
    
    tests = [
        ("Basic Dataset Creation", test_1_basic_usage),
        ("Load Saved Dataset", test_2_load_saved_dataset),
        ("TensorFlow Conversion", test_3_tensorflow_conversion),
        ("Dataset Filtering", test_4_filtering),
        ("Config Save/Load", test_5_config_save_load),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your dataset creation is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_all_tests()
    sys.exit(exit_code)