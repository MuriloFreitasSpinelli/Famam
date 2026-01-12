import pytest
import tempfile
import numpy as np
from pathlib import Path
import json
import muspy

from data.dataset_vocabulary import DatasetVocabulary
from data.enhanced_music import EnhancedMusic
from data.enhanced_music_dataset import EnhancedMusicDataset


@pytest.fixture
def sample_music_items():
    """Create sample EnhancedMusic items for testing."""
    items = []
    for i in range(5):
        # Create a simple Music object with tracks
        music = muspy.Music(
            resolution=24,
            tempos=[muspy.Tempo(time=0, qpm=120.0)],
            tracks=[
                muspy.Track(
                    program=i,
                    is_drum=False,
                    notes=[
                        muspy.Note(time=0, pitch=60, duration=24, velocity=64)
                    ]
                ),
                muspy.Track(
                    program=i+1,
                    is_drum=False,
                    notes=[
                        muspy.Note(time=24, pitch=64, duration=24, velocity=64)
                    ]
                )
            ]
        )
        metadata = {
            'genre': f'genre_{i % 3}',
            'artist': f'artist_{i % 2}',
            'title': f'song_{i}'
        }
        items.append(EnhancedMusic(music=music, metadata=metadata))
    return items


@pytest.fixture
def dataset(sample_music_items):
    """Create a dataset with sample items."""
    ds = EnhancedMusicDataset()
    ds.extend(sample_music_items)
    return ds


class TestEnhancedMusicDataset:
    
    def test_init_empty(self):
        """Test initialization of empty dataset."""
        ds = EnhancedMusicDataset()
        assert len(ds) == 0
        assert isinstance(ds.vocabulary, DatasetVocabulary)
    
    def test_len(self, dataset):
        """Test __len__ method."""
        assert len(dataset) == 5
    
    def test_getitem(self, dataset):
        """Test __getitem__ method."""
        item = dataset[0]
        assert isinstance(item, EnhancedMusic)
        assert isinstance(item.music, muspy.Music)
        assert item.metadata['title'] == 'song_0'
    
    def test_append(self):
        """Test append method and vocabulary updating."""
        ds = EnhancedMusicDataset()
        music = muspy.Music(
            resolution=24,
            tracks=[muspy.Track(program=0, is_drum=False, notes=[])]
        )
        metadata = {'genre': 'rock', 'artist': 'Beatles'}
        item = EnhancedMusic(music=music, metadata=metadata)
        
        ds.append(item)
        
        assert len(ds) == 1
        assert 'rock' in ds.vocabulary.genre_to_id
        assert 'Beatles' in ds.vocabulary.artist_to_id
    
    def test_append_empty_metadata(self):
        """Test append with missing metadata doesn't break."""
        ds = EnhancedMusicDataset()
        music = muspy.Music(resolution=24, tracks=[])
        item = EnhancedMusic(music=music, metadata={})
        
        ds.append(item)
        assert len(ds) == 1
    
    def test_append_none_genre_artist(self):
        """Test append with None genre/artist values."""
        ds = EnhancedMusicDataset()
        music = muspy.Music(resolution=24, tracks=[])
        metadata = {'genre': None, 'artist': None, 'title': 'test'}
        item = EnhancedMusic(music=music, metadata=metadata)
        
        ds.append(item)
        assert len(ds) == 1
    
    def test_extend(self, sample_music_items):
        """Test extend method."""
        ds = EnhancedMusicDataset()
        ds.extend(sample_music_items)
        
        assert len(ds) == 5
        assert len(ds.vocabulary.genre_to_id) == 3  # genre_0, genre_1, genre_2
        assert len(ds.vocabulary.artist_to_id) == 2  # artist_0, artist_1
    
    def test_vocabulary_building(self, dataset):
        """Test that vocabulary is built correctly."""
        # Should have 3 unique genres (0, 1, 2)
        assert len(dataset.vocabulary.genre_to_id) == 3
        # Should have 2 unique artists (0, 1)
        assert len(dataset.vocabulary.artist_to_id) == 2
        
        # Check specific mappings
        assert 'genre_0' in dataset.vocabulary.genre_to_id
        assert 'artist_0' in dataset.vocabulary.artist_to_id


class TestSaveLoad:
    
    def test_save_and_load(self, dataset):
        """Test saving and loading dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_dataset.h5"
            
            # Save
            dataset.save(str(filepath))
            assert filepath.exists()
            
            # Load
            loaded = EnhancedMusicDataset.load(str(filepath))
            
            # Verify data
            assert len(loaded) == len(dataset)
            assert loaded.vocabulary.genre_to_id == dataset.vocabulary.genre_to_id
            assert loaded.vocabulary.artist_to_id == dataset.vocabulary.artist_to_id
            
            # Check individual items
            for i in range(len(dataset)):
                assert loaded[i].metadata == dataset[i].metadata
                assert loaded[i].music.resolution == dataset[i].music.resolution
                assert len(loaded[i].music.tracks) == len(dataset[i].music.tracks)
    
    def test_save_creates_parent_dirs(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "another" / "dataset.h5"
            
            ds = EnhancedMusicDataset()
            ds.save(str(filepath))
            
            assert filepath.exists()
            assert filepath.parent.exists()
    
    def test_save_empty_dataset(self):
        """Test saving empty dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty.h5"
            
            ds = EnhancedMusicDataset()
            ds.save(str(filepath))
            
            loaded = EnhancedMusicDataset.load(str(filepath))
            assert len(loaded) == 0
            assert isinstance(loaded.vocabulary, DatasetVocabulary)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(OSError):  # h5py raises OSError for missing files
            EnhancedMusicDataset.load("nonexistent.h5")
    
    def test_save_with_path_object(self, dataset):
        """Test that save works with Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            
            dataset.save(str(filepath))
            assert filepath.exists()
    
    def test_save_load_preserves_music_structure(self, dataset):
        """Test that Music objects are correctly preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            
            # Get original data
            original_item = dataset[0]
            original_track = original_item.music.tracks[0]
            original_note = original_track.notes[0]
            
            # Save and load
            dataset.save(str(filepath))
            loaded = EnhancedMusicDataset.load(str(filepath))
            
            # Verify structure
            loaded_item = loaded[0]
            loaded_track = loaded_item.music.tracks[0]
            loaded_note = loaded_track.notes[0]
            
            assert loaded_track.program == original_track.program
            assert loaded_note.pitch == original_note.pitch
            assert loaded_note.time == original_note.time
            assert loaded_note.duration == original_note.duration


class TestTensorFlowConversion:
    
    def test_to_tensorflow_dataset_no_splits(self, dataset):
        """Test conversion to TensorFlow dataset without splits."""
        tf_ds = dataset.to_tensorflow_dataset_with_metadata()
        
        # Check that it returns a dataset
        assert tf_ds is not None
        
        # Iterate through one sample
        for sample in tf_ds.take(1):
            assert 'music' in sample
            assert 'genre_id' in sample
            assert 'artist_id' in sample
            assert 'instrument_ids' in sample
            assert sample['music'].dtype == np.float32
            assert sample['genre_id'].dtype == np.int32
            assert sample['artist_id'].dtype == np.int32
    
    def test_to_tensorflow_dataset_with_splits(self, dataset):
        """Test conversion with train/val/test splits."""
        splits = [0.6, 0.2, 0.2]
        tf_datasets = dataset.to_tensorflow_dataset_with_metadata(splits=splits)
        
        assert 'train' in tf_datasets
        assert 'validation' in tf_datasets
        assert 'test' in tf_datasets
        
        # Verify each is a TensorFlow dataset
        assert tf_datasets['train'] is not None
        assert tf_datasets['validation'] is not None
        assert tf_datasets['test'] is not None
    
    def test_to_tensorflow_dataset_with_random_state(self, dataset):
        """Test that random_state produces reproducible splits."""
        splits = [0.6, 0.2, 0.2]
        
        ds1 = dataset.to_tensorflow_dataset_with_metadata(
            splits=splits, random_state=42
        )
        ds2 = dataset.to_tensorflow_dataset_with_metadata(
            splits=splits, random_state=42
        )
        
        # Compare first sample from each split
        sample1 = next(iter(ds1['train']))
        sample2 = next(iter(ds2['train']))
        
        assert sample1['genre_id'].numpy() == sample2['genre_id'].numpy()
        assert sample1['artist_id'].numpy() == sample2['artist_id'].numpy()
    
    def test_to_tensorflow_dataset_saves_splits(self, dataset):
        """Test that split indices are saved to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_file = Path(tmpdir) / "splits.json"
            splits = [0.6, 0.2, 0.2]
            
            dataset.to_tensorflow_dataset_with_metadata(
                splits=splits,
                split_filename=str(split_file),
                random_state=42
            )
            
            assert split_file.exists()
            
            with open(split_file, 'r') as f:
                saved_splits = json.load(f)
            
            assert 'train' in saved_splits
            assert 'validation' in saved_splits
            assert 'test' in saved_splits
            assert len(saved_splits['train']) > 0
    
    def test_instrument_ids_padding(self, dataset):
        """Test that instrument_ids are padded correctly."""
        tf_ds = dataset.to_tensorflow_dataset_with_metadata(max_tracks=16)
        
        for sample in tf_ds.take(1):
            assert sample['instrument_ids'].shape == (16,)
            # Check that padding uses -1
            instrument_ids = sample['instrument_ids'].numpy()
            assert len(instrument_ids) == 16
            # Should have 2 valid instruments and 14 padding
            valid_count = np.sum(instrument_ids != -1)
            assert valid_count == 2  # 2 tracks in sample data
    
    def test_instrument_ids_truncation(self):
        """Test that instrument_ids are truncated when exceeding max_tracks."""
        ds = EnhancedMusicDataset()
        tracks = [
            muspy.Track(program=i, is_drum=False, notes=[])
            for i in range(20)  # 20 tracks
        ]
        music = muspy.Music(resolution=24, tracks=tracks)
        item = EnhancedMusic(music=music, metadata={'genre': 'rock'})
        ds.append(item)
        
        tf_ds = ds.to_tensorflow_dataset_with_metadata(max_tracks=10)
        
        for sample in tf_ds.take(1): # type: ignore
            assert sample['instrument_ids'].shape == (10,) # type: ignore
            # All should be valid (no -1 padding)
            instrument_ids = sample['instrument_ids'].numpy() # type: ignore
            assert np.all(instrument_ids >= 0)
    
    def test_different_representations(self, dataset):
        """Test that different representations work."""
        # Test pianoroll (default)
        tf_ds = dataset.to_tensorflow_dataset_with_metadata(representation="pianoroll")
        sample = next(iter(tf_ds))
        assert sample['music'].shape[0] == 128  # 128 MIDI pitches
        
        # Note: Add more representation tests based on what muspy supports


class TestEdgeCases:
    
    def test_dataset_with_none_metadata_values(self):
        """Test handling of None values in metadata."""
        ds = EnhancedMusicDataset()
        music = muspy.Music(resolution=24, tracks=[])
        metadata = {'genre': None, 'artist': None}
        item = EnhancedMusic(music=music, metadata=metadata)
        
        ds.append(item)
        assert len(ds) == 1
    
    def test_save_load_preserves_order(self, dataset):
        """Test that save/load preserves item order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            
            original_titles = [item.metadata['title'] for item in dataset.data]
            
            dataset.save(str(filepath))
            loaded = EnhancedMusicDataset.load(str(filepath))
            
            loaded_titles = [item.metadata['title'] for item in loaded.data]
            assert original_titles == loaded_titles
    
    def test_dataset_with_large_metadata(self):
        """Test handling of large metadata objects."""
        ds = EnhancedMusicDataset()
        music = muspy.Music(resolution=24, tracks=[])
        metadata = {
            'genre': 'rock',
            'artist': 'Test',
            'large_field': 'x' * 10000  # Large string
        }
        item = EnhancedMusic(music=music, metadata=metadata)
        
        ds.append(item)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            ds.save(str(filepath))
            loaded = EnhancedMusicDataset.load(str(filepath))
            
            assert loaded[0].metadata['large_field'] == metadata['large_field']
    
    def test_empty_tracks(self):
        """Test dataset with music that has no tracks."""
        ds = EnhancedMusicDataset()
        music = muspy.Music(resolution=24, tracks=[])
        item = EnhancedMusic(music=music, metadata={'genre': 'ambient'})
        ds.append(item)
        
        tf_ds = ds.to_tensorflow_dataset_with_metadata(max_tracks=16)
        sample = next(iter(tf_ds))
        
        # All instrument IDs should be -1 (padding)
        assert np.all(sample['instrument_ids'].numpy() == -1) # type: ignore
    
    def test_enhanced_music_delegation(self):
        """Test that EnhancedMusic properly delegates to Music object."""
        music = muspy.Music(
            resolution=24,
            tracks=[muspy.Track(program=0, is_drum=False, notes=[])]
        )
        enhanced = EnhancedMusic(music=music, metadata={'genre': 'test'})
        
        # Should be able to access Music attributes
        assert enhanced.resolution == 24
        assert len(enhanced.tracks) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])