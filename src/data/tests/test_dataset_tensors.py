import pytest
import tempfile
import numpy as np
import tensorflow as tf
import muspy
from pathlib import Path
from unittest.mock import patch

from data.dataset_tensors import save_tensors, load_tensors, tensors_exist, TENSORS_DIR
from data.enhanced_music import EnhancedMusic
from data.enhanced_music_dataset import EnhancedMusicDataset
from data.configs.data_tensors_config import DataTensorsConfig
from data.scripts.generate_tensors import generate_tensors, get_tensors


@pytest.fixture
def sample_dataset():
    """Create a sample EnhancedMusicDataset for testing."""
    ds = EnhancedMusicDataset()
    for i in range(10):
        music = muspy.Music(
            resolution=24,
            tracks=[muspy.Track(program=i % 5, is_drum=False, notes=[
                muspy.Note(time=0, pitch=60 + i, duration=24, velocity=64)
            ])]
        )
        ds.append(EnhancedMusic(music=music, metadata={
            'genre': f'genre_{i % 3}',
            'artist': f'artist_{i % 2}'
        }))
    return ds


@pytest.fixture
def sample_tf_datasets(sample_dataset):
    """Create sample TensorFlow datasets."""
    return sample_dataset.to_tensorflow_dataset_with_metadata(
        representation='piano-roll',
        splits=[0.6, 0.2, 0.2],
        random_state=42
    )


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample config for testing."""
    return DataTensorsConfig(
        tensor_name='test_tensors',
        tensor_type='music-genre',
        representation_type='piano-roll',
        train_split=0.6,
        val_split=0.2,
        test_split=0.2,
        output_dir=str(tmp_path),
    )


class TestSaveTensors:

    def test_save_creates_file(self, sample_tf_datasets, tmp_path):
        """Test that save_tensors creates an H5 file."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            filepath = save_tensors(sample_tf_datasets, 'test_save')
            assert filepath.exists()
            assert filepath.suffix == '.h5'

    def test_save_contains_all_splits(self, sample_tf_datasets, tmp_path):
        """Test that saved file contains train, validation, test splits."""
        import h5py
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            filepath = save_tensors(sample_tf_datasets, 'test_splits')
            with h5py.File(filepath, 'r') as f:
                assert 'train' in f.keys()
                assert 'validation' in f.keys()
                assert 'test' in f.keys()

    def test_save_contains_all_keys(self, sample_tf_datasets, tmp_path):
        """Test that each split contains music, genre_id, artist_id, instrument_ids."""
        import h5py
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            filepath = save_tensors(sample_tf_datasets, 'test_keys')
            with h5py.File(filepath, 'r') as f:
                for split in ['train', 'validation', 'test']:
                    assert 'music' in f[split].keys() # type: ignore
                    assert 'genre_id' in f[split].keys() # type: ignore
                    assert 'artist_id' in f[split].keys() # type: ignore
                    assert 'instrument_ids' in f[split].keys() # type: ignore

    def test_save_creates_directory(self, sample_tf_datasets, tmp_path):
        """Test that save_tensors creates the tensors directory if needed."""
        new_dir = tmp_path / 'new_subdir'
        with patch('data.dataset_tensors.TENSORS_DIR', new_dir):
            filepath = save_tensors(sample_tf_datasets, 'test_mkdir')
            assert new_dir.exists()
            assert filepath.exists()


class TestLoadTensors:

    def test_load_returns_tf_datasets(self, sample_tf_datasets, tmp_path):
        """Test that load_tensors returns TensorFlow datasets."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            save_tensors(sample_tf_datasets, 'test_load')
            loaded = load_tensors('test_load')

            assert 'train' in loaded
            assert 'validation' in loaded
            assert 'test' in loaded
            assert isinstance(loaded['train'], tf.data.Dataset)

    def test_load_preserves_data(self, sample_tf_datasets, tmp_path):
        """Test that loaded data matches original data."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            save_tensors(sample_tf_datasets, 'test_preserve')
            loaded = load_tensors('test_preserve')

            # Count samples in each split
            original_counts = {
                name: sum(1 for _ in ds)
                for name, ds in sample_tf_datasets.items()
            }
            loaded_counts = {
                name: sum(1 for _ in ds)
                for name, ds in loaded.items()
            }

            assert original_counts == loaded_counts

    def test_load_preserves_shapes(self, sample_tf_datasets, tmp_path):
        """Test that loaded tensors have correct shapes."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            save_tensors(sample_tf_datasets, 'test_shapes')
            loaded = load_tensors('test_shapes')

            original_sample = next(iter(sample_tf_datasets['train']))
            loaded_sample = next(iter(loaded['train']))

            assert original_sample['music'].shape == loaded_sample['music'].shape # type: ignore
            assert original_sample['instrument_ids'].shape == loaded_sample['instrument_ids'].shape # type: ignore

    def test_load_nonexistent_raises_error(self, tmp_path):
        """Test that loading non-existent file raises FileNotFoundError."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            with pytest.raises(FileNotFoundError):
                load_tensors('nonexistent')

    def test_load_values_match(self, sample_tf_datasets, tmp_path):
        """Test that actual tensor values are preserved."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            save_tensors(sample_tf_datasets, 'test_values')
            loaded = load_tensors('test_values')

            # Get all genre_ids from original and loaded
            original_genres = [s['genre_id'].numpy() for s in sample_tf_datasets['train']]
            loaded_genres = [s['genre_id'].numpy() for s in loaded['train']] # type: ignore

            assert original_genres == loaded_genres


class TestTensorsExist:

    def test_exists_returns_true(self, sample_tf_datasets, tmp_path):
        """Test tensors_exist returns True when file exists."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            save_tensors(sample_tf_datasets, 'test_exists')
            assert tensors_exist('test_exists') is True

    def test_exists_returns_false(self, tmp_path):
        """Test tensors_exist returns False when file doesn't exist."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            assert tensors_exist('nonexistent') is False


class TestGenerateTensors:

    def test_generate_returns_tf_datasets(self, sample_config, sample_dataset, tmp_path):
        """Test that generate_tensors returns TensorFlow datasets."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            result = generate_tensors(sample_config, sample_dataset, save=False)

            assert 'train' in result
            assert 'validation' in result
            assert 'test' in result

    def test_generate_with_save(self, sample_config, sample_dataset, tmp_path):
        """Test that generate_tensors saves when save=True."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            generate_tensors(sample_config, sample_dataset, save=True)
            assert tensors_exist(sample_config.tensor_name)

    def test_generate_without_save(self, sample_config, sample_dataset, tmp_path):
        """Test that generate_tensors doesn't save when save=False."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            generate_tensors(sample_config, sample_dataset, save=False)
            assert not tensors_exist(sample_config.tensor_name)

    def test_generate_correct_split_sizes(self, sample_config, sample_dataset, tmp_path):
        """Test that splits have approximately correct sizes."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            result = generate_tensors(sample_config, sample_dataset, save=False)

            train_count = sum(1 for _ in result['train'])
            val_count = sum(1 for _ in result['validation'])
            test_count = sum(1 for _ in result['test'])

            total = train_count + val_count + test_count
            assert total == len(sample_dataset)
            assert train_count == 6  # 60% of 10
            assert val_count == 2    # 20% of 10
            assert test_count == 2   # 20% of 10


class TestGetTensors:

    def test_get_generates_when_not_cached(self, sample_config, sample_dataset, tmp_path):
        """Test that get_tensors generates when cache doesn't exist."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            result = get_tensors(sample_config, sample_dataset)

            assert 'train' in result
            assert tensors_exist(sample_config.tensor_name)

    def test_get_loads_from_cache(self, sample_config, sample_dataset, tmp_path):
        """Test that get_tensors loads from cache when it exists."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            # First call generates
            generate_tensors(sample_config, sample_dataset, save=True)

            # Second call should load (no dataset needed)
            result = get_tensors(sample_config)

            assert 'train' in result

    def test_get_raises_when_no_cache_and_no_dataset(self, sample_config, tmp_path):
        """Test that get_tensors raises error when no cache and no dataset."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            with pytest.raises(ValueError, match="not found"):
                get_tensors(sample_config)


class TestRoundTrip:

    def test_full_roundtrip(self, sample_dataset, tmp_path):
        """Test complete save/load cycle preserves data integrity."""
        with patch('data.dataset_tensors.TENSORS_DIR', tmp_path):
            # Generate original
            original = sample_dataset.to_tensorflow_dataset_with_metadata(
                representation='piano-roll',
                splits=[0.6, 0.2, 0.2],
                random_state=42
            )

            # Save
            save_tensors(original, 'roundtrip_test')

            # Load
            loaded = load_tensors('roundtrip_test')

            # Compare all data
            for split in ['train', 'validation', 'test']:
                orig_list = list(original[split])
                load_list = list(loaded[split])

                assert len(orig_list) == len(load_list)

                for orig, load in zip(orig_list, load_list):
                    np.testing.assert_array_equal(
                        orig['music'].numpy(),
                        load['music'].numpy()
                    )
                    assert orig['genre_id'].numpy() == load['genre_id'].numpy()
                    assert orig['artist_id'].numpy() == load['artist_id'].numpy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
