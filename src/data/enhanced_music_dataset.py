from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import tensorflow as tf
import h5py
import json
import muspy
from pathlib import Path

from data.enhanced_music import EnhancedMusic
from data.dataset_vocabulary import DatasetVocabulary

#TODO: Add the config this was created with as well

@dataclass
class EnhancedMusicDataset:
    """Simple dataset wrapper for EnhancedMusic objects with vocabulary support."""
    
    data: List[EnhancedMusic] = field(default_factory=list)
    vocabulary: DatasetVocabulary = field(default_factory=DatasetVocabulary)
    

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> EnhancedMusic:
        return self.data[index]
    
    def append(self, item: EnhancedMusic) -> None:
        """Add an item and update vocabulary."""
        self.data.append(item)
        if 'genre' in item.metadata and item.metadata['genre']:
            self.vocabulary.add_genre(item.metadata['genre'])
        if 'artist' in item.metadata and item.metadata['artist']:
            self.vocabulary.add_artist(item.metadata['artist'])
    
    def extend(self, items: List[EnhancedMusic]) -> None:
        """Add multiple items and update vocabulary."""
        for item in items:
            self.append(item)
    
    def build_vocabulary(self) -> None:
        """
        Build vocabulary from all items in the dataset.
        
        Iterates through all EnhancedMusic objects and adds unique genres
        and artists to the vocabulary. Instruments are already predefined
        in the vocabulary based on General MIDI standard.
        """
        # Clear existing genre and artist vocabularies
        self.vocabulary.genre_to_id.clear()
        self.vocabulary.artist_to_id.clear()
        
        # Loop through all items and build vocabulary
        for item in self.data:
            # Add genre if present in metadata
            if 'genre' in item.metadata and item.metadata['genre']:
                self.vocabulary.add_genre(item.metadata['genre'])
            
            # Add artist if present in metadata
            if 'artist' in item.metadata and item.metadata['artist']:
                self.vocabulary.add_artist(item.metadata['artist'])

    def to_tensorflow_dataset(
        self,
        representation: str = "pianoroll",
        split_filename=None,
        splits=None,
        random_state=None,
        **kwargs
    ):
        """
        Convert dataset to TensorFlow dataset with music data only (no metadata).
        
        Each sample contains:
            'music': float32 array of the musical representation
        """
        def generator(indices):
            for idx in indices:
                item: EnhancedMusic = self.data[idx]

                # Check if there are any notes in the music
                has_notes = any(len(track.notes) > 0 for track in item.music.tracks)
                if has_notes:
                    music_data = item.music.to_representation(representation, **kwargs)
                    # Transpose to (128, time_steps) so first dimension is MIDI pitches
                    music_data = music_data.T
                else:
                    # Return empty array with shape (128, 0) for music with no notes
                    music_data = np.zeros((128, 0), dtype=np.float32)

                yield {
                    'music': music_data.astype(np.float32),
                }
        
        # Get output signature
        sample = next(generator([0]))
        output_signature = {
            'music': tf.TensorSpec(shape=sample['music'].shape, dtype=tf.float32), # type: ignore
        }
        
        return self._create_datasets(generator, output_signature, splits, split_filename, random_state)
    
    def to_tensorflow_dataset_with_genre(
        self,
        representation: str = "pianoroll",
        split_filename=None,
        splits=None,
        random_state=None,
        **kwargs
    ):
        """
        Convert dataset to TensorFlow dataset with music and genre.
        
        Each sample contains:
            'music': float32 array of the musical representation
            'genre_id': int32 genre vocabulary ID
        """
        def generator(indices):
            for idx in indices:
                item: EnhancedMusic = self.data[idx]

                # Check if there are any notes in the music
                has_notes = any(len(track.notes) > 0 for track in item.music.tracks)
                if has_notes:
                    music_data = item.music.to_representation(representation, **kwargs)
                    # Transpose to (128, time_steps) so first dimension is MIDI pitches
                    music_data = music_data.T
                else:
                    # Return empty array with shape (128, 0) for music with no notes
                    music_data = np.zeros((128, 0), dtype=np.float32)

                yield {
                    'music': music_data.astype(np.float32),
                    'genre_id': np.int32(self.vocabulary.get_genre_id(item.metadata.get('genre', ''))),
                }
        
        # Get output signature
        sample = next(generator([0]))
        output_signature = {
            'music': tf.TensorSpec(shape=sample['music'].shape, dtype=tf.float32), # type: ignore
            'genre_id': tf.TensorSpec(shape=(), dtype=tf.int32), # type: ignore
        }
        
        return self._create_datasets(generator, output_signature, splits, split_filename, random_state)
    
    def to_tensorflow_dataset_with_instruments(
        self,
        representation: str = "pianoroll",
        split_filename=None,
        splits=None,
        random_state=None,
        max_tracks: int = 16,
        **kwargs
    ):
        """
        Convert dataset to TensorFlow dataset with music and instruments.
        
        Each sample contains:
            'music': float32 array of the musical representation
            'instrument_ids': int32 array[max_tracks] of MIDI program numbers
        """
        def generator(indices):
            for idx in indices:
                item: EnhancedMusic = self.data[idx]

                # Check if there are any notes in the music
                has_notes = any(len(track.notes) > 0 for track in item.music.tracks)
                if has_notes:
                    music_data = item.music.to_representation(representation, **kwargs)
                    # Transpose to (128, time_steps) so first dimension is MIDI pitches
                    music_data = music_data.T
                else:
                    # Return empty array with shape (128, 0) for music with no notes
                    music_data = np.zeros((128, 0), dtype=np.float32)

                instrument_ids = [track.program for track in item.music.tracks]
                if len(instrument_ids) < max_tracks:
                    instrument_ids.extend([-1] * (max_tracks - len(instrument_ids)))
                else:
                    instrument_ids = instrument_ids[:max_tracks]

                yield {
                    'music': music_data.astype(np.float32),
                    'instrument_ids': np.array(instrument_ids, dtype=np.int32),
                }
        
        # Get output signature
        sample = next(generator([0]))
        output_signature = {
            'music': tf.TensorSpec(shape=sample['music'].shape, dtype=tf.float32), # type: ignore
            'instrument_ids': tf.TensorSpec(shape=(max_tracks,), dtype=tf.int32), # type: ignore
        }
        
        return self._create_datasets(generator, output_signature, splits, split_filename, random_state)
    
    def _create_datasets(self, generator, output_signature, splits, split_filename, random_state):
        """
        Helper method to create TensorFlow datasets with optional train/val/test splits.
        
        Args:
            generator: Generator function that takes indices
            output_signature: TensorFlow output signature
            splits: Tuple of (train, val, test) proportions or None
            split_filename: Path to save split indices
            random_state: Random seed for reproducible splits
            
        Returns:
            Single dataset if splits is None, otherwise dict with 'train', 'validation', 'test'
        """
        # No splits - return single dataset
        if splits is None:
            return tf.data.Dataset.from_generator(
                lambda: generator(range(len(self))),
                output_signature=output_signature
            )
        
        # Handle splits with random_state
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        
        train_end = int(len(self) * splits[0])
        val_end = train_end + int(len(self) * splits[1])
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # Save splits if filename provided
        if split_filename is not None:
            import json
            from pathlib import Path
            Path(split_filename).parent.mkdir(parents=True, exist_ok=True)
            with open(split_filename, 'w') as f:
                json.dump({
                    'train': train_idx.tolist(),
                    'validation': val_idx.tolist(),
                    'test': test_idx.tolist()
                }, f)
        
        return {
            'train': tf.data.Dataset.from_generator(
                lambda: generator(train_idx),
                output_signature=output_signature
            ),
            'validation': tf.data.Dataset.from_generator(
                lambda: generator(val_idx),
                output_signature=output_signature
            ),
            'test': tf.data.Dataset.from_generator(
                lambda: generator(test_idx),
                output_signature=output_signature
            ),
        }
    
    def to_tensorflow_dataset_with_metadata(
        self,
        representation: str = "pianoroll",
        split_filename=None,
        splits=None,
        random_state=None,
        max_tracks: int = 16,
        **kwargs
    ):
        """
        Convert dataset to TensorFlow dataset with full metadata (genre, artist, instruments).
        
        Each sample contains:
            'music': float32 array of the musical representation
            'genre_id': int32 genre vocabulary ID
            'artist_id': int32 artist vocabulary ID  
            'instrument_ids': int32 array[max_tracks] of MIDI program numbers
        """
        def generator(indices):
            for idx in indices:
                item: EnhancedMusic = self.data[idx]

                # Check if there are any notes in the music
                has_notes = any(len(track.notes) > 0 for track in item.music.tracks)
                if has_notes:
                    music_data = item.music.to_representation(representation, **kwargs)
                    # Transpose to (128, time_steps) so first dimension is MIDI pitches
                    music_data = music_data.T
                else:
                    # Return empty array with shape (128, 0) for music with no notes
                    music_data = np.zeros((128, 0), dtype=np.float32)

                instrument_ids = [track.program for track in item.music.tracks]
                if len(instrument_ids) < max_tracks:
                    instrument_ids.extend([-1] * (max_tracks - len(instrument_ids)))
                else:
                    instrument_ids = instrument_ids[:max_tracks]

                yield {
                    'music': music_data.astype(np.float32),
                    'genre_id': np.int32(self.vocabulary.get_genre_id(item.metadata.get('genre', ''))),
                    'artist_id': np.int32(self.vocabulary.get_artist_id(item.metadata.get('artist', ''))),
                    'instrument_ids': np.array(instrument_ids, dtype=np.int32),
                }
        
        # Get output signature
        sample = next(generator([0]))
        output_signature = {
            'music': tf.TensorSpec(shape=sample['music'].shape, dtype=tf.float32), # type: ignore
            'genre_id': tf.TensorSpec(shape=(), dtype=tf.int32), # type: ignore
            'artist_id': tf.TensorSpec(shape=(), dtype=tf.int32), # type: ignore 
            'instrument_ids': tf.TensorSpec(shape=(max_tracks,), dtype=tf.int32), # type: ignore
        } 
        
        return self._create_datasets(generator, output_signature, splits, split_filename, random_state)
    def save(self, filepath: str) -> None:
        """
        Save dataset to HDF5 file.

        Args:
            filepath: Path to save the .h5 file
        """
        import pickle

        filepath = Path(filepath) # type: ignore
        filepath.parent.mkdir(parents=True, exist_ok=True) # type: ignore

        with h5py.File(filepath, 'w') as f:
            # Save vocabulary
            vocab_group = f.create_group('vocabulary')
            vocab_group.attrs['genre_to_id'] = json.dumps(self.vocabulary.genre_to_id)
            vocab_group.attrs['artist_to_id'] = json.dumps(self.vocabulary.artist_to_id)

            # Save each EnhancedMusic item
            data_group = f.create_group('data')
            for idx, item in enumerate(self.data):
                item_group = data_group.create_group(str(idx))

                # Save the Music object as bytes (pickled) - use dataset instead of attr for large data
                pickled_data = np.frombuffer(pickle.dumps(item.music), dtype=np.uint8)
                item_group.create_dataset('music_pickle', data=pickled_data)

                # Save metadata as attribute (small enough)
                item_group.attrs['metadata'] = json.dumps(item.metadata)

    @staticmethod
    def load(filepath: str) -> 'EnhancedMusicDataset':
        """
        Load dataset from HDF5 file.

        Args:
            filepath: Path to the .h5 file

        Returns:
            EnhancedMusicDataset instance
        """
        import pickle

        with h5py.File(filepath, 'r') as f:
            # Load vocabulary
            vocab = DatasetVocabulary()
            vocab.genre_to_id = json.loads(f['vocabulary'].attrs['genre_to_id']) # type: ignore
            vocab.artist_to_id = json.loads(f['vocabulary'].attrs['artist_to_id']) # type: ignore

            # Load data
            data = []
            data_group = f['data']
            for idx in sorted(data_group.keys(), key=int): # type: ignore
                item_group = data_group[idx] # type: ignore

                # Unpickle Music object from dataset
                music = pickle.loads(item_group['music_pickle'][:].tobytes()) # type: ignore

                # Load metadata
                metadata = json.loads(item_group.attrs['metadata']) # type: ignore

                data.append(EnhancedMusic(music=music, metadata=metadata))

        return EnhancedMusicDataset(data=data, vocabulary=vocab)