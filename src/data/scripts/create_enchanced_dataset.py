from pathlib import Path
from typing import List, Optional, Dict, Any
import muspy
import numpy as np
from tqdm import tqdm

from data.enhanced_music import EnhancedMusic, midi_to_enchanced_music
from data.enhanced_music_dataset import EnhancedMusicDataset
from data.dataset_vocabulary import DatasetVocabulary
from data.configs.enhanced_dataset_config import EnhancedDatasetConfig


def find_midi_files(input_dirs: List[str], max_samples: Optional[int] = None) -> List[Path]:
    """
    Find all MIDI files in the input directories.
    """
    midi_files = []
    for input_dir in input_dirs:
        dir_path = Path(input_dir)
        if not dir_path.exists():
            print(f"Warning: Directory does not exist: {input_dir}")
            continue
        for ext in ['*.mid', '*.midi']:
            midi_files.extend(dir_path.rglob(ext))
        if max_samples and len(midi_files) >= max_samples:
            break
    if max_samples:
        midi_files = midi_files[:max_samples]
    return midi_files


def extract_metadata_from_path(
    filepath: Path,
    extract_genre: bool = True,
    extract_artist: bool = True,
    genre_level: int = -2,
    artist_level: int = -1
) -> Dict[str, str]:
    metadata = {}
    parts = filepath.parts
    if extract_genre and len(parts) > abs(genre_level):
        metadata['genre'] = parts[genre_level]
    if extract_artist and len(parts) > abs(artist_level):
        metadata['artist'] = parts[artist_level]
    return metadata


def passes_filter(music: muspy.Music, config: EnhancedDatasetConfig) -> bool:
    num_tracks = len(music.tracks)
    if num_tracks < config.min_tracks or num_tracks > config.max_tracks:
        return False
    total_notes = sum(len(track.notes) for track in music.tracks)
    if total_notes < config.min_notes:
        return False
    if config.min_duration is not None or config.max_duration is not None:
        duration = music.get_end_time() / music.resolution
        if config.min_duration is not None and duration < config.min_duration:
            return False
        if config.max_duration is not None and duration > config.max_duration:
            return False
    if config.allowed_instruments or config.excluded_instruments:
        vocab = DatasetVocabulary()
        track_instruments = [track.program for track in music.tracks if track.program is not None]
        program_to_name = {v: k for k, v in vocab.instrument_to_id.items()}
        track_instrument_names = [program_to_name.get(prog, f"program_{prog}") for prog in track_instruments]
        if config.allowed_instruments:
            if not any(inst in config.allowed_instruments for inst in track_instrument_names):
                return False
        if config.excluded_instruments:
            if any(inst in config.excluded_instruments for inst in track_instrument_names):
                return False
    return True


def preprocess_music(music: muspy.Music, config: EnhancedDatasetConfig) -> muspy.Music:
    if music.resolution != config.resolution:
        music = music.adjust_resolution(config.resolution)
    if config.quantize:
        for track in music.tracks:
            if track.notes:
                for note in track.notes:
                    note.time = round(note.time / config.resolution) * config.resolution
                    note.duration = max(1, round(note.duration / config.resolution)) * config.resolution
    if config.remove_empty_tracks:
        music.tracks = [track for track in music.tracks if len(track.notes) > 0]
    return music


def create_enhanced_dataset(config: EnhancedDatasetConfig) -> EnhancedMusicDataset:
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    if config.verbose:
        print(f"Searching for MIDI files in {len(config.input_dirs)} directories...")
    midi_files = find_midi_files(config.input_dirs, config.max_samples)
    if config.verbose:
        print(f"Found {len(midi_files)} MIDI files")
    dataset = EnhancedMusicDataset()
    successful = 0
    failed = 0
    filtered_out = 0
    iterator = tqdm(midi_files, desc="Processing MIDI files") if config.verbose else midi_files
    for midi_path in iterator:
        try:
            music = muspy.read_midi(str(midi_path))
            if not passes_filter(music, config):
                filtered_out += 1
                continue
            music = preprocess_music(music, config)
            metadata = extract_metadata_from_path(
                midi_path,
                config.extract_genre_from_path,
                config.extract_artist_from_path,
                config.genre_folder_level,
                config.artist_folder_level
            )
            vocab = DatasetVocabulary()
            for track in music.tracks:
                if track.program is not None:
                    if vocab.get_instrument_id(str(track.program)) == -1:
                        pass
            enhanced_music = EnhancedMusic(music=music, metadata=metadata)
            dataset.append(enhanced_music)
            successful += 1
        except Exception as e:
            failed += 1
            if config.verbose:
                print(f"\nError processing {midi_path}: {e}")
    if config.verbose:
        print(f"\nDataset creation complete:")
        print(f"  Successfully processed: {successful}")
        print(f"  Filtered out: {filtered_out}")
        print(f"  Failed: {failed}")
        print(f"  Total in dataset: {len(dataset)}")
        print(f"  Genres in vocabulary: {dataset.vocabulary.num_genres}")
        print(f"  Artists in vocabulary: {dataset.vocabulary.num_artists}")
    return dataset


def create_and_save_dataset(config: EnhancedDatasetConfig) -> EnhancedMusicDataset:
    dataset = create_enhanced_dataset(config)
    if config.verbose:
        print(f"\nSaving dataset to {config.output_path}...")
    dataset.save(config.output_path)
    if config.verbose:
        print(f"Dataset saved successfully!")
    config_path = Path(config.output_path).with_suffix('.config.json')
    config.save(str(config_path))
    if config.verbose:
        print(f"Configuration saved to {config_path}")
    return dataset


def find_midi_files(input_dirs: List[str], max_samples: Optional[int] = None) -> List[Path]:
    """
    Find all MIDI files in the input directories.
    
    Args:
        input_dirs: List of directories to search
        max_samples: Optional limit on number of files to return
        
    Returns:
        List of Path objects to MIDI files
    """
    midi_files = []
    
    for input_dir in input_dirs:
        dir_path = Path(input_dir)
        if not dir_path.exists():
            print(f"Warning: Directory does not exist: {input_dir}")
            continue
        
        # Find all .mid and .midi files recursively
        for ext in ['*.mid', '*.midi']:
            midi_files.extend(dir_path.rglob(ext))
        
        if max_samples and len(midi_files) >= max_samples:
            break
    
    if max_samples:
        midi_files = midi_files[:max_samples]
    
    return midi_files


def extract_metadata_from_path(
    filepath: Path,
    extract_genre: bool = True,
    extract_artist: bool = True,
    genre_level: int = -2,
    artist_level: int = -1
) -> Dict[str, str]:
    """
    Extract metadata from file path structure.
    
    Args:
        filepath: Path to the MIDI file
        extract_genre: Whether to extract genre from path
        extract_artist: Whether to extract artist from path
        genre_level: Folder level for genre (negative = from end)
        artist_level: Folder level for artist (negative = from end)
        
    Returns:
        Dictionary with 'genre' and 'artist' keys
    """
    metadata = {}
    parts = filepath.parts
    
    if extract_genre and len(parts) > abs(genre_level):
        metadata['genre'] = parts[genre_level]
    
    if extract_artist and len(parts) > abs(artist_level):
        metadata['artist'] = parts[artist_level]
    
    return metadata


def passes_filter(music: muspy.Music, config: EnhancedDatasetConfig) -> bool:
    """
    Check if a Music object passes the configured filters.
    
    Args:
        music: muspy.Music object to check
        config: Configuration with filter settings
        
    Returns:
        True if music passes all filters, False otherwise
    """
    # Check number of tracks
    num_tracks = len(music.tracks)
    if num_tracks < config.min_tracks or num_tracks > config.max_tracks:
        return False
    
    # Check total number of notes
    total_notes = sum(len(track.notes) for track in music.tracks)
    if total_notes < config.min_notes:
        return False
    
    # Check duration
    if config.min_duration is not None or config.max_duration is not None:
        duration = music.get_end_time() / music.resolution  # Convert to seconds
        if config.min_duration is not None and duration < config.min_duration:
            return False
        if config.max_duration is not None and duration > config.max_duration:
            return False
    
    # Check instruments
    if config.allowed_instruments or config.excluded_instruments:
        vocab = DatasetVocabulary()
        track_instruments = [track.program for track in music.tracks if track.program is not None]
        
        # Get instrument names from program numbers
        program_to_name = {v: k for k, v in vocab.instrument_to_id.items()}
        track_instrument_names = [
            program_to_name.get(prog, f"program_{prog}") 
            for prog in track_instruments
        ]
        
        # Check allowed instruments
        if config.allowed_instruments:
            if not any(inst in config.allowed_instruments for inst in track_instrument_names):
                return False
        
        # Check excluded instruments
        if config.excluded_instruments:
            if any(inst in config.excluded_instruments for inst in track_instrument_names):
                return False
    
    return True


def preprocess_music(music: muspy.Music, config: EnhancedDatasetConfig) -> muspy.Music:
    """
    Preprocess a Music object according to configuration.
    
    Args:
        music: muspy.Music object to preprocess
        config: Configuration with preprocessing settings
        
    Returns:
        Preprocessed muspy.Music object
    """
    # Set resolution
    if music.resolution != config.resolution:
        music = music.adjust_resolution(config.resolution)
    
    # Quantize notes
    if config.quantize:
        for track in music.tracks:
            if track.notes:
                for note in track.notes:
                    note.time = round(note.time / config.resolution) * config.resolution
                    note.duration = max(1, round(note.duration / config.resolution)) * config.resolution
    
    # Remove empty tracks
    if config.remove_empty_tracks:
        music.tracks = [track for track in music.tracks if len(track.notes) > 0]
    
    return music


def create_enhanced_dataset(config: EnhancedDatasetConfig) -> EnhancedMusicDataset:
    """
    Create an EnhancedMusicDataset from MIDI files according to configuration.
    
    Args:
        config: Configuration for dataset creation
        
    Returns:
        EnhancedMusicDataset with filtered and preprocessed music
    """
    # Set random seed if provided
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    
    # Find all MIDI files
    if config.verbose:
        print(f"Searching for MIDI files in {len(config.input_dirs)} directories...")
    
    midi_files = find_midi_files(config.input_dirs, config.max_samples)
    
    if config.verbose:
        print(f"Found {len(midi_files)} MIDI files")
    
    # Initialize dataset with empty vocabulary
    dataset = EnhancedMusicDataset()
    
    # Process each MIDI file
    successful = 0
    failed = 0
    filtered_out = 0
    
    iterator = tqdm(midi_files, desc="Processing MIDI files") if config.verbose else midi_files
    
    for midi_path in iterator:
        try:
            # Read MIDI file
            music = muspy.read_midi(str(midi_path))
            
            # Apply filters
            if not passes_filter(music, config):
                filtered_out += 1
                continue
            
            # Preprocess
            music = preprocess_music(music, config)
            
            # Extract metadata
            metadata = extract_metadata_from_path(
                midi_path,
                config.extract_genre_from_path,
                config.extract_artist_from_path,
                config.genre_folder_level,
                config.artist_folder_level
            )
            
            # Map instruments to vocabulary IDs
            vocab = DatasetVocabulary()
            for track in music.tracks:
                if track.program is not None:
                    # Verify instrument is in vocabulary
                    if vocab.get_instrument_id(str(track.program)) == -1:
                        # If program number, keep as is; vocabulary will handle it
                        pass
            
            # Create EnhancedMusic object
            enhanced_music = EnhancedMusic(music=music, metadata=metadata)
            
            # Add to dataset (this also updates vocabulary)
            dataset.append(enhanced_music)
            successful += 1
            
        except Exception as e:
            failed += 1
            if config.verbose:
                print(f"\nError processing {midi_path}: {e}")
    
    # Print summary
    if config.verbose:
        print(f"\nDataset creation complete:")
        print(f"  Successfully processed: {successful}")
        print(f"  Filtered out: {filtered_out}")
        print(f"  Failed: {failed}")
        print(f"  Total in dataset: {len(dataset)}")
        print(f"  Genres in vocabulary: {dataset.vocabulary.num_genres}")
        print(f"  Artists in vocabulary: {dataset.vocabulary.num_artists}")
    
    return dataset


def create_and_save_dataset(config: EnhancedDatasetConfig) -> EnhancedMusicDataset:
    """
    Create an EnhancedMusicDataset and save it to disk.
    
    Args:
        config: Configuration for dataset creation
        
    Returns:
        The created EnhancedMusicDataset
    """
    # Create dataset
    dataset = create_enhanced_dataset(config)
    
    # Save dataset
    if config.verbose:
        print(f"\nSaving dataset to {config.output_path}...")
    
    dataset.save(config.output_path)
    
    if config.verbose:
        print(f"Dataset saved successfully!")
    
    # Save config alongside dataset
    config_path = Path(config.output_path).with_suffix('.config.json')
    config.save(str(config_path))
    
    if config.verbose:
        print(f"Configuration saved to {config_path}")
    
    return dataset


# Example usage
if __name__ == "__main__":
    # Create a configuration
    config = EnhancedDatasetConfig(
        dataset_name="test_dataset",
        input_dirs=["./data/midi/"],
        output_path="./data/datasets/test_dataset.h5",
        max_samples=100,  # Limit for testing
        verbose=True
    )
    
    # Create and save the dataset
    dataset = create_and_save_dataset(config)
    
    print(f"\nDataset ready with {len(dataset)} samples")
    
    
    
    
    
    