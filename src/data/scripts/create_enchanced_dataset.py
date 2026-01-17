from pathlib import Path
from typing import List, Optional, Dict
import muspy
import numpy as np
from tqdm import tqdm

from data.enhanced_music import EnhancedMusic
from data.enhanced_music_dataset import EnhancedMusicDataset
from data.dataset_vocabulary import DatasetVocabulary
from data.configs.enhanced_dataset_config import EnhancedDatasetConfig


def find_midi_files(input_dirs: List[str], max_samples: Optional[int] = None) -> List[Path]:
    """Find all MIDI files in the input directories."""
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


def load_genre_mapping(tsv_path: str) -> Dict[str, str]:
    """
    Load genre mapping from TSV file.
    Format: Artist/SongName\tGenre
    
    Returns:
        Dictionary mapping "Artist/SongName" to genre
    """
    genre_map = {}
    
    if not Path(tsv_path).exists():
        print(f"Warning: Genre TSV not found: {tsv_path}")
        return genre_map
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                key = parts[0]  # "Artist/SongName"
                genre = parts[1]
                genre_map[key] = genre
    
    return genre_map


def extract_metadata(filepath: Path, genre_map: Dict[str, str]) -> Dict[str, str]:
    """
    Extract metadata from file path and genre mapping.
    
    Args:
        filepath: Path to MIDI file (e.g., .../Artist/song.mid)
        genre_map: Genre mapping dictionary
    
    Returns:
        Dictionary with 'artist' and optionally 'genre'
    """
    metadata = {}
    
    # Artist is the parent folder name
    metadata['artist'] = filepath.parent.name
    
    # Genre from TSV using "Artist/SongName" format
    song_name = filepath.stem  # filename without extension
    lookup_key = f"{metadata['artist']}/{song_name}"
    
    if lookup_key in genre_map:
        metadata['genre'] = genre_map[lookup_key]
    
    return metadata


def passes_filter(music: muspy.Music, config: EnhancedDatasetConfig) -> bool:
    """Check if music passes the configured filters."""
    
    # Check number of tracks
    num_tracks = len(music.tracks)
    if num_tracks < config.min_tracks or num_tracks > config.max_tracks:
        return False
    
    # Check total notes
    total_notes = sum(len(track.notes) for track in music.tracks)
    if total_notes < config.min_notes:
        return False
    
    # Check duration
    if config.min_duration or config.max_duration:
        # Calculate duration in seconds
        end_time = music.get_end_time()
        tempo = music.tempos[0].qpm if music.tempos else 120.0
        duration = (end_time / music.resolution) * (60.0 / tempo)
        
        if config.min_duration and duration < config.min_duration:
            return False
        if config.max_duration and duration > config.max_duration:
            return False
    
    # Check instruments (if filters are set)
    if config.allowed_instruments or config.excluded_instruments:
        vocab = DatasetVocabulary()
        
        # Get instrument names from program numbers
        track_instruments = []
        for track in music.tracks:
            if track.is_drum:
                track_instruments.append('drums')
            elif track.program is not None:
                # Convert program number to instrument name
                program_to_name = {v: k for k, v in vocab.instrument_to_id.items()}
                inst_name = program_to_name.get(track.program, f"program_{track.program}")
                track_instruments.append(inst_name)
        
        # Check allowed instruments
        if config.allowed_instruments:
            if not any(inst in config.allowed_instruments for inst in track_instruments):
                return False
        
        # Check excluded instruments
        if config.excluded_instruments:
            if any(inst in config.excluded_instruments for inst in track_instruments):
                return False
    
    return True


def preprocess_music(music: muspy.Music, config: EnhancedDatasetConfig) -> muspy.Music:
    """Preprocess music according to config."""
    
    # Adjust resolution
    if music.resolution != config.resolution:
        music = music.adjust_resolution(config.resolution)
    
    # Quantize notes
    if config.quantize:
        for track in music.tracks:
            for note in track.notes:
                note.time = round(note.time / config.resolution) * config.resolution
                note.duration = max(1, round(note.duration / config.resolution)) * config.resolution
    
    # Remove empty tracks
    if config.remove_empty_tracks:
        music.tracks = [track for track in music.tracks if len(track.notes) > 0]
    
    return music


def create_enhanced_dataset(config: EnhancedDatasetConfig) -> EnhancedMusicDataset:
    """
    Create an EnhancedMusicDataset from MIDI files.
    
    Process:
    1. Find all MIDI files
    2. Load genre mapping from TSV
    3. For each MIDI:
       - Read file
       - Apply filters
       - Preprocess
       - Extract metadata (artist from folder, genre from TSV)
       - Create EnhancedMusic
       - Add to dataset
    """
    
    # Set random seed
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    
    # Load genre mapping
    genre_map = {}
    if config.genre_tsv_path:
        if config.verbose:
            print(f"Loading genre mapping from {config.genre_tsv_path}...")
        genre_map = load_genre_mapping(config.genre_tsv_path)
        if config.verbose:
            print(f"Loaded {len(genre_map)} genre mappings")
    
    # Find MIDI files
    if config.verbose:
        print(f"Searching for MIDI files...")
    
    midi_files = find_midi_files(config.input_dirs, config.max_samples)
    
    if config.verbose:
        print(f"Found {len(midi_files)} MIDI files")
    
    # Initialize dataset
    dataset = EnhancedMusicDataset()
    
    # Counters
    successful = 0
    filtered_out = 0
    failed = 0
    
    # Process each MIDI file
    iterator = tqdm(midi_files, desc="Processing") if config.verbose else midi_files
    
    for midi_path in iterator:
        try:
            # Read MIDI
            music = muspy.read_midi(str(midi_path))
            
            # Apply filters
            if not passes_filter(music, config):
                filtered_out += 1
                continue
            
            # Preprocess
            music = preprocess_music(music, config)
            
            # Extract metadata
            metadata = extract_metadata(midi_path, genre_map)
            
            # Create EnhancedMusic and add to dataset
            enhanced_music = EnhancedMusic(music=music, metadata=metadata)
            dataset.append(enhanced_music)
            
            successful += 1
            
        except Exception as e:
            failed += 1
            if config.verbose:
                print(f"\nError processing {midi_path.name}: {e}")
    
    # Print summary
    if config.verbose:
        print(f"\n{'='*50}")
        print(f"Dataset Creation Summary:")
        print(f"  Successfully processed: {successful}")
        print(f"  Filtered out: {filtered_out}")
        print(f"  Failed: {failed}")
        print(f"  Total in dataset: {len(dataset)}")
        print(f"  Unique genres: {dataset.vocabulary.num_genres}")
        print(f"  Unique artists: {dataset.vocabulary.num_artists}")
        print(f"{'='*50}")
    
    return dataset


def create_and_save_dataset(config: EnhancedDatasetConfig) -> EnhancedMusicDataset:
    """Create dataset and save to disk."""
    
    # Create dataset
    dataset = create_enhanced_dataset(config)
    
    # Save dataset
    if config.verbose:
        print(f"\nSaving dataset to {config.output_path}...")
    
    dataset.save(config.output_path)
    
    # Save config
    config_path = Path(config.output_path).with_suffix('.config.json')
    config.save(str(config_path))
    
    if config.verbose:
        print(f"✓ Dataset saved to {config.output_path}")
        print(f"✓ Config saved to {config_path}")
    
    return dataset


if __name__ == "__main__":
    # Example usage
    config = EnhancedDatasetConfig(
        dataset_name="my_dataset",
        input_dirs=["./data/core_datasets/clean_midi"],
        output_path="./data/datasets/my_dataset.h5",
        genre_tsv_path="./data/core_datasets/clean_midi/genre.tsv",
        max_samples=100,
        verbose=True
    )
    
    dataset = create_and_save_dataset(config)
    print(f"\nDataset ready with {len(dataset)} samples!")