from pathlib import Path
from typing import List, Optional, Dict, Any
import muspy
import numpy as np
from tqdm import tqdm

from src.data.configs.enchanced_dataset_config import EnhancedDatasetConfig
from src.data.dataset_vocabulary import DatasetVocabulary
from src.data.enhanced_music import EnhancedMusic
from src.data.enhanced_music_dataset import EnhancedMusicDataset


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


def load_genre_mapping(genre_tsv_path: str) -> Dict[str, str]:
    """
    Load genre mapping from TSV file.

    The TSV format is: Artist/SongName\tGenre

    Args:
        genre_tsv_path: Path to the genre.tsv file

    Returns:
        Dictionary mapping "Artist/SongName" to genre
    """
    genre_map = {}

    tsv_path = Path(genre_tsv_path)
    if not tsv_path.exists():
        print(f"Warning: Genre TSV file does not exist: {genre_tsv_path}")
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


def extract_metadata_from_path(
    filepath: Path,
    extract_genre: bool = True,
    extract_artist: bool = True,
    artist_folder_level: int = -2,
    genre_map: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Extract metadata from file path structure and genre mapping.

    Args:
        filepath: Path to the MIDI file
        extract_genre: Whether to extract genre from genre_map
        extract_artist: Whether to extract artist from path
        artist_folder_level: Folder level for artist (negative = from end, -2 = parent folder)
        genre_map: Dictionary mapping "Artist/SongName" to genre

    Returns:
        Dictionary with 'genre' and 'artist' keys
    """
    import re

    metadata = {}
    parts = filepath.parts

    # Extract artist from parent folder (level -2)
    if extract_artist and len(parts) > abs(artist_folder_level):
        metadata['artist'] = parts[artist_folder_level]

    # Extract genre from TSV mapping using Artist/SongName key
    if extract_genre and genre_map:
        # Get artist folder name and song name (without extension)
        artist = parts[-2] if len(parts) > 1 else ""
        song_name = filepath.stem  # filename without extension

        # Strip version suffix like ".1", ".2", ".10" from song name
        # e.g., "Dreadlock Holiday.1" -> "Dreadlock Holiday"
        song_name_base = re.sub(r'\.\d+$', '', song_name)

        # Build the lookup key
        lookup_key = f"{artist}/{song_name_base}"

        if lookup_key in genre_map:
            metadata['genre'] = genre_map[lookup_key]

    return metadata


def get_duration_seconds(music: muspy.Music) -> float:
    """
    Calculate the duration of a Music object in seconds.

    Args:
        music: muspy.Music object

    Returns:
        Duration in seconds
    """
    end_time_ticks = music.get_end_time()

    if not music.tempos:
        # Default tempo is 120 BPM if no tempo is specified
        default_qpm = 120.0
        quarter_notes = end_time_ticks / music.resolution
        return quarter_notes * (60.0 / default_qpm)

    # Use the first tempo for simplicity (could be extended to handle tempo changes)
    qpm = music.tempos[0].qpm
    quarter_notes = end_time_ticks / music.resolution
    return quarter_notes * (60.0 / qpm)


def get_instrument_name_from_program(program: int, vocab: DatasetVocabulary) -> str:
    """
    Get instrument name from MIDI program number using vocabulary.

    Args:
        program: MIDI program number (0-127, or 128 for drums)
        vocab: DatasetVocabulary instance

    Returns:
        Instrument name or "unknown_{program}" if not found
    """
    # Build reverse mapping from program number to name
    program_to_name = {v: k for k, v in vocab.instrument_to_id.items()}
    return program_to_name.get(program, f"unknown_{program}")


def passes_filter(music: muspy.Music, config: EnhancedDatasetConfig) -> tuple[bool, str]:
    """
    Check if a Music object passes the configured filters.

    Args:
        music: muspy.Music object to check
        config: Configuration with filter settings

    Returns:
        Tuple of (passes, reason) - reason is empty if passes, otherwise describes why filtered
    """
    # Check number of tracks
    num_tracks = len(music.tracks)
    if num_tracks < config.min_tracks:
        return False, f"too_few_tracks ({num_tracks} < {config.min_tracks})"
    if num_tracks > config.max_tracks:
        return False, f"too_many_tracks ({num_tracks} > {config.max_tracks})"

    # Check total number of notes
    total_notes = sum(len(track.notes) for track in music.tracks)
    if total_notes < config.min_notes:
        return False, f"too_few_notes ({total_notes} < {config.min_notes})"

    # Check duration in seconds
    if config.min_duration is not None or config.max_duration is not None:
        duration = get_duration_seconds(music)
        if config.min_duration is not None and duration < config.min_duration:
            return False, f"too_short ({duration:.1f}s < {config.min_duration}s)"
        if config.max_duration is not None and duration > config.max_duration:
            return False, f"too_long ({duration:.1f}s > {config.max_duration}s)"

    # Check instruments
    if config.allowed_instruments or config.excluded_instruments:
        vocab = DatasetVocabulary()

        # Get instrument names for all tracks
        track_instrument_names = []
        for track in music.tracks:
            if track.program is not None:
                name = get_instrument_name_from_program(track.program, vocab)
                track_instrument_names.append(name)
            elif track.is_drum:
                track_instrument_names.append('drums')

        # Check allowed instruments - at least one track must have an allowed instrument
        if config.allowed_instruments:
            if not any(inst in config.allowed_instruments for inst in track_instrument_names):
                return False, "no_allowed_instruments"

        # Check excluded instruments - no track can have an excluded instrument
        if config.excluded_instruments:
            if any(inst in config.excluded_instruments for inst in track_instrument_names):
                return False, "has_excluded_instruments"

    return True, ""


def validate_and_map_instruments(music: muspy.Music, vocab: DatasetVocabulary, verbose: bool = False) -> List[str]:
    """
    Validate track instruments against vocabulary and return their names.

    Args:
        music: muspy.Music object
        vocab: DatasetVocabulary instance
        verbose: Whether to print warnings for unknown instruments

    Returns:
        List of instrument names for each track
    """
    instrument_names = []

    for i, track in enumerate(music.tracks):
        if track.is_drum:
            instrument_names.append('drums')
        elif track.program is not None:
            name = get_instrument_name_from_program(track.program, vocab)
            instrument_names.append(name)

            # Warn if instrument is not in vocabulary (unknown program)
            if name.startswith('unknown_') and verbose:
                print(f"  Warning: Track {i} has unknown program number: {track.program}")
        else:
            instrument_names.append('unknown')
            if verbose:
                print(f"  Warning: Track {i} has no program number set")

    return instrument_names


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

    # Load genre mapping from TSV if path is provided
    genre_map = None
    if config.extract_genre_from_path and config.genre_tsv_path:
        if config.verbose:
            print(f"Loading genre mapping from {config.genre_tsv_path}...")
        genre_map = load_genre_mapping(config.genre_tsv_path)
        if config.verbose:
            print(f"Loaded {len(genre_map)} genre mappings")

    # Find all MIDI files
    if config.verbose:
        print(f"Searching for MIDI files in {len(config.input_dirs)} directories...")

    midi_files = find_midi_files(config.input_dirs, config.max_samples)

    if config.verbose:
        print(f"Found {len(midi_files)} MIDI files")

    # Initialize dataset with empty vocabulary
    dataset = EnhancedMusicDataset()
    vocab = DatasetVocabulary()

    # Process each MIDI file
    successful = 0
    failed = 0
    filtered_out = 0
    no_genre_found = 0
    filter_reasons: Dict[str, int] = {}  # Track why files were filtered

    iterator = tqdm(midi_files, desc="Processing MIDI files") if config.verbose else midi_files

    for midi_path in iterator:
        try:
            # Read MIDI file
            music = muspy.read_midi(str(midi_path))

            # Apply filters
            passes, reason = passes_filter(music, config)
            if not passes:
                filtered_out += 1
                # Track filter reason (use base reason without specific values)
                base_reason = reason.split(' ')[0]
                filter_reasons[base_reason] = filter_reasons.get(base_reason, 0) + 1
                continue

            # Preprocess
            music = preprocess_music(music, config)

            # Extract metadata
            metadata = extract_metadata_from_path(
                midi_path,
                config.extract_genre_from_path,
                config.extract_artist_from_path,
                config.artist_folder_level,
                genre_map
            )

            # Track if genre wasn't found
            if config.extract_genre_from_path and 'genre' not in metadata:
                no_genre_found += 1

            # Validate and map instruments
            instrument_names = validate_and_map_instruments(music, vocab, verbose=False)
            metadata['instruments'] = instrument_names # type: ignore

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
        if filter_reasons:
            print(f"  Filter breakdown:")
            for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count}")
        print(f"  Failed: {failed}")
        if config.extract_genre_from_path:
            print(f"  No genre found in TSV: {no_genre_found}")
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
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent.parent.parent

    # Create a configuration
    config = EnhancedDatasetConfig(
        dataset_name="test_dataset",
        input_dirs=[str(project_root / "data" / "core_datasets" / "clean_midi")],
        output_path=str(project_root / "data" / "datasets" / "test_dataset.h5"),
        genre_tsv_path=str(project_root / "data" / "core_datasets" / "clean_midi" / "genre.tsv"),
        max_samples=100,  # Limit for testing
        verbose=True
    )

    # Create and save the dataset
    dataset = create_and_save_dataset(config)

    print(f"\nDataset ready with {len(dataset)} samples")
