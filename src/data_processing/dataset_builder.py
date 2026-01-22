"""
Dataset builder for creating MusicDataset from MIDI files.

Handles:
- Loading genre mappings from TSV
- Finding MIDI files in directories
- Filtering by genre, track count, duration
- Preprocessing (resolution, quantization, segmentation, augmentation)
- Building MusicDataset with filtered results
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import muspy # type: ignore
from tqdm import tqdm # type: ignore

from .configs import MusicDatasetConfig, PreprocessingConfig
from .preprocessing import preprocess_music
from ..core import MusicDataset


def load_genre_map(tsv_path: str) -> Dict[str, str]:
    """
    Load genre mapping from TSV file.

    TSV format: Artist/SongName<TAB>Genre

    Args:
        tsv_path: Path to genre.tsv file

    Returns:
        Dict mapping "Artist/SongName" to genre string
    """
    genre_map = {}
    path = Path(tsv_path)

    if not path.exists():
        print(f"Warning: Genre TSV not found: {tsv_path}")
        return genre_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                genre_map[parts[0]] = parts[1]

    return genre_map


def find_midi_files(input_dirs: List[str], max_files: Optional[int] = None) -> List[Path]:
    """
    Find all MIDI files in directories recursively.

    Args:
        input_dirs: List of directories to search
        max_files: Optional limit on files to return

    Returns:
        List of Path objects to .mid/.midi files
    """
    import os

    midi_files: set[Path] = set()
    extensions = {".mid", ".midi"}

    for input_dir in input_dirs:
        dir_path = Path(input_dir)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {input_dir}")
            continue

        # Use os.walk for better Windows compatibility with problematic paths
        for root, dirs, files in os.walk(str(dir_path), onerror=lambda e: None):
            try:
                for filename in files:
                    if Path(filename).suffix.lower() in extensions:
                        midi_files.add(Path(root) / filename)

                        if max_files and len(midi_files) >= max_files:
                            break
            except (OSError, FileNotFoundError):
                continue

            if max_files and len(midi_files) >= max_files:
                break

        if max_files and len(midi_files) >= max_files:
            break

    return list(midi_files)[:max_files] if max_files else list(midi_files)


def get_genre_for_file(filepath: Path, genre_map: Dict[str, str]) -> Optional[str]:
    """
    Look up genre for a MIDI file using the genre map.

    Args:
        filepath: Path to MIDI file
        genre_map: Dict mapping "Artist/SongName" to genre

    Returns:
        Genre string or None if not found
    """
    parts = filepath.parts
    if len(parts) < 2:
        return None

    artist = parts[-2]  # Parent folder = artist
    song_name = filepath.stem  # Filename without extension

    # Strip version suffix like ".1", ".2" from song name
    song_name_base = re.sub(r"\.\d+$", "", song_name)

    lookup_key = f"{artist}/{song_name_base}"
    return genre_map.get(lookup_key)


def get_duration_seconds(music: muspy.Music) -> float:
    """Calculate duration of Music object in seconds."""
    end_time_ticks = music.get_end_time()

    if not music.tempos:
        # Default 120 BPM
        qpm = 120.0
    else:
        qpm = music.tempos[0].qpm

    quarter_notes = end_time_ticks / music.resolution
    return quarter_notes * (60.0 / qpm)


def passes_filters(music: muspy.Music, config: MusicDatasetConfig) -> Tuple[bool, str]:
    """
    Check if a Music object passes all configured filters.

    Args:
        music: MusPy Music object
        config: Dataset configuration with filter settings

    Returns:
        (passes, reason) - reason is empty if passes, otherwise describes why filtered
    """
    # Track count
    num_tracks = len(music.tracks)
    if num_tracks < config.min_tracks:
        return False, f"too_few_tracks ({num_tracks})"
    if num_tracks > config.max_tracks:
        return False, f"too_many_tracks ({num_tracks})"

    # Notes per track
    for i, track in enumerate(music.tracks):
        if len(track.notes) < config.min_notes_per_track:
            return False, f"track_{i}_too_few_notes ({len(track.notes)})"

    # Duration
    if config.min_duration_seconds or config.max_duration_seconds:
        duration = get_duration_seconds(music)
        if config.min_duration_seconds and duration < config.min_duration_seconds:
            return False, f"too_short ({duration:.1f}s)"
        if config.max_duration_seconds and duration > config.max_duration_seconds:
            return False, f"too_long ({duration:.1f}s)"

    return True, ""


def build_dataset(
    config: MusicDatasetConfig,
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> MusicDataset:
    """
    Build a MusicDataset from MIDI files according to config.

    Args:
        config: MusicDatasetConfig specifying paths and filters
        preprocessing_config: Optional preprocessing settings (segmentation, augmentation)

    Returns:
        MusicDataset with filtered and preprocessed music objects
    """
    # Load genre mapping
    genre_map = {}
    if config.genre_tsv_path:
        if config.verbose:
            print(f"Loading genre map from {config.genre_tsv_path}...")
        genre_map = load_genre_map(config.genre_tsv_path)
        if config.verbose:
            print(f"  Loaded {len(genre_map)} genre mappings")

    # Find MIDI files
    if config.verbose:
        print(f"Searching for MIDI files in {len(config.input_dirs)} directories...")
    midi_files = find_midi_files(config.input_dirs, config.max_samples)
    if config.verbose:
        print(f"  Found {len(midi_files)} MIDI files")

    # Log preprocessing config if provided
    if config.verbose and preprocessing_config:
        print(f"Preprocessing enabled:")
        if preprocessing_config.segment_length:
            print(f"  Segmentation: {preprocessing_config.segment_length} ticks")
        if preprocessing_config.enable_transposition:
            print(f"  Transposition: {preprocessing_config.transposition_semitones}")
        if preprocessing_config.enable_tempo_variation:
            print(f"  Tempo variation: {preprocessing_config.tempo_variation_range}")

    # Create dataset with config
    dataset = MusicDataset(config=config)

    # Stats
    stats = {
        "files_processed": 0,
        "samples_added": 0,
        "failed": 0,
        "filtered": 0,
        "no_genre": 0,
        "genre_not_allowed": 0,
        "filter_reasons": {},
    }

    # Process files
    iterator = tqdm(midi_files, desc="Processing") if config.verbose else midi_files

    for midi_path in iterator:
        try:
            # Get genre from map
            genre = get_genre_for_file(midi_path, genre_map)

            # Skip if no genre found and we need genre
            if genre is None:
                stats["no_genre"] += 1
                if config.allowed_genres:
                    # Can't filter by genre if we don't have one
                    continue
                genre = "unknown"

            # Filter by allowed genres
            if config.allowed_genres and genre not in config.allowed_genres:
                stats["genre_not_allowed"] += 1
                continue

            # Load MIDI
            music = muspy.read_midi(str(midi_path))

            # Apply filters (before preprocessing)
            passes, reason = passes_filters(music, config)
            if not passes:
                stats["filtered"] += 1
                base_reason = reason.split(" ")[0]
                stats["filter_reasons"][base_reason] = stats["filter_reasons"].get(base_reason, 0) + 1
                continue

            # Build song_id from artist/song name
            song_id = f"{midi_path.parts[-2]}/{midi_path.stem}" if len(midi_path.parts) >= 2 else midi_path.stem

            # Apply preprocessing if configured
            if preprocessing_config:
                processed_samples = preprocess_music(music, preprocessing_config)

                if len(processed_samples) == 0:
                    stats["filtered"] += 1
                    stats["filter_reasons"]["preprocessing_empty"] = (
                        stats["filter_reasons"].get("preprocessing_empty", 0) + 1
                    )
                    continue

                # Add all preprocessed samples
                for idx, sample in enumerate(processed_samples):
                    # Include segment index in song_id for preprocessed samples
                    segment_song_id = f"{song_id}_seg{idx}" if len(processed_samples) > 1 else song_id
                    dataset.add(sample, genre, song_id=segment_song_id)
                    stats["samples_added"] += 1
            else:
                # No preprocessing, add original
                dataset.add(music, genre, song_id=song_id)
                stats["samples_added"] += 1

            stats["files_processed"] += 1

        except Exception as e:
            stats["failed"] += 1
            if config.verbose:
                tqdm.write(f"Error: {midi_path.name}: {e}")

    # Print summary
    if config.verbose:
        print(f"\nBuild complete:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Samples added: {stats['samples_added']}")
        if preprocessing_config:
            expansion = stats['samples_added'] / max(1, stats['files_processed'])
            print(f"  Expansion ratio: {expansion:.1f}x")
        print(f"  Filtered: {stats['filtered']}")
        if stats["filter_reasons"]:
            print(f"  Filter breakdown:")
            for reason, count in sorted(stats["filter_reasons"].items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")
        print(f"  No genre found: {stats['no_genre']}")
        if config.allowed_genres:
            print(f"  Genre not allowed: {stats['genre_not_allowed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total entries: {len(dataset)}")
        print(f"  Total tracks: {dataset.count_tracks()}")
        print(f"  Genres: {list(dataset.vocabulary.genre_to_id.keys())}")
        print(f"  Active instruments: {dataset.vocabulary.num_active_instruments}")

        # Show top instruments
        inst_stats = dataset.vocabulary.get_instrument_stats()
        if inst_stats:
            top_instruments = sorted(inst_stats.items(), key=lambda x: -x[1])[:5]
            print(f"  Top instruments:")
            for inst_name, count in top_instruments:
                print(f"    {inst_name}: {count} songs")

    return dataset


def build_and_save_dataset(
    config: MusicDatasetConfig,
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> MusicDataset:
    """
    Build dataset and save to disk.

    Args:
        config: MusicDatasetConfig
        preprocessing_config: Optional preprocessing settings

    Returns:
        The built MusicDataset
    """
    dataset = build_dataset(config, preprocessing_config)

    if config.verbose:
        print(f"\nSaving dataset to {config.output_path}...")

    # Save dataset to HDF5
    dataset.save(config.output_path)

    if config.verbose:
        print(f"Dataset saved to {config.output_path}")

    # Save config alongside dataset
    config.save()

    # Save preprocessing config if provided
    if preprocessing_config:
        preproc_path = Path(config.output_path).with_suffix(".preprocessing.json")
        preprocessing_config.save(str(preproc_path))
        if config.verbose:
            print(f"Preprocessing config saved to {preproc_path}")

    if config.verbose:
        print(f"Config saved to {Path(config.output_path).with_suffix('.config.json')}")

    return dataset
