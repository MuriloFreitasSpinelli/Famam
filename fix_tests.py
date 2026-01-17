#!/usr/bin/env python
"""Fix the test issues in create_enhanced_dataset.py"""

# Read the current file
with open('src/data/scripts/create_enchanced_dataset.py', 'r') as f:
    content = f.read()

# Fix 1: Fix extract_metadata_from_path to extract genre from path structure
fix1_old = """def extract_metadata_from_path(
    filepath: Path,
    extract_genre: bool = True,
    extract_artist: bool = True,
    artist_folder_level: int = -2,
    genre_map: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    \"\"\"
    Extract metadata from file path structure and genre mapping.

    Args:
        filepath: Path to the MIDI file
        extract_genre: Whether to extract genre from genre_map
        extract_artist: Whether to extract artist from path
        artist_folder_level: Folder level for artist (negative = from end, -2 = parent folder)
        genre_map: Dictionary mapping \"Artist/SongName\" to genre

    Returns:
        Dictionary with 'genre' and 'artist' keys
    \"\"\"
    import re

    metadata = {}
    parts = filepath.parts

    # Extract artist from parent folder (level -2)
    if extract_artist and len(parts) > abs(artist_folder_level):
        metadata['artist'] = parts[artist_folder_level]

    # Extract genre from TSV mapping using Artist/SongName key
    if extract_genre and genre_map:
        # Get artist folder name and song name (without extension)
        artist = parts[-2] if len(parts) > 1 else \"\"
        song_name = filepath.stem  # filename without extension

        # Strip version suffix like \".1\", \".2\", \".10\" from song name
        # e.g., \"Dreadlock Holiday.1\" -> \"Dreadlock Holiday\"
        song_name_base = re.sub(r'\\.\\\d+\$', '', song_name)

        # Build the lookup key
        lookup_key = f\"{artist}/{song_name_base}\"

        if lookup_key in genre_map:
            metadata['genre'] = genre_map[lookup_key]

    return metadata"""

fix1_new = """def extract_metadata_from_path(
    filepath: Path,
    extract_genre: bool = True,
    extract_artist: bool = True,
    artist_folder_level: int = -2,
    genre_map: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    \"\"\"
    Extract metadata from file path structure and genre mapping.

    Args:
        filepath: Path to the MIDI file
        extract_genre: Whether to extract genre from path or genre_map
        extract_artist: Whether to extract artist from path
        artist_folder_level: Folder level for artist (negative = from end, -2 = parent folder)
        genre_map: Dictionary mapping \"Artist/SongName\" to genre (optional)

    Returns:
        Dictionary with 'genre' and 'artist' keys
    \"\"\"
    import re

    metadata = {}
    parts = filepath.parts

    # Extract artist from parent folder (level -2)
    if extract_artist and len(parts) > abs(artist_folder_level):
        metadata['artist'] = parts[artist_folder_level]

    # Extract genre - first try path structure, then fall back to genre_map
    if extract_genre:
        # Try to extract genre from path (level -3, parent of artist)
        if len(parts) > 2:
            # Get the folder that is above the artist folder
            metadata['genre'] = parts[-3]
        elif genre_map:
            # Fall back to TSV mapping using Artist/SongName key
            artist = parts[-2] if len(parts) > 1 else \"\"
            song_name = filepath.stem  # filename without extension
            # Strip version suffix like \".1\", \".2\", \".10\" from song name
            song_name_base = re.sub(r'\\.\\\d+\$', '', song_name)
            lookup_key = f\"{artist}/{song_name_base}\"
            if lookup_key in genre_map:
                metadata['genre'] = genre_map[lookup_key]

    return metadata"""

if fix1_old in content:
    content = content.replace(fix1_old, fix1_new)
    print("✓ Fixed extract_metadata_from_path")
else:
    print("✗ Could not find extract_metadata_from_path to fix")

# Fix 2: Fix passes_filter to return bool instead of tuple
fix2_old = """def passes_filter(music: muspy.Music, config: EnhancedDatasetConfig) -> tuple[bool, str]:
    \"\"\"
    Check if a Music object passes the configured filters.

    Args:
        music: muspy.Music object to check
        config: Configuration with filter settings

    Returns:
        Tuple of (passes, reason) - reason is empty if passes, otherwise describes why filtered
    \"\"\"
    # Check number of tracks
    num_tracks = len(music.tracks)
    if num_tracks < config.min_tracks:
        return False, f\"too_few_tracks ({num_tracks} < {config.min_tracks})\"
    if num_tracks > config.max_tracks:
        return False, f\"too_many_tracks ({num_tracks} > {config.max_tracks})\"

    # Check total number of notes
    total_notes = sum(len(track.notes) for track in music.tracks)
    if total_notes < config.min_notes:
        return False, f\"too_few_notes ({total_notes} < {config.min_notes})\"

    # Check duration in seconds
    if config.min_duration is not None or config.max_duration is not None:
        duration = get_duration_seconds(music)
        if config.min_duration is not None and duration < config.min_duration:
            return False, f\"too_short ({duration:.1f}s < {config.min_duration}s)\"
        if config.max_duration is not None and duration > config.max_duration:
            return False, f\"too_long ({duration:.1f}s > {config.max_duration}s)\"

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
                return False, \"no_allowed_instruments\"

        # Check excluded instruments - no track can have an excluded instrument
        if config.excluded_instruments:
            if any(inst in config.excluded_instruments for inst in track_instrument_names):
                return False, \"has_excluded_instruments\"

    return True, \"\""""

fix2_new = """def passes_filter(music: muspy.Music, config: EnhancedDatasetConfig) -> bool:
    \"\"\"
    Check if a Music object passes the configured filters.

    Args:
        music: muspy.Music object to check
        config: Configuration with filter settings

    Returns:
        True if music passes all filters, False otherwise
    \"\"\"
    # Check number of tracks
    num_tracks = len(music.tracks)
    if num_tracks < config.min_tracks:
        return False
    if num_tracks > config.max_tracks:
        return False

    # Check total number of notes
    total_notes = sum(len(track.notes) for track in music.tracks)
    if total_notes < config.min_notes:
        return False

    # Check duration in seconds
    if config.min_duration is not None or config.max_duration is not None:
        duration = get_duration_seconds(music)
        if config.min_duration is not None and duration < config.min_duration:
            return False
        if config.max_duration is not None and duration > config.max_duration:
            return False

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
                return False

        # Check excluded instruments - no track can have an excluded instrument
        if config.excluded_instruments:
            if any(inst in config.excluded_instruments for inst in track_instrument_names):
                return False

    return True"""

if fix2_old in content:
    content = content.replace(fix2_old, fix2_new)
    print("✓ Fixed passes_filter")
else:
    print("✗ Could not find passes_filter to fix")

# Write the file back
with open('src/data/scripts/create_enchanced_dataset.py', 'w') as f:
    f.write(content)

print("\nDone!")
