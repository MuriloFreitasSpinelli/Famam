"""
Preprocessing functions for Music objects.

Handles:
    - Resolution adjustment
    - Quantization
    - Track cleanup
    - Segmentation into fixed-length chunks
    - Augmentation (transposition, tempo variation)
"""

from typing import List, Tuple, TYPE_CHECKING
import copy

import numpy as np

if TYPE_CHECKING:
    import muspy
    from ..config import MusicDatasetConfig


def adjust_resolution(music: "muspy.Music", target_resolution: int) -> "muspy.Music":
    """
    Adjust music resolution to target ticks per quarter note.

    Args:
        music: MusPy Music object
        target_resolution: Target resolution in ticks per quarter note

    Returns:
        Music object with adjusted resolution
    """
    if music.resolution == target_resolution:
        return music
    return music.adjust_resolution(target_resolution)


def quantize_music(music: "muspy.Music", grid: int = 1) -> "muspy.Music":
    """
    Quantize note timings to a grid.

    Args:
        music: MusPy Music object
        grid: Quantization grid size in ticks

    Returns:
        Music object with quantized note timings
    """
    music = copy.deepcopy(music)

    for track in music.tracks:
        for note in track.notes:
            note.time = round(note.time / grid) * grid
            note.duration = max(grid, round(note.duration / grid) * grid)

    return music


def remove_empty_tracks(music: "muspy.Music") -> "muspy.Music":
    """
    Remove tracks that have no notes.

    Args:
        music: MusPy Music object

    Returns:
        Music object with empty tracks removed
    """
    music = copy.deepcopy(music)
    music.tracks = [track for track in music.tracks if len(track.notes) > 0]
    return music


def _extract_segment(
    music: "muspy.Music",
    start_time: int,
    end_time: int,
) -> "muspy.Music":
    """
    Extract a time segment from a Music object.

    Args:
        music: Source Music object
        start_time: Start time in ticks
        end_time: End time in ticks

    Returns:
        New Music object containing only notes within the time range
    """
    import muspy

    segment = muspy.Music(
        resolution=music.resolution,
        tempos=copy.deepcopy(music.tempos),
        metadata=copy.deepcopy(music.metadata),
    )

    for track in music.tracks:
        new_track = muspy.Track(
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
        )

        for note in track.notes:
            # Note starts within segment
            if start_time <= note.time < end_time:
                new_note = muspy.Note(
                    time=note.time - start_time,
                    pitch=note.pitch,
                    duration=min(note.duration, end_time - note.time),
                    velocity=note.velocity,
                )
                new_track.notes.append(new_note)

            # Note starts before segment but extends into it
            elif note.time < start_time < note.time + note.duration:
                new_note = muspy.Note(
                    time=0,
                    pitch=note.pitch,
                    duration=min(note.time + note.duration - start_time, end_time - start_time),
                    velocity=note.velocity,
                )
                new_track.notes.append(new_note)

        if len(new_track.notes) > 0:
            segment.tracks.append(new_track)

    return segment


def segment_music(
    music: "muspy.Music",
    segment_length: int,
    max_padding_ratio: float = 0.3,
) -> List["muspy.Music"]:
    """
    Split music into fixed-length segments.

    If the final segment requires more than max_padding_ratio padding,
    it will be discarded.

    Args:
        music: MusPy Music object
        segment_length: Length of each segment in ticks
        max_padding_ratio: Maximum allowed padding ratio (0.0-1.0)

    Returns:
        List of Music objects, each of segment_length duration
    """
    end_time = music.get_end_time()

    if end_time == 0:
        return []

    segments = []
    num_full_segments = end_time // segment_length
    remainder = end_time % segment_length

    # Create full segments
    for i in range(num_full_segments):
        start_time = i * segment_length
        end_time_segment = (i + 1) * segment_length
        segment = _extract_segment(music, start_time, end_time_segment)
        segments.append(segment)

    # Handle remainder (partial segment)
    if remainder > 0:
        padding_ratio = (segment_length - remainder) / segment_length

        if padding_ratio <= max_padding_ratio:
            start_time = num_full_segments * segment_length
            segment = _extract_segment(music, start_time, start_time + remainder)
            segments.append(segment)

    return segments


def transpose_music(music: "muspy.Music", semitones: int) -> "muspy.Music":
    """
    Transpose all pitches by a constant interval.

    Notes that would fall outside valid MIDI range (0-127) are removed.

    Args:
        music: MusPy Music object
        semitones: Number of semitones to shift (positive = up, negative = down)

    Returns:
        Transposed Music object
    """
    music = copy.deepcopy(music)

    for track in music.tracks:
        if track.is_drum:
            continue

        valid_notes = []
        for note in track.notes:
            new_pitch = note.pitch + semitones
            if 0 <= new_pitch <= 127:
                note.pitch = new_pitch
                valid_notes.append(note)

        track.notes = valid_notes

    return music


def generate_transpositions(
    music: "muspy.Music",
    semitones: Tuple[int, ...],
) -> List["muspy.Music"]:
    """
    Generate multiple transposed versions of a Music object.

    Args:
        music: MusPy Music object
        semitones: Tuple of semitone shifts to apply

    Returns:
        List of transposed Music objects (does not include original)
    """
    return [transpose_music(music, shift) for shift in semitones]


def vary_tempo(music: "muspy.Music", factor: float) -> "muspy.Music":
    """
    Vary the tempo by scaling all note timings and durations.

    Args:
        music: MusPy Music object
        factor: Tempo variation factor (< 1.0 = slower, > 1.0 = faster)

    Returns:
        Music object with varied tempo
    """
    music = copy.deepcopy(music)

    for track in music.tracks:
        for note in track.notes:
            note.time = round(note.time / factor)
            note.duration = max(1, round(note.duration / factor))

    return music


def generate_tempo_variations(
    music: "muspy.Music",
    variation_range: Tuple[float, float],
    num_variations: int,
) -> List["muspy.Music"]:
    """
    Generate multiple tempo-varied versions of a Music object.

    Args:
        music: MusPy Music object
        variation_range: (min_factor, max_factor) for tempo scaling
        num_variations: Number of variations to generate

    Returns:
        List of tempo-varied Music objects (does not include original)
    """
    if num_variations < 1:
        return []

    min_factor, max_factor = variation_range

    if num_variations == 1:
        factors = [(min_factor + max_factor) / 2]
    else:
        factors = np.linspace(min_factor, max_factor, num_variations).tolist()

    # Remove factor=1.0 if present (that's the original)
    factors = [f for f in factors if abs(f - 1.0) > 0.01]

    return [vary_tempo(music, f) for f in factors]


def preprocess_music(
    music: "muspy.Music",
    config: "MusicDatasetConfig",
) -> List["muspy.Music"]:
    """
    Apply full preprocessing pipeline to a Music object.

    Pipeline:
        1. Adjust resolution
        2. Quantize (if enabled)
        3. Remove empty tracks (if enabled)
        4. Segment (if segment_length specified)
        5. Augment (transposition, tempo variation)

    This may produce multiple Music objects due to segmentation and augmentation.

    Args:
        music: MusPy Music object
        config: MusicDatasetConfig with preprocessing settings

    Returns:
        List of preprocessed Music objects
    """
    # Step 1: Adjust resolution
    music = adjust_resolution(music, config.resolution)

    # Step 2: Quantize
    if config.quantize:
        music = quantize_music(music, config.quantize_grid)

    # Step 3: Remove empty tracks
    if config.remove_empty_tracks:
        music = remove_empty_tracks(music)
        if len(music.tracks) == 0:
            return []

    # Step 4: Segment
    if config.segment_length is not None:
        segments = segment_music(music, config.segment_length, config.max_padding_ratio)
    else:
        segments = [music]

    if len(segments) == 0:
        return []

    # Step 5: Augmentation (applied independently per segment)
    results = []

    for segment in segments:
        # Add original
        results.append(segment)

        # Transposition augmentation
        if config.enable_transposition:
            transposed = generate_transpositions(segment, config.transposition_semitones)
            results.extend(transposed)

        # Tempo variation augmentation
        if config.enable_tempo_variation:
            tempo_varied = generate_tempo_variations(
                segment,
                config.tempo_variation_range,
                config.tempo_variation_steps,
            )
            results.extend(tempo_varied)

    return results