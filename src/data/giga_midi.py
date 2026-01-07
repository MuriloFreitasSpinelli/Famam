from dataclasses import dataclass
from io import BytesIO
from datasets import load_dataset
from typing import Optional, List, Tuple, Iterator, Dict, Any
import muspy
import pretty_midi
from symusic import Score
import symusic

##Gigamidi dataset is 80% train 10% val 10% test, we will use all of train and split it ourselves, so we
##Will be losing out on 20% of the data, but thats not a problem
dataset = load_dataset("Metacreation/GigaMIDI", split="train", streaming=True)

def get_filtered_samples(
    bpm_range: Optional[Tuple[float, float]] = None,
    genres: Optional[List[str]] = None,
    num_tracks_range: Optional[Tuple[int, int]] = None,
    loop_instruments: Optional[List[str]] = None,
    artists: Optional[List[str]] = None,
    max_samples: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    Filter GigaMIDI samples based on specified criteria.
    
    Args:
        bpm_range: Tuple of (min_bpm, max_bpm) or None to not filter
        genres: List of genres to include or None to not filter
        num_tracks_range: Tuple of (min_tracks, max_tracks) or None to not filter
        loop_instruments: List of loop instrument types to include or None to not filter
        artists: List of artists to include or None to not filter
        max_samples: Maximum number of samples to return or None for unlimited
    
    Yields:
        Filtered samples from the dataset
    """
    samples_yielded: int = 0
    
    for sample in dataset:
        # Check if we've reached the sample limit
        if max_samples is not None and samples_yielded >= max_samples:
            break
        
        # Filter by BPM
        if bpm_range is not None:
            try:
                tempo: float = float(sample['tempo'])
                if tempo < bpm_range[0] or tempo > bpm_range[1]:
                    continue
            except (ValueError, TypeError):
                continue
        
        # Filter by genre
        if genres is not None:
            # Check if any of the specified genres match
            sample_genres: List[str] = sample.get('music_styles_curated', [])
            if not any(genre.lower() in [g.lower() for g in sample_genres] for genre in genres):
                continue
        
        # Filter by number of tracks
        if num_tracks_range is not None:
            num_tracks: int = sample.get('num_tracks', 0)
            if num_tracks < num_tracks_range[0] or num_tracks > num_tracks_range[1]:
                continue
        
        # Filter by loop instruments
        if loop_instruments is not None:
            sample_instruments: List[str] = sample.get('loop_instrument_type', [])
            if not any(instr.lower() in [si.lower() for si in sample_instruments] for instr in loop_instruments):
                continue
        
        # Filter by artist
        if artists is not None:
            sample_artist: str = sample.get('artist', '')
            if not any(artist.lower() in sample_artist.lower() for artist in artists):
                continue
        
        if not is_giga_metadata_complete(sample):
            continue

        yield sample
        samples_yielded += 1

def is_giga_metadata_complete(sample: Dict[str, Any]) -> bool:
    """
    Check if the sample contains all expected GigaMIDI metadata fields.
    
    Args:
        sample: A dictionary from the GigaMIDI dataset
        
    Returns:
        True if all expected metadata fields are present, False otherwise
    """
    required_fields = [
        'md5', 'NOMML', 'num_tracks', 'TPQN', 'total_notes',
        'avg_note_duration', 'avg_velocity', 'min_velocity', 'max_velocity',
        'tempo', 'loop_track_idx', 'loop_instrument_type', 'loop_start',
        'loop_end', 'loop_duration_beats', 'loop_note_density', 'Type',
        'instrument_category__drums-only__0__all-instruments-with-drums__1_no-drums__2',
        'music_styles_curated', 'music_style_scraped', 'music_style_audio_text_Discogs',
        'music_style_audio_text_Lastfm', 'music_style_audio_text_Tagtraum',
        'title', 'artist', 'audio_text_matches_score', 'audio_text_matches_sid',
        'audio_text_matches_mbid', 'MIDI_program_number__expressive_',
        'instrument_group__expressive_', 'start_tick__expressive_',
        'end_tick__expressive_', 'duration_beats__expressive_',
        'note_density__expressive_', 'loopability__expressive_'
    ]
    
    return all(field in sample for field in required_fields)