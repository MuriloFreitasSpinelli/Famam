from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict

import muspy
import pretty_midi

from .giga_midi import is_giga_metadata_complete

##Music class from Muspy has limited metadata, we should use all the metadata available for training usually

@dataclass
class EnhancedMusic:
    """A wrapper around MusPy Music with additional GigaMIDI metadata"""
    music: muspy.Music
    gigamidi_metadata: Dict[str, Any]
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying Music object"""
        return getattr(self.music, name)

def sample_to_enhanced_music(sample: Dict[str, Any]) -> EnhancedMusic:
    """
    Convert a GigaMIDI sample dictionary to an EnhancedMusic object.
    
    Args:
        sample: A dictionary from the GigaMIDI dataset
        
    Returns:
        An EnhancedMusic object with both MusPy Music and GigaMIDI metadata
    """
    # Step 1: Read the MIDI bytes using pretty_midi
    midi_bytes: bytes = sample['music']
    midi_file = BytesIO(midi_bytes)
    pm = pretty_midi.PrettyMIDI(midi_file)
    
    # Step 2: Convert pretty_midi to MusPy Music
    music: muspy.Music = muspy.from_pretty_midi(pm)
    
    # Copy all metadata except 'music' and 'split'
    gigamidi_metadata = {k: v for k, v in sample.items() if k not in ['music', 'split']}

    # Add has_drums convenience field (None if metadata not available)
    drum_category = sample.get('instrument_category__drums-only__0__all-instruments-with-drums__1_no-drums__2')
    if drum_category is not None:
        gigamidi_metadata['has_drums'] = drum_category in [0, 1]
    else:
        gigamidi_metadata['has_drums'] = None
    return EnhancedMusic(music=music, gigamidi_metadata=gigamidi_metadata)
