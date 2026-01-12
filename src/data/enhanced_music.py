from dataclasses import dataclass
from typing import Any, Dict

import muspy

@dataclass
class EnhancedMusic:
    music: muspy.Music
    metadata: Dict[str, Any]
    

    def __getattr__(self, name):
        """Delegate attribute access to the underlying Music object"""
        return getattr(self.music, name)

def midi_to_enchanced_music(path: str) -> EnhancedMusic:
    music = muspy.read_midi(path)
    return EnhancedMusic(music=music, metadata={})