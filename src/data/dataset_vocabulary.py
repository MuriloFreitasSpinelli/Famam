from typing import Dict, Optional


class DatasetVocabulary:
    def __init__(
        self,
        genres: Optional[list[str]] = None,
        artists: Optional[list[str]] = None
    ):
        """
        Initialize the vocabulary.
        
        Args:
            genres: List of genre names to include in vocabulary
            artists: List of artist names to include in vocabulary
        """
        # Initialize genre vocabulary
        self.genre_to_id: Dict[str, int] = {}
        if genres:
            self.genre_to_id = {genre: idx for idx, genre in enumerate(genres)}
        
        # Initialize artist vocabulary
        self.artist_to_id: Dict[str, int] = {}
        if artists:
            self.artist_to_id = {artist: idx for idx, artist in enumerate(artists)}
        
        # Initialize instrument vocabulary following General MIDI conventions
        # GM Program numbers: 0-127 for instruments, 128+ for percussion/drums
        self.instrument_to_id: Dict[str, int] = self._create_midi_instrument_map()
    
    def _create_midi_instrument_map(self) -> Dict[str, int]:
        """
        Create instrument to ID mapping following General MIDI standard.
        
        Returns:
            Dictionary mapping instrument names to MIDI program numbers
        """
        # General MIDI Program Change messages (0-127)
        instruments = {
            # Piano (0-7)
            'acoustic_grand_piano': 0,
            'bright_acoustic_piano': 1,
            'electric_grand_piano': 2,
            'honky_tonk_piano': 3,
            'electric_piano_1': 4,
            'electric_piano_2': 5,
            'harpsichord': 6,
            'clavinet': 7,
            
            # Chromatic Percussion (8-15)
            'celesta': 8,
            'glockenspiel': 9,
            'music_box': 10,
            'vibraphone': 11,
            'marimba': 12,
            'xylophone': 13,
            'tubular_bells': 14,
            'dulcimer': 15,
            
            # Organ (16-23)
            'drawbar_organ': 16,
            'percussive_organ': 17,
            'rock_organ': 18,
            'church_organ': 19,
            'reed_organ': 20,
            'accordion': 21,
            'harmonica': 22,
            'tango_accordion': 23,
            
            # Guitar (24-31)
            'acoustic_guitar_nylon': 24,
            'acoustic_guitar_steel': 25,
            'electric_guitar_jazz': 26,
            'electric_guitar_clean': 27,
            'electric_guitar_muted': 28,
            'overdriven_guitar': 29,
            'distortion_guitar': 30,
            'guitar_harmonics': 31,
            
            # Bass (32-39)
            'acoustic_bass': 32,
            'electric_bass_finger': 33,
            'electric_bass_pick': 34,
            'fretless_bass': 35,
            'slap_bass_1': 36,
            'slap_bass_2': 37,
            'synth_bass_1': 38,
            'synth_bass_2': 39,
            
            # Strings (40-47)
            'violin': 40,
            'viola': 41,
            'cello': 42,
            'contrabass': 43,
            'tremolo_strings': 44,
            'pizzicato_strings': 45,
            'orchestral_harp': 46,
            'timpani': 47,
            
            # Ensemble (48-55)
            'string_ensemble_1': 48,
            'string_ensemble_2': 49,
            'synth_strings_1': 50,
            'synth_strings_2': 51,
            'choir_aahs': 52,
            'voice_oohs': 53,
            'synth_choir': 54,
            'orchestra_hit': 55,
            
            # Brass (56-63)
            'trumpet': 56,
            'trombone': 57,
            'tuba': 58,
            'muted_trumpet': 59,
            'french_horn': 60,
            'brass_section': 61,
            'synth_brass_1': 62,
            'synth_brass_2': 63,
            
            # Reed (64-71)
            'soprano_sax': 64,
            'alto_sax': 65,
            'tenor_sax': 66,
            'baritone_sax': 67,
            'oboe': 68,
            'english_horn': 69,
            'bassoon': 70,
            'clarinet': 71,
            
            # Pipe (72-79)
            'piccolo': 72,
            'flute': 73,
            'recorder': 74,
            'pan_flute': 75,
            'blown_bottle': 76,
            'shakuhachi': 77,
            'whistle': 78,
            'ocarina': 79,
            
            # Synth Lead (80-87)
            'lead_1_square': 80,
            'lead_2_sawtooth': 81,
            'lead_3_calliope': 82,
            'lead_4_chiff': 83,
            'lead_5_charang': 84,
            'lead_6_voice': 85,
            'lead_7_fifths': 86,
            'lead_8_bass_lead': 87,
            
            # Synth Pad (88-95)
            'pad_1_new_age': 88,
            'pad_2_warm': 89,
            'pad_3_polysynth': 90,
            'pad_4_choir': 91,
            'pad_5_bowed': 92,
            'pad_6_metallic': 93,
            'pad_7_halo': 94,
            'pad_8_sweep': 95,
            
            # Synth Effects (96-103)
            'fx_1_rain': 96,
            'fx_2_soundtrack': 97,
            'fx_3_crystal': 98,
            'fx_4_atmosphere': 99,
            'fx_5_brightness': 100,
            'fx_6_goblins': 101,
            'fx_7_echoes': 102,
            'fx_8_sci_fi': 103,
            
            # Ethnic (104-111)
            'sitar': 104,
            'banjo': 105,
            'shamisen': 106,
            'koto': 107,
            'kalimba': 108,
            'bag_pipe': 109,
            'fiddle': 110,
            'shanai': 111,
            
            # Percussive (112-119)
            'tinkle_bell': 112,
            'agogo': 113,
            'steel_drums': 114,
            'woodblock': 115,
            'taiko_drum': 116,
            'melodic_tom': 117,
            'synth_drum': 118,
            'reverse_cymbal': 119,
            
            # Sound Effects (120-127)
            'guitar_fret_noise': 120,
            'breath_noise': 121,
            'seashore': 122,
            'bird_tweet': 123,
            'telephone_ring': 124,
            'helicopter': 125,
            'applause': 126,
            'gunshot': 127,
            
            # Drums (128 - special marker for drum tracks)
            'drums': 128,
            'percussion': 128,
        }
        
        return instruments
    
    def get_genre_id(self, genre: str) -> int:
        """
        Get the integer ID for a genre.
        
        Args:
            genre: Genre name
            
        Returns:
            Integer ID for the genre, or -1 if not found
        """
        return self.genre_to_id.get(genre, -1)
    
    def get_artist_id(self, artist: str) -> int:
        """
        Get the integer ID for an artist.
        
        Args:
            artist: Artist name
            
        Returns:
            Integer ID for the artist, or -1 if not found
        """
        return self.artist_to_id.get(artist, -1)
    
    def get_instrument_id(self, instrument: str) -> int:
        """
        Get the MIDI program number for an instrument.
        
        Args:
            instrument: Instrument name (e.g., 'acoustic_grand_piano', 'trumpet')
            
        Returns:
            MIDI program number (0-127 for instruments, 128 for drums), or -1 if not found
        """
        return self.instrument_to_id.get(instrument.lower(), -1)
    
    def add_genre(self, genre: str) -> int:
        """
        Add a new genre to the vocabulary.
        
        Args:
            genre: Genre name to add
            
        Returns:
            ID assigned to the genre
        """
        if genre not in self.genre_to_id:
            new_id = len(self.genre_to_id)
            self.genre_to_id[genre] = new_id
            return new_id
        return self.genre_to_id[genre]
    
    def add_artist(self, artist: str) -> int:
        """
        Add a new artist to the vocabulary.
        
        Args:
            artist: Artist name to add
            
        Returns:
            ID assigned to the artist
        """
        if artist not in self.artist_to_id:
            new_id = len(self.artist_to_id)
            self.artist_to_id[artist] = new_id
            return new_id
        return self.artist_to_id[artist]
    
    @property
    def num_genres(self) -> int:
        """Number of genres in vocabulary."""
        return len(self.genre_to_id)
    
    @property
    def num_artists(self) -> int:
        """Number of artists in vocabulary."""
        return len(self.artist_to_id)
    
    @property
    def num_instruments(self) -> int:
        """Number of instruments in vocabulary (129: 0-127 + drums)."""
        return len(self.instrument_to_id)