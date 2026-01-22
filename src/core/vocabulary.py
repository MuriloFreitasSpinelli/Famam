from typing import Dict, Optional, List, Set
import json


# General MIDI Instrument Names (0-127) + Drums (128)
GENERAL_MIDI_INSTRUMENTS: Dict[int, str] = {
    # Piano (0-7)
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano",
    3: "Honky-tonk Piano", 4: "Electric Piano 1", 5: "Electric Piano 2",
    6: "Harpsichord", 7: "Clavinet",
    # Chromatic Percussion (8-15)
    8: "Celesta", 9: "Glockenspiel", 10: "Music Box", 11: "Vibraphone",
    12: "Marimba", 13: "Xylophone", 14: "Tubular Bells", 15: "Dulcimer",
    # Organ (16-23)
    16: "Drawbar Organ", 17: "Percussive Organ", 18: "Rock Organ", 19: "Church Organ",
    20: "Reed Organ", 21: "Accordion", 22: "Harmonica", 23: "Tango Accordion",
    # Guitar (24-31)
    24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)", 26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)", 28: "Electric Guitar (muted)", 29: "Overdriven Guitar",
    30: "Distortion Guitar", 31: "Guitar Harmonics",
    # Bass (32-39)
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)",
    35: "Fretless Bass", 36: "Slap Bass 1", 37: "Slap Bass 2",
    38: "Synth Bass 1", 39: "Synth Bass 2",
    # Strings (40-47)
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass",
    44: "Tremolo Strings", 45: "Pizzicato Strings", 46: "Orchestral Harp", 47: "Timpani",
    # Ensemble (48-55)
    48: "String Ensemble 1", 49: "String Ensemble 2", 50: "Synth Strings 1", 51: "Synth Strings 2",
    52: "Choir Aahs", 53: "Voice Oohs", 54: "Synth Voice", 55: "Orchestra Hit",
    # Brass (56-63)
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet",
    60: "French Horn", 61: "Brass Section", 62: "Synth Brass 1", 63: "Synth Brass 2",
    # Reed (64-71)
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax",
    68: "Oboe", 69: "English Horn", 70: "Bassoon", 71: "Clarinet",
    # Pipe (72-79)
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute",
    76: "Blown Bottle", 77: "Shakuhachi", 78: "Whistle", 79: "Ocarina",
    # Synth Lead (80-87)
    80: "Lead 1 (square)", 81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)", 83: "Lead 4 (chiff)",
    84: "Lead 5 (charang)", 85: "Lead 6 (voice)", 86: "Lead 7 (fifths)", 87: "Lead 8 (bass + lead)",
    # Synth Pad (88-95)
    88: "Pad 1 (new age)", 89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)", 91: "Pad 4 (choir)",
    92: "Pad 5 (bowed)", 93: "Pad 6 (metallic)", 94: "Pad 7 (halo)", 95: "Pad 8 (sweep)",
    # Synth Effects (96-103)
    96: "FX 1 (rain)", 97: "FX 2 (soundtrack)", 98: "FX 3 (crystal)", 99: "FX 4 (atmosphere)",
    100: "FX 5 (brightness)", 101: "FX 6 (goblins)", 102: "FX 7 (echoes)", 103: "FX 8 (sci-fi)",
    # Ethnic (104-111)
    104: "Sitar", 105: "Banjo", 106: "Shamisen", 107: "Koto",
    108: "Kalimba", 109: "Bagpipe", 110: "Fiddle", 111: "Shanai",
    # Percussive (112-119)
    112: "Tinkle Bell", 113: "Agogo", 114: "Steel Drums", 115: "Woodblock",
    116: "Taiko Drum", 117: "Melodic Tom", 118: "Synth Drum", 119: "Reverse Cymbal",
    # Sound Effects (120-127)
    120: "Guitar Fret Noise", 121: "Breath Noise", 122: "Seashore", 123: "Bird Tweet",
    124: "Telephone Ring", 125: "Helicopter", 126: "Applause", 127: "Gunshot",
    # Drums (special program 128)
    128: "Drums",
}

# Reverse mapping: name -> id
INSTRUMENT_NAME_TO_ID: Dict[str, int] = {name: id for id, name in GENERAL_MIDI_INSTRUMENTS.items()}


class Vocabulary:
    """Tracks genre, artist, and instrument mappings to integer IDs for the dataset."""

    def __init__(
        self,
        genres: Optional[List[str]] = None,
        artists: Optional[List[str]] = None
    ):
        self.genre_to_id: Dict[str, int] = {}
        self.artist_to_id: Dict[str, int] = {}

        # Instrument mappings
        # instrument_to_id maps instrument names to their MIDI program numbers (0-128)
        self.instrument_to_id: Dict[str, int] = INSTRUMENT_NAME_TO_ID.copy()

        # Track which songs use which instruments: instrument_id -> set of song identifiers
        self.instrument_to_songs: Dict[int, Set[str]] = {i: set() for i in range(129)}

        # Track which instruments are used in which genre: genre -> set of instrument_ids
        self.genre_to_instruments: Dict[str, Set[int]] = {}

        if genres:
            for genre in genres:
                self.add_genre(genre)

        if artists:
            for artist in artists:
                self.add_artist(artist)

    def add_genre(self, genre: str) -> int:
        """Add genre to vocabulary, returns its ID."""
        if genre not in self.genre_to_id:
            self.genre_to_id[genre] = len(self.genre_to_id)
        return self.genre_to_id[genre]

    def add_artist(self, artist: str) -> int:
        """Add artist to vocabulary, returns its ID."""
        if artist not in self.artist_to_id:
            self.artist_to_id[artist] = len(self.artist_to_id)
        return self.artist_to_id[artist]

    def get_genre_id(self, genre: str) -> int:
        """Get genre ID, returns -1 if not found."""
        return self.genre_to_id.get(genre, -1)

    def get_artist_id(self, artist: str) -> int:
        """Get artist ID, returns -1 if not found."""
        return self.artist_to_id.get(artist, -1)

    def get_instrument_id(self, instrument: str) -> int:
        """Get instrument ID (MIDI program number), returns -1 if not found."""
        return self.instrument_to_id.get(instrument, -1)

    def get_instrument_name(self, instrument_id: int) -> str:
        """Get instrument name from ID, returns 'Unknown' if not found."""
        return GENERAL_MIDI_INSTRUMENTS.get(instrument_id, "Unknown")

    def register_instrument_usage(
        self,
        instrument_id: int,
        song_id: str,
        genre: Optional[str] = None
    ) -> None:
        """
        Register that a song uses a specific instrument.

        Args:
            instrument_id: MIDI program number (0-127) or 128 for drums
            song_id: Unique identifier for the song
            genre: Optional genre to associate with this instrument
        """
        if 0 <= instrument_id <= 128:
            self.instrument_to_songs[instrument_id].add(song_id)

            if genre:
                if genre not in self.genre_to_instruments:
                    self.genre_to_instruments[genre] = set()
                self.genre_to_instruments[genre].add(instrument_id)

    def get_songs_for_instrument(self, instrument_id: int) -> Set[str]:
        """Get all songs that use a specific instrument."""
        return self.instrument_to_songs.get(instrument_id, set())

    def get_instruments_for_genre(self, genre: str) -> Set[int]:
        """Get all instruments used in a specific genre."""
        return self.genre_to_instruments.get(genre, set())

    def get_top_instruments_for_genre(
        self,
        genre: str,
        top_n: int = 3,
        exclude_drums: bool = True
    ) -> List[int]:
        """
        Get the top N most frequently used instruments for a genre.

        Instruments are ranked by their global usage frequency (number of songs
        that use them) among instruments that appear in this genre.

        Args:
            genre: Genre name to get instruments for
            top_n: Number of instruments to return
            exclude_drums: If True, exclude drums from the selection (drums=128)

        Returns:
            List of instrument IDs sorted by frequency (most frequent first)
        """
        genre_instruments = self.get_instruments_for_genre(genre)

        if not genre_instruments:
            return []

        # Filter out drums if requested
        if exclude_drums:
            genre_instruments = {i for i in genre_instruments if i != 128}

        # Sort by global frequency (number of songs using each instrument)
        sorted_instruments = sorted(
            genre_instruments,
            key=lambda i: len(self.instrument_to_songs.get(i, set())),
            reverse=True
        )

        return sorted_instruments[:top_n]

    def get_instrument_stats(self) -> Dict[str, int]:
        """Get count of songs for each instrument that has at least one song."""
        return {
            GENERAL_MIDI_INSTRUMENTS[i]: len(songs)
            for i, songs in self.instrument_to_songs.items()
            if len(songs) > 0
        }

    @property
    def num_genres(self) -> int:
        return len(self.genre_to_id)

    @property
    def num_artists(self) -> int:
        return len(self.artist_to_id)

    @property
    def num_instruments(self) -> int:
        """Total number of possible instruments (129: 0-127 + drums)."""
        return 129

    @property
    def num_active_instruments(self) -> int:
        """Number of instruments actually used in the dataset."""
        return sum(1 for songs in self.instrument_to_songs.values() if len(songs) > 0)

    def to_dict(self) -> dict:
        """Serialize vocabulary to dict."""
        return {
            "genre_to_id": self.genre_to_id,
            "artist_to_id": self.artist_to_id,
            "instrument_to_songs": {str(k): list(v) for k, v in self.instrument_to_songs.items()},
            "genre_to_instruments": {k: list(v) for k, v in self.genre_to_instruments.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        """Deserialize vocabulary from dict."""
        vocab = cls()
        vocab.genre_to_id = data.get("genre_to_id", {})
        vocab.artist_to_id = data.get("artist_to_id", {})

        # Load instrument-to-songs mapping
        inst_to_songs = data.get("instrument_to_songs", {})
        for k, v in inst_to_songs.items():
            vocab.instrument_to_songs[int(k)] = set(v)

        # Load genre-to-instruments mapping
        genre_to_inst = data.get("genre_to_instruments", {})
        for k, v in genre_to_inst.items():
            vocab.genre_to_instruments[k] = set(v)

        return vocab

    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))
