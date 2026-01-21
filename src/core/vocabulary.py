from typing import Dict, Optional, List
import json


class Vocabulary:
    """Tracks genre and artist mappings to integer IDs for the dataset."""

    def __init__(
        self,
        genres: Optional[List[str]] = None,
        artists: Optional[List[str]] = None
    ):
        self.genre_to_id: Dict[str, int] = {}
        self.artist_to_id: Dict[str, int] = {}

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

    @property
    def num_genres(self) -> int:
        return len(self.genre_to_id)

    @property
    def num_artists(self) -> int:
        return len(self.artist_to_id)

    def to_dict(self) -> dict:
        """Serialize vocabulary to dict."""
        return {
            "genre_to_id": self.genre_to_id,
            "artist_to_id": self.artist_to_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        """Deserialize vocabulary from dict."""
        vocab = cls()
        vocab.genre_to_id = data.get("genre_to_id", {})
        vocab.artist_to_id = data.get("artist_to_id", {})
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
