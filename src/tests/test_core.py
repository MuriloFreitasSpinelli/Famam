"""
Unit tests for src_v4.core module.

Tests:
    - Vocabulary: genre/artist/instrument management, serialization
    - MusicDataset: entry management, statistics, save/load

Author: Murilo de Freitas Spinelli
"""

import pytest
import json
import numpy as np
from pathlib import Path

from src_v4.core.vocabulary import (
    Vocabulary,
    GENERAL_MIDI_INSTRUMENTS,
    INSTRUMENT_NAME_TO_ID,
    DRUM_PROGRAM_ID,
)


class TestVocabularyGenres:
    """Tests for Vocabulary genre management."""

    def test_init_empty(self):
        vocab = Vocabulary()
        assert vocab.num_genres == 0
        assert vocab.genres == []

    def test_init_with_genres(self, sample_genres):
        vocab = Vocabulary(genres=sample_genres)
        assert vocab.num_genres == len(sample_genres)
        assert set(vocab.genres) == set(sample_genres)

    def test_add_genre(self):
        vocab = Vocabulary()
        id1 = vocab.add_genre("Rock")
        id2 = vocab.add_genre("Jazz")
        id3 = vocab.add_genre("Rock")  # Duplicate

        assert id1 == 0
        assert id2 == 1
        assert id3 == id1  # Same ID for duplicate
        assert vocab.num_genres == 2

    def test_get_genre_id(self, sample_genres):
        vocab = Vocabulary(genres=sample_genres)

        assert vocab.get_genre_id("Rock") == 0
        assert vocab.get_genre_id("Jazz") == 1
        assert vocab.get_genre_id("NonExistent") == -1

    def test_get_genre_name(self, sample_genres):
        vocab = Vocabulary(genres=sample_genres)

        assert vocab.get_genre_name(0) == "Rock"
        assert vocab.get_genre_name(1) == "Jazz"
        assert vocab.get_genre_name(999) is None


class TestVocabularyArtists:
    """Tests for Vocabulary artist management."""

    def test_init_with_artists(self, sample_artists):
        vocab = Vocabulary(artists=sample_artists)
        assert vocab.num_artists == len(sample_artists)

    def test_add_artist(self):
        vocab = Vocabulary()
        id1 = vocab.add_artist("Artist A")
        id2 = vocab.add_artist("Artist B")
        id3 = vocab.add_artist("Artist A")  # Duplicate

        assert id1 == 0
        assert id2 == 1
        assert id3 == id1
        assert vocab.num_artists == 2

    def test_get_artist_id(self, sample_artists):
        vocab = Vocabulary(artists=sample_artists)

        assert vocab.get_artist_id("Artist A") == 0
        assert vocab.get_artist_id("NonExistent") == -1


class TestVocabularyInstruments:
    """Tests for Vocabulary instrument management."""

    def test_instrument_constants(self):
        assert DRUM_PROGRAM_ID == 128
        assert len(GENERAL_MIDI_INSTRUMENTS) == 129
        assert GENERAL_MIDI_INSTRUMENTS[0] == "Acoustic Grand Piano"
        assert GENERAL_MIDI_INSTRUMENTS[128] == "Drums"

    def test_get_instrument_id(self):
        vocab = Vocabulary()

        assert vocab.get_instrument_id("Acoustic Grand Piano") == 0
        assert vocab.get_instrument_id("Drums") == 128
        assert vocab.get_instrument_id("Electric Bass (finger)") == 33
        assert vocab.get_instrument_id("NonExistent") == -1

    def test_get_instrument_name(self):
        vocab = Vocabulary()

        assert vocab.get_instrument_name(0) == "Acoustic Grand Piano"
        assert vocab.get_instrument_name(128) == "Drums"
        assert vocab.get_instrument_name(999) == "Unknown"

    def test_num_instruments(self):
        vocab = Vocabulary()
        assert vocab.num_instruments == 129

    def test_register_instrument_usage(self):
        vocab = Vocabulary()
        vocab.add_genre("Rock")

        vocab.register_instrument_usage(0, "song1", "Rock")
        vocab.register_instrument_usage(33, "song1", "Rock")
        vocab.register_instrument_usage(0, "song2", "Rock")
        vocab.register_instrument_usage(128, "song1", "Rock")

        assert vocab.num_active_instruments == 3
        assert "song1" in vocab.get_songs_for_instrument(0)
        assert "song2" in vocab.get_songs_for_instrument(0)
        assert len(vocab.get_songs_for_instrument(0)) == 2

    def test_get_instruments_for_genre(self):
        vocab = Vocabulary()
        vocab.add_genre("Rock")
        vocab.add_genre("Jazz")

        vocab.register_instrument_usage(0, "song1", "Rock")
        vocab.register_instrument_usage(33, "song1", "Rock")
        vocab.register_instrument_usage(65, "song2", "Jazz")

        rock_instruments = vocab.get_instruments_for_genre("Rock")
        assert 0 in rock_instruments
        assert 33 in rock_instruments
        assert 65 not in rock_instruments

    def test_get_top_instruments_for_genre(self):
        vocab = Vocabulary()
        vocab.add_genre("Rock")

        # Piano in 3 songs, Bass in 2 songs, Drums in 1 song
        vocab.register_instrument_usage(0, "song1", "Rock")
        vocab.register_instrument_usage(0, "song2", "Rock")
        vocab.register_instrument_usage(0, "song3", "Rock")
        vocab.register_instrument_usage(33, "song1", "Rock")
        vocab.register_instrument_usage(33, "song2", "Rock")
        vocab.register_instrument_usage(128, "song1", "Rock")

        top = vocab.get_top_instruments_for_genre("Rock", top_n=2, exclude_drums=True)
        assert len(top) == 2
        assert top[0] == 0  # Piano most frequent
        assert top[1] == 33  # Bass second

    def test_get_instrument_stats(self):
        vocab = Vocabulary()

        vocab.register_instrument_usage(0, "song1")
        vocab.register_instrument_usage(0, "song2")
        vocab.register_instrument_usage(33, "song1")

        stats = vocab.get_instrument_stats()
        assert stats["Acoustic Grand Piano"] == 2
        assert stats["Electric Bass (finger)"] == 1


class TestVocabularySerialization:
    """Tests for Vocabulary serialization."""

    def test_to_dict(self, sample_genres, sample_artists):
        vocab = Vocabulary(genres=sample_genres, artists=sample_artists)
        vocab.register_instrument_usage(0, "song1", "Rock")

        data = vocab.to_dict()

        assert "genre_to_id" in data
        assert "artist_to_id" in data
        assert "instrument_to_songs" in data
        assert "genre_to_instruments" in data

    def test_from_dict(self, sample_genres, sample_artists):
        vocab1 = Vocabulary(genres=sample_genres, artists=sample_artists)
        vocab1.register_instrument_usage(0, "song1", "Rock")
        vocab1.register_instrument_usage(33, "song1", "Rock")

        data = vocab1.to_dict()
        vocab2 = Vocabulary.from_dict(data)

        assert vocab2.num_genres == vocab1.num_genres
        assert vocab2.num_artists == vocab1.num_artists
        assert vocab2.get_genre_id("Rock") == vocab1.get_genre_id("Rock")
        assert "song1" in vocab2.get_songs_for_instrument(0)

    def test_save_load(self, temp_dir, sample_genres):
        filepath = temp_dir / "vocab.json"

        vocab1 = Vocabulary(genres=sample_genres)
        vocab1.register_instrument_usage(0, "song1", "Rock")
        vocab1.save(str(filepath))

        assert filepath.exists()

        vocab2 = Vocabulary.load(str(filepath))
        assert vocab2.num_genres == vocab1.num_genres
        assert vocab2.get_genre_id("Rock") == vocab1.get_genre_id("Rock")

    def test_repr(self, sample_genres, sample_artists):
        vocab = Vocabulary(genres=sample_genres, artists=sample_artists)
        vocab.register_instrument_usage(0, "song1")

        repr_str = repr(vocab)
        assert "Vocabulary" in repr_str
        assert "num_genres=4" in repr_str
        assert "num_artists=3" in repr_str


class TestMusicDataset:
    """Tests for MusicDataset."""

    def test_init_empty(self):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset()
        assert len(dataset) == 0
        assert dataset.resolution == 24
        assert dataset.max_seq_length == 2048

    def test_init_with_params(self):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset(resolution=48, max_seq_length=1024)
        assert dataset.resolution == 48
        assert dataset.max_seq_length == 1024

    def test_add_entry(self, mock_music):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset()
        dataset.add(mock_music, genre="Rock", song_id="test_song")

        assert len(dataset) == 1
        assert dataset.entries[0].genre == "Rock"
        assert dataset.entries[0].song_id == "test_song"

    def test_add_auto_song_id(self, mock_music):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset()
        dataset.add(mock_music, genre="Rock")

        assert dataset.entries[0].song_id == "entry_0"

    def test_count_tracks(self, mock_music):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset()
        dataset.add(mock_music, genre="Rock", song_id="song1")
        dataset.add(mock_music, genre="Jazz", song_id="song2")

        # mock_music has 3 tracks, added twice
        assert dataset.count_tracks() == 6

    def test_instrument_registration(self, mock_music):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset()
        dataset.add(mock_music, genre="Rock", song_id="song1")

        # mock_music has Piano (0), Bass (33), Drums (128)
        assert dataset.vocabulary.num_active_instruments == 3
        assert "song1" in dataset.vocabulary.get_songs_for_instrument(0)
        assert "song1" in dataset.vocabulary.get_songs_for_instrument(33)
        assert "song1" in dataset.vocabulary.get_songs_for_instrument(128)

    def test_get_stats(self, mock_music):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset()
        dataset.vocabulary.add_genre("Rock")
        dataset.add(mock_music, genre="Rock", song_id="song1")

        stats = dataset.get_stats()

        assert stats["num_entries"] == 1
        assert stats["num_tracks"] == 3
        assert stats["resolution"] == 24
        assert "Rock" in stats["genres"]

    def test_save_load(self, temp_dir, mock_music):
        from src_v4.core.music_dataset import MusicDataset

        filepath = temp_dir / "dataset.h5"

        dataset1 = MusicDataset(resolution=24, max_seq_length=1024)
        dataset1.vocabulary.add_genre("Rock")
        dataset1.add(mock_music, genre="Rock", song_id="song1")
        dataset1.save(str(filepath))

        assert filepath.exists()

        dataset2 = MusicDataset.load(str(filepath))

        assert len(dataset2) == len(dataset1)
        assert dataset2.resolution == dataset1.resolution
        assert dataset2.max_seq_length == dataset1.max_seq_length
        assert dataset2.vocabulary.num_genres == dataset1.vocabulary.num_genres
        assert dataset2.entries[0].genre == "Rock"
        assert dataset2.entries[0].song_id == "song1"
        assert len(dataset2.entries[0].music.tracks) == 3

    def test_get_base_song_id(self):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset()

        assert dataset._get_base_song_id("Artist/Song") == "Artist/Song"
        assert dataset._get_base_song_id("Artist/Song_0") == "Artist/Song"
        assert dataset._get_base_song_id("Artist/Song_seg0") == "Artist/Song_seg"
        assert dataset._get_base_song_id("entry_123") == "entry_123"

    def test_repr(self, mock_music):
        from src_v4.core.music_dataset import MusicDataset

        dataset = MusicDataset()
        dataset.vocabulary.add_genre("Rock")
        dataset.add(mock_music, genre="Rock")

        repr_str = repr(dataset)
        assert "MusicDataset" in repr_str
        assert "entries=1" in repr_str
        assert "tracks=3" in repr_str


class TestMusicEntry:
    """Tests for MusicEntry dataclass."""

    def test_music_entry(self, mock_music):
        from src_v4.core.music_dataset import MusicEntry

        entry = MusicEntry(
            music=mock_music,
            genre="Rock",
            song_id="test_song"
        )

        assert entry.genre == "Rock"
        assert entry.song_id == "test_song"
        assert entry.music is mock_music
