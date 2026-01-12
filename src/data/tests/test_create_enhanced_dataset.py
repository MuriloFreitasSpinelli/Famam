import pytest
import tempfile
from pathlib import Path
import os

# Skip this test module if muspy is not installed to avoid import errors during collection.
muspy = pytest.importorskip("muspy")

# Ensure project `src` is on sys.path so imports like `from data...` work even if pytest is run from repo root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data.scripts.create_enchanced_dataset import (
    find_midi_files,
    extract_metadata_from_path,
    passes_filter,
    preprocess_music,
    create_enhanced_dataset,
    create_and_save_dataset,
)
from data.configs.enchanced_dataset_config import EnhancedDatasetConfig
from data.enhanced_music import EnhancedMusic


def test_find_midi_files_basic(tmp_path):
    base = tmp_path / "midi_dir"
    base.mkdir()
    (base / "a.mid").write_text("dummy")
    sub = base / "sub"
    sub.mkdir()
    (sub / "b.midi").write_text("dummy")

    found = find_midi_files([str(base)])
    found_strs = sorted([p.name for p in found])

    assert "a.mid" in found_strs
    assert "b.midi" in found_strs


def test_find_midi_files_max_samples(tmp_path):
    base = tmp_path / "many"
    base.mkdir()
    for i in range(5):
        (base / f"f{i}.mid").write_text("x")

    found = find_midi_files([str(base)], max_samples=2)
    assert len(found) == 2


def test_find_midi_files_nonexistent_dir():
    found = find_midi_files(["/this/dir/does/not/exist"])
    assert found == []


def test_extract_metadata_from_path_levels(tmp_path):
    p = Path("/root/genreX/artistY/song.mid")

    meta = extract_metadata_from_path(p)
    assert meta["genre"] == "genreX"
    assert meta["artist"] == "artistY"

    # Turn off genre extraction
    meta2 = extract_metadata_from_path(p, extract_genre=False)
    assert "genre" not in meta2


def make_simple_music(program=0, notes_per_track=1, resolution=24):
    # Helper to construct a simple muspy.Music object
    tracks = []
    for t in range(2):
        notes = [muspy.Note(time=i * resolution, pitch=60 + i, duration=resolution, velocity=64)
                 for i in range(notes_per_track)]
        tracks.append(muspy.Track(program=program + t, is_drum=False, notes=notes))
    return muspy.Music(resolution=resolution, tempos=[muspy.Tempo(time=0, qpm=120.0)], tracks=tracks)


def test_passes_filter_min_tracks_and_notes():
    music = make_simple_music(notes_per_track=2)
    cfg = EnhancedDatasetConfig(dataset_name="t", input_dirs=["."], output_path="out.h5")
    cfg.min_tracks = 2
    cfg.min_notes = 2

    assert passes_filter(music, cfg) is True

    cfg.min_notes = 10
    assert passes_filter(music, cfg) is False


def test_passes_filter_instrument_filters():
    # program 0 maps to acoustic_grand_piano
    music = make_simple_music(program=0, notes_per_track=1)

    cfg = EnhancedDatasetConfig(dataset_name="t2", input_dirs=["."], output_path="out2.h5")
    cfg.allowed_instruments = ["acoustic_grand_piano"]
    assert passes_filter(music, cfg) is True

    cfg.allowed_instruments = ["trumpet"]
    assert passes_filter(music, cfg) is False

    cfg.allowed_instruments = None
    cfg.excluded_instruments = ["acoustic_grand_piano"]
    assert passes_filter(music, cfg) is False


def test_preprocess_music_resolution_quantize_and_remove_empty():
    music = make_simple_music(program=0, notes_per_track=1, resolution=12)
    # Add an empty track
    music.tracks.append(muspy.Track(program=5, is_drum=False, notes=[]))

    cfg = EnhancedDatasetConfig(dataset_name="t3", input_dirs=["."], output_path="out3.h5")
    cfg.resolution = 24
    cfg.quantize = True
    cfg.remove_empty_tracks = True

    processed = preprocess_music(music, cfg)
    assert processed.resolution == 24
    # Empty track should be removed
    assert all(len(t.notes) > 0 for t in processed.tracks)


def test_create_enhanced_dataset_and_save(tmp_path, monkeypatch):
    # Create a small directory structure with some .mid files
    base = tmp_path / "midir"
    (base / "rock" / "Beatles").mkdir(parents=True)

    files = []
    for i in range(3):
        f = base / "rock" / "Beatles" / f"song{i}.mid"
        f.write_text("x")
        files.append(f)

    # Monkeypatch muspy.read_midi to return a valid music object for each path
    def fake_read_midi(path):
        return make_simple_music(program=0, notes_per_track=2)

    monkeypatch.setattr(muspy, "read_midi", fake_read_midi)

    out = tmp_path / "out.h5"
    cfg = EnhancedDatasetConfig(dataset_name="test_create", input_dirs=[str(base)], output_path=str(out))
    cfg.verbose = False

    ds = create_and_save_dataset(cfg)

    assert len(ds) == 3
    assert out.exists()

    # Check vocabulary contains genre and artist
    assert ds.vocabulary.get_genre_id("rock") != -1


def test_create_enhanced_dataset_handles_read_errors(tmp_path, monkeypatch):
    # One file will raise on read
    base = tmp_path / "midir2"
    (base / "g" / "a").mkdir(parents=True)

    good = base / "g" / "a" / "good.mid"
    bad = base / "g" / "a" / "bad.mid"
    good.write_text("x")
    bad.write_text("x")

    def fake_read_midi(path):
        if path.endswith("bad.mid"):
            raise ValueError("corrupt midi")
        return make_simple_music(program=0, notes_per_track=2)

    monkeypatch.setattr(muspy, "read_midi", fake_read_midi)

    out = tmp_path / "out2.h5"
    cfg = EnhancedDatasetConfig(dataset_name="test_err", input_dirs=[str(base)], output_path=str(out))
    cfg.verbose = False

    ds = create_and_save_dataset(cfg)

    # Only the good file should be included
    assert len(ds) == 1
    assert out.exists()
