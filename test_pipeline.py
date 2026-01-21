"""Quick test of the refactored data processing pipeline."""

from pathlib import Path
from src.data_processing import (
    MusicDatasetConfig,
    PreprocessingConfig,
    build_dataset,
)

# Use absolute paths
project_root = Path(__file__).parent.resolve()

# Create config for 50 samples from clean_midi
config = MusicDatasetConfig(
    name="test_50_samples",
    input_dirs=[str(project_root / "data" / "midi" / "clean_midi")],
    output_path=str(project_root / "data" / "datasets" / "test_50.h5"),
    genre_tsv_path=str(project_root / "data" / "midi" / "clean_midi" / "genre.tsv"),
    max_samples=50,
    max_tracks=64,  # Relaxed from default 16
    min_notes_per_track=0,  # Don't filter by notes
    verbose=True,
)

# Basic preprocessing with segmentation
preprocessing = PreprocessingConfig(
    target_resolution=24,
    segment_length=2400,  # ~100 beats at resolution 24
    max_padding_ratio=0.7,
)

print("Building dataset with 50 samples...")
print("=" * 50)

dataset = build_dataset(config, preprocessing)

print("=" * 50)
print(f"Dataset entries: {len(dataset)}")
print(f"Total tracks: {dataset.count_tracks()}")
print(f"Genres: {dataset.vocabulary.genre_to_id}")

# Test TensorFlow conversion
print()
print("Testing TensorFlow dataset conversion...")
tf_ds = dataset.to_tensorflow_dataset()
sample = next(iter(tf_ds))
print(f"Sample pianoroll shape: {sample['pianoroll'].shape}") # type: ignore
print(f"Sample instrument_id: {sample['instrument_id'].numpy()}") # type: ignore
print(f"Sample genre_id: {sample['genre_id'].numpy()}") # type: ignore

print()
print("Test complete!")
