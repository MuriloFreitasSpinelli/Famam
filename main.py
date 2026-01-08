"""
Famam Data Pipeline Demo
========================
Demonstrates the data pipeline components:
- DataPipelineConfig: Configuration for filtering and output
- GigaMIDI: Streaming filtered samples from the GigaMIDI dataset
- EnhancedMusic: Wrapper combining muspy.Music with GigaMIDI metadata
- EnhancedMusicDataset: Custom muspy Dataset for training
- DataPipeline: Full pipeline execution
"""

from src.data.data_pipeline_config import DataPipelineConfig
from src.data.giga_midi import get_filtered_samples, is_giga_metadata_complete
from src.data.enhanced_music import EnhancedMusic, sample_to_enhanced_music
from src.data.enhanced_music_dataset import EnhancedMusicDataset
from src.data.data_pipeline import run_pipeline


def demo_config():
    """Demo: Creating and using DataPipelineConfig"""
    print("\n" + "=" * 60)
    print("DEMO 1: DataPipelineConfig")
    print("=" * 60)

    # Create a config with various filters
    config = DataPipelineConfig(
        bpm_range=(80, 140),
        genres=["electronic", "pop"],
        num_tracks_range=(2, 8),
        max_samples=10,
        output_path="data/processed/demo_dataset.h5"
    )

    print(f"BPM Range: {config.bpm_range}")
    print(f"Genres: {config.genres}")
    print(f"Track Range: {config.num_tracks_range}")
    print(f"Max Samples: {config.max_samples}")
    print(f"Output Path: {config.output_path}")

    # Save and reload config
    config.save("data/demo_config.json")
    print("\nConfig saved to data/demo_config.json")

    loaded_config = DataPipelineConfig.load("data/demo_config.json")
    print(f"Config reloaded successfully: bpm_range={loaded_config.bpm_range}")

    return config


def demo_giga_midi_filtering():
    """Demo: Filtering samples from GigaMIDI"""
    print("\n" + "=" * 60)
    print("DEMO 2: GigaMIDI Filtering")
    print("=" * 60)

    # Get a few samples with filters
    print("Fetching 3 samples with BPM between 100-130...")
    samples = list(get_filtered_samples(
        bpm_range=(100, 130),
        max_samples=3
    ))

    print(f"Retrieved {len(samples)} samples")

    for i, sample in enumerate(samples):
        print(f"\nSample {i + 1}:")
        print(f"  Title: {sample.get('title', 'Unknown')}")
        print(f"  Artist: {sample.get('artist', 'Unknown')}")
        print(f"  Tempo: {sample.get('tempo', 'N/A')} BPM")
        print(f"  Tracks: {sample.get('num_tracks', 'N/A')}")
        print(f"  Genres: {sample.get('music_styles_curated', [])}")
        print(f"  Metadata complete: {is_giga_metadata_complete(sample)}")

    return samples


def demo_enhanced_music(samples):
    """Demo: Converting samples to EnhancedMusic"""
    print("\n" + "=" * 60)
    print("DEMO 3: EnhancedMusic Conversion")
    print("=" * 60)

    if not samples:
        print("No samples available for conversion demo")
        return []

    enhanced_list = []
    for i, sample in enumerate(samples):
        try:
            enhanced = sample_to_enhanced_music(sample)
            enhanced_list.append(enhanced)

            print(f"\nEnhancedMusic {i + 1}:")
            print(f"  Resolution: {enhanced.music.resolution}")
            print(f"  Tracks: {len(enhanced.music.tracks)}")
            print(f"  Has drums: {enhanced.gigamidi_metadata.get('has_drums', False)}")

            # Show tempo info
            if enhanced.music.tempos:
                print(f"  First tempo: {enhanced.music.tempos[0].qpm:.1f} QPM")

            # Show track info
            for j, track in enumerate(enhanced.music.tracks[:3]):  # Show first 3 tracks
                print(f"  Track {j}: {track.name or 'Unnamed'} - {len(track.notes)} notes")

        except Exception as e:
            print(f"Failed to convert sample {i + 1}: {e}")

    return enhanced_list


def demo_enhanced_music_dataset(enhanced_list):
    """Demo: Using EnhancedMusicDataset"""
    print("\n" + "=" * 60)
    print("DEMO 4: EnhancedMusicDataset")
    print("=" * 60)

    if not enhanced_list:
        print("No EnhancedMusic objects available for dataset demo")
        return None

    # Create dataset from list
    dataset = EnhancedMusicDataset(data=enhanced_list)
    print(f"Created dataset with {len(dataset)} samples")

    # Access items
    print("\nAccessing dataset items:")
    for i in range(len(dataset)):
        music = dataset[i]  # Returns muspy.Music
        enhanced = dataset.get_enhanced(i)  # Returns full EnhancedMusic
        metadata = dataset.get_metadata(i)  # Returns just metadata

        print(f"  [{i}] Tracks: {len(music.tracks)}, "
              f"Title: {metadata.get('title', 'Unknown')[:30]}")

    # Save to HDF5
    output_path = "data/processed/demo_dataset.h5"
    print(f"\nSaving dataset to {output_path}...")
    dataset.save_h5(output_path)
    print("Dataset saved!")

    # Reload from HDF5
    print("Reloading dataset from HDF5...")
    reloaded = EnhancedMusicDataset.from_h5(output_path)
    print(f"Reloaded dataset with {len(reloaded)} samples")

    return dataset


def demo_full_pipeline():
    """Demo: Running the full data pipeline"""
    print("\n" + "=" * 60)
    print("DEMO 5: Full Data Pipeline")
    print("=" * 60)

    # Create config for a small demo run
    config = DataPipelineConfig(
        bpm_range=(90, 150),
        max_samples=5,
        output_path="data/processed/pipeline_demo.h5"
    )

    print("Running pipeline with config:")
    print(f"  BPM Range: {config.bpm_range}")
    print(f"  Max Samples: {config.max_samples}")
    print(f"  Output: {config.output_path}")
    print()

    # Run the pipeline
    dataset = run_pipeline(config)

    print(f"\nPipeline complete!")
    print(f"Final dataset size: {len(dataset)} samples")

    # Show summary of results
    if len(dataset) > 0:
        print("\nDataset summary:")
        total_notes = 0
        total_tracks = 0
        for i in range(len(dataset)):
            music = dataset[i]
            total_tracks += len(music.tracks)
            for track in music.tracks:
                total_notes += len(track.notes)

        print(f"  Total tracks: {total_tracks}")
        print(f"  Total notes: {total_notes}")
        print(f"  Avg tracks per sample: {total_tracks / len(dataset):.1f}")
        print(f"  Avg notes per sample: {total_notes / len(dataset):.1f}")

    return dataset


def main():
    """Run all demos"""
    print("=" * 60)
    print("FAMAM DATA PIPELINE DEMO")
    print("=" * 60)

    # Run individual component demos
    config = demo_config()
    samples = demo_giga_midi_filtering()
    enhanced_list = demo_enhanced_music(samples)
    dataset = demo_enhanced_music_dataset(enhanced_list)

    # Run full pipeline demo
    pipeline_dataset = demo_full_pipeline()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
