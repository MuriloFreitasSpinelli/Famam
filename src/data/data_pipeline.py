from typing import List, Any, Dict
from pathlib import Path
import json
import h5py
import numpy as np
from tqdm import tqdm

from .giga_midi import get_filtered_samples
from .enhanced_music import EnhancedMusic, sample_to_enhanced_music
from .data_pipeline_config import DataPipelineConfig
from .enhanced_music_dataset import EnhancedMusicDataset


def _serialize_enhanced_music(em: EnhancedMusic) -> Dict[str, Any]:
    """Serialize EnhancedMusic to a dictionary suitable for HDF5 storage."""
    music = em.music

    # Serialize tracks
    tracks_data = []
    for track in music.tracks:
        notes_data = []
        for note in track.notes:
            notes_data.append({
                'time': note.time,
                'pitch': note.pitch,
                'duration': note.duration,
                'velocity': note.velocity
            })
        tracks_data.append({
            'program': track.program,
            'is_drum': track.is_drum,
            'name': track.name or '',
            'notes': notes_data
        })

    return {
        'resolution': music.resolution,
        'tempos': [(t.time, t.qpm) for t in music.tempos] if music.tempos else [],
        'time_signatures': [(ts.time, ts.numerator, ts.denominator) for ts in music.time_signatures] if music.time_signatures else [],
        'tracks': tracks_data,
        'metadata': em.gigamidi_metadata
    }


def _save_dataset_to_h5(dataset: List[EnhancedMusic], output_path: str) -> None:
    """Save a list of EnhancedMusic objects to an HDF5 file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_samples'] = len(dataset)

        for idx, em in enumerate(tqdm(dataset, desc="Saving to HDF5")):
            grp = f.create_group(f'sample_{idx}')
            serialized = _serialize_enhanced_music(em)

            # Store resolution
            grp.attrs['resolution'] = serialized['resolution']

            # Store tempos as dataset
            if serialized['tempos']:
                tempos_arr = np.array(serialized['tempos'], dtype=np.float64)
                grp.create_dataset('tempos', data=tempos_arr)

            # Store time signatures
            if serialized['time_signatures']:
                ts_arr = np.array(serialized['time_signatures'], dtype=np.int32)
                grp.create_dataset('time_signatures', data=ts_arr)

            # Store tracks
            tracks_grp = grp.create_group('tracks')
            for t_idx, track in enumerate(serialized['tracks']):
                track_grp = tracks_grp.create_group(f'track_{t_idx}')
                track_grp.attrs['program'] = track['program']
                track_grp.attrs['is_drum'] = track['is_drum']
                track_grp.attrs['name'] = track['name']

                if track['notes']:
                    notes_arr = np.array([
                        (n['time'], n['pitch'], n['duration'], n['velocity'])
                        for n in track['notes']
                    ], dtype=np.int32)
                    track_grp.create_dataset('notes', data=notes_arr)

            # Store metadata as JSON string
            metadata_str = json.dumps(serialized['metadata'], default=str)
            grp.attrs['metadata'] = metadata_str


def run_pipeline(config: DataPipelineConfig) -> EnhancedMusicDataset:
    """
    Run the data pipeline with the given configuration.

    Args:
        config: DataPipelineConfig with filter settings and output path

    Returns:
        EnhancedMusicDataset that can be used with muspy's to_tensorflow_dataset()
    """
    # Step 1: Get filtered samples from GigaMIDI
    print("Fetching samples from GigaMIDI...")
    samples_iterator = get_filtered_samples(
        bpm_range=config.bpm_range,
        genres=config.genres,
        num_tracks_range=config.num_tracks_range,
        loop_instruments=config.loop_instruments,
        artists=config.artists,
        max_samples=config.max_samples
    )

    # Step 2: Transform all samples into EnhancedMusic
    print("Converting samples to EnhancedMusic...")
    samples: List[EnhancedMusic] = []
    for sample in tqdm(samples_iterator, desc="Processing samples", total=config.max_samples):
        try:
            enhanced = sample_to_enhanced_music(sample)
            samples.append(enhanced)
        except Exception as e:
            print(f"Warning: Failed to convert sample: {e}")
            continue

    print(f"Converted {len(samples)} samples to EnhancedMusic")

    # Step 3: Create dataset and save before preprocessing
    dataset = EnhancedMusicDataset(data=samples)
    print(f"Saving dataset to {config.output_path}...")
    dataset.save_h5(config.output_path)
    print("Dataset saved successfully")

    # Step 4: Data preprocessing (placeholder for future implementation)
    for i in range(len(dataset)):
        enhanced_music = dataset.get_enhanced(i)
        # TODO: Add preprocessing steps here
        pass

    ##dataset.to_tensorflow_dataset() with custom factory method for creating tensors for training
    ##put all relevant metadata we want for training in this facotry method as well as segment here
    ##Encoding is also done automatically here we can pick from various encodings. 
    return dataset


if __name__ == "__main__":
    # Example usage
    config = DataPipelineConfig(
        bpm_range=(60, 180),
        max_samples=100,
        output_path="data/processed/dataset.h5"
    )

    dataset = run_pipeline(config)
    print(f"Pipeline complete. Processed {len(dataset)} samples.")
