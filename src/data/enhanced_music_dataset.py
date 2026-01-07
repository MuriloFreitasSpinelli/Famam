from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import h5py
import numpy as np
import muspy
from tqdm import tqdm

from data.enhanced_music import EnhancedMusic


class EnhancedMusicDataset(muspy.Dataset):
    """
    A custom muspy Dataset for EnhancedMusic objects.

    Allows using muspy's to_tensorflow_dataset() with custom factory methods
    while preserving GigaMIDI metadata.
    """

    _info = muspy.DatasetInfo(
        name="EnhancedMusicDataset",
        description="Dataset of EnhancedMusic objects with GigaMIDI metadata",
        homepage="",
    )

    def __init__(self, data: Optional[List[EnhancedMusic]] = None):
        """
        Initialize the dataset.

        Args:
            data: List of EnhancedMusic objects, or None to create empty dataset
        """
        self._data: List[EnhancedMusic] = data if data is not None else []

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> muspy.Music:
        """Return the muspy.Music object at the given index."""
        return self._data[index].music

    def get_enhanced(self, index: int) -> EnhancedMusic:
        """Return the full EnhancedMusic object at the given index."""
        return self._data[index]

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """Return the GigaMIDI metadata for the given index."""
        return self._data[index].gigamidi_metadata

    def append(self, item: EnhancedMusic) -> None:
        """Add an EnhancedMusic object to the dataset."""
        self._data.append(item)

    def extend(self, items: List[EnhancedMusic]) -> None:
        """Add multiple EnhancedMusic objects to the dataset."""
        self._data.extend(items)

    @classmethod
    def from_h5(cls, path: str) -> 'EnhancedMusicDataset':
        """
        Load dataset from an HDF5 file.

        Args:
            path: Path to the HDF5 file

        Returns:
            EnhancedMusicDataset instance
        """
        def extract_scalar(val: Any) -> Any:
            """Extract scalar value from numpy array or return as-is."""
            if isinstance(val, (np.ndarray, np.generic)):
                return val.item()
            return val
        
        data: List[EnhancedMusic] = []

        with h5py.File(path, 'r') as f:
            # Extract scalar from attribute
            num_samples_val = f.attrs['num_samples']
            if isinstance(num_samples_val, (int, np.integer)):
                num_samples = int(num_samples_val)
            else:
                num_samples = int(num_samples_val[()]) # type: ignore

            for idx in range(num_samples):
                grp_item = f[f'sample_{idx}']
                
                # Type guard: ensure grp is a Group
                if not isinstance(grp_item, h5py.Group):
                    continue
                    
                grp = grp_item

                # Load resolution - extract scalar from attribute
                resolution_val = grp.attrs['resolution']
                if isinstance(resolution_val, (int, np.integer)):
                    resolution = int(resolution_val)
                else:
                    resolution = int(resolution_val[()]) # type: ignore

                # Load tempos
                tempos = []
                if 'tempos' in grp.keys():
                    tempos_item = grp['tempos']
                    if isinstance(tempos_item, h5py.Dataset):
                        tempos_arr = tempos_item[:]
                        tempos = [
                            muspy.Tempo(
                                time=int(extract_scalar(t[0])),
                                qpm=float(extract_scalar(t[1]))
                            )
                            for t in tempos_arr
                        ]

                # Load time signatures
                time_signatures = []
                if 'time_signatures' in grp.keys():
                    ts_item = grp['time_signatures']
                    if isinstance(ts_item, h5py.Dataset):
                        ts_arr = ts_item[:]
                        time_signatures = [
                            muspy.TimeSignature(
                                time=int(extract_scalar(ts[0])),
                                numerator=int(extract_scalar(ts[1])),
                                denominator=int(extract_scalar(ts[2]))
                            )
                            for ts in ts_arr
                        ]

                # Load tracks
                tracks = []
                tracks_grp_item = grp['tracks']
                if isinstance(tracks_grp_item, h5py.Group):
                    tracks_grp = tracks_grp_item
                    track_idx = 0
                    while f'track_{track_idx}' in tracks_grp.keys():
                        track_grp_item = tracks_grp[f'track_{track_idx}']
                        
                        if isinstance(track_grp_item, h5py.Group):
                            track_grp = track_grp_item

                            # Extract scalar attributes
                            program_val = track_grp.attrs['program']
                            if isinstance(program_val, (int, np.integer)):
                                program = int(program_val)
                            else:
                                program = int(program_val[()]) # type: ignore
                            
                            is_drum_val = track_grp.attrs['is_drum']
                            if isinstance(is_drum_val, (bool, np.bool_)):
                                is_drum = bool(is_drum_val)
                            elif hasattr(is_drum_val, 'item'):
                                is_drum = bool(is_drum_val.item()) # type: ignore
                            elif hasattr(is_drum_val, '__getitem__'):
                                is_drum = bool(is_drum_val[()]) # type: ignore
                            else:
                                is_drum = bool(is_drum_val)
                            
                            name_val = track_grp.attrs['name']
                            if isinstance(name_val, (str, bytes)):
                                name = name_val.decode('utf-8') if isinstance(name_val, bytes) else name_val
                            elif hasattr(name_val, 'item'):
                                name_decoded = name_val.item() # type: ignore
                                name = name_decoded.decode('utf-8') if isinstance(name_decoded, bytes) else str(name_decoded)
                            elif hasattr(name_val, '__getitem__'):
                                name_decoded = name_val[()] # type: ignore
                                name = name_decoded.decode('utf-8') if isinstance(name_decoded, bytes) else str(name_decoded)
                            else:
                                name = str(name_val)

                            notes = []
                            if 'notes' in track_grp.keys():
                                notes_item = track_grp['notes']
                                if isinstance(notes_item, h5py.Dataset):
                                    notes_arr = notes_item[:]
                                    notes = [
                                        muspy.Note(
                                            time=int(extract_scalar(n[0])),
                                            pitch=int(extract_scalar(n[1])),
                                            duration=int(extract_scalar(n[2])),
                                            velocity=int(extract_scalar(n[3]))
                                        )
                                        for n in notes_arr
                                    ]

                            tracks.append(muspy.Track(
                                program=program,
                                is_drum=is_drum,
                                name=name,
                                notes=notes
                            ))
                        track_idx += 1

                # Create Music object
                music = muspy.Music(
                    resolution=resolution,
                    tempos=tempos,
                    time_signatures=time_signatures,
                    tracks=tracks
                )

                # Load metadata
                metadata_val = grp.attrs['metadata']
                if isinstance(metadata_val, (str, bytes)):
                    metadata_str = metadata_val.decode('utf-8') if isinstance(metadata_val, bytes) else metadata_val
                elif hasattr(metadata_val, 'item'):
                    metadata_decoded = metadata_val.item() # type: ignore
                    metadata_str = metadata_decoded.decode('utf-8') if isinstance(metadata_decoded, bytes) else str(metadata_decoded)
                elif hasattr(metadata_val, '__getitem__'):
                    metadata_decoded = metadata_val[()] # type: ignore
                    metadata_str = metadata_decoded.decode('utf-8') if isinstance(metadata_decoded, bytes) else str(metadata_decoded)
                else:
                    metadata_str = str(metadata_val)
                metadata = json.loads(metadata_str)

                # Create EnhancedMusic
                enhanced = EnhancedMusic(music=music, gigamidi_metadata=metadata)
                data.append(enhanced)

        return cls(data=data)

    def save_h5(self, path: str) -> None:
        """
        Save dataset to an HDF5 file.

        Args:
            path: Path to save the HDF5 file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, 'w') as f:
            f.attrs['num_samples'] = len(self._data)

            for idx, em in enumerate(tqdm(self._data, desc="Saving to HDF5")):
                grp = f.create_group(f'sample_{idx}')
                music = em.music

                # Store resolution
                grp.attrs['resolution'] = music.resolution

                # Store tempos
                if music.tempos:
                    tempos_arr = np.array(
                        [(t.time, t.qpm) for t in music.tempos],
                        dtype=np.float64
                    )
                    grp.create_dataset('tempos', data=tempos_arr)

                # Store time signatures
                if music.time_signatures:
                    ts_arr = np.array(
                        [(ts.time, ts.numerator, ts.denominator) for ts in music.time_signatures],
                        dtype=np.int32
                    )
                    grp.create_dataset('time_signatures', data=ts_arr)

                # Store tracks
                tracks_grp = grp.create_group('tracks')
                for t_idx, track in enumerate(music.tracks):
                    track_grp = tracks_grp.create_group(f'track_{t_idx}')
                    track_grp.attrs['program'] = track.program
                    track_grp.attrs['is_drum'] = track.is_drum
                    track_grp.attrs['name'] = track.name or ''

                    if track.notes:
                        notes_arr = np.array([
                            (n.time, n.pitch, n.duration, n.velocity)
                            for n in track.notes
                        ], dtype=np.int32)
                        track_grp.create_dataset('notes', data=notes_arr)

                # Store metadata as JSON string
                metadata_str = json.dumps(em.gigamidi_metadata, default=str)
                grp.attrs['metadata'] = metadata_str