import muspy
import io
from datasets import load_dataset

def gigamidi_to_muspy(sample):
    """Convert a GigaMIDI sample to MusPy Music object with metadata."""
    
    # Load MIDI bytes
    midi_bytes = io.BytesIO(sample['music'])
    music = muspy.read_midi(midi_bytes)
    
    # Initialize metadata if needed
    if music.metadata is None:
        music.metadata = muspy.Metadata()
    
    # Add basic metadata
    music.metadata.title = sample.get('title', '')
    music.metadata.creators = [sample.get('artist', '')] if sample.get('artist') else []
    music.metadata.collection = 'GigaMIDI'
    music.metadata.source_filename = sample['md5']
    
    # Store additional metadata as annotations
    if sample.get('music_styles_curated'):
        music.annotations.append(muspy.Annotation(
            time=0,
            annotation=f"genres: {','.join(sample['music_styles_curated'])}"
        ))
    
    if sample.get('tempo'):
        music.annotations.append(muspy.Annotation(
            time=0,
            annotation=f"tempo: {sample['tempo']}"
        ))
    
    if sample.get('loop_instrument_type'):
        music.annotations.append(muspy.Annotation(
            time=0,
            annotation=f"instruments: {','.join(sample['loop_instrument_type'])}"
        ))
    
    return music

# Use it
dataset = load_dataset("Metacreation/GigaMIDI", split="train", streaming=True)
sample = next(iter(dataset))
music = gigamidi_to_muspy(sample)

music.print()


from datasets import load_dataset

# Filter and save
dataset = load_dataset("Metacreation/GigaMIDI", split="train")
filtered = dataset.filter(lambda x: 'classical' in str(x.get('music_styles_curated', '')).lower())

# Save to disk
filtered.save_to_disk("gigamidi_classical")

# Later, load quickly
from datasets import load_from_disk
classical_data = load_from_disk("gigamidi_classical")

Use muspy factory for extra tensors in muspy

Create Vocab for genres and artists etc... map genre to artists tha produce them
    put it in src/data
    
Use cProfile

Use Symusic as used in gigamidi