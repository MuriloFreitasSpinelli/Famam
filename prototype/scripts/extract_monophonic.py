from pathlib import Path
import music21 as m21
import pandas as pd

def extract_monophonic_parts(midi_path, output_dir):
    """
    Extract individual parts/voices from a polyphonic MIDI file
    Returns list of extracted monophonic MIDI files
    """
    try:
        score = m21.converter.parse(midi_path)
        extracted_files = []
        
        # Iterate through each part (voice/instrument)
        for part_idx, part in enumerate(score.parts):
            
            # Get notes from this part
            notes = part.flat.notes
            
            # Skip if too few notes (likely not a melody)
            if len(notes) < 20:
                continue
            
            # Get instrument name
            instruments = part.flat.getElementsByClass(m21.instrument.Instrument)
            if instruments:
                instrument_name = instruments[0].instrumentName.replace(' ', '_')
            else:
                instrument_name = f'part_{part_idx}'
            
            # Create new score with just this part
            new_score = m21.stream.Score()
            new_score.append(part)
            
            # Generate filename
            base_name = midi_path.stem
            output_filename = f"{base_name}_{instrument_name}_part{part_idx}.mid"
            output_path = output_dir / output_filename
            
            # Write MIDI file
            new_score.write('midi', fp=output_path)
            
            extracted_files.append({
                'original_file': midi_path.name,
                'extracted_file': output_filename,
                'part_index': part_idx,
                'instrument': instrument_name,
                'num_notes': len(notes),
                'path': str(output_path)
            })
            
        return extracted_files
        
    except Exception as e:
        print(f"  ✗ Error with {midi_path.name}: {e}")
        return []

def extract_all_monophonic_parts(input_dir, output_dir):
    """
    Extract all monophonic parts from all MIDI files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    midi_files = list(input_dir.glob("*.mid"))
    
    print(f"Extracting monophonic parts from {len(midi_files)} MIDI files...")
    print("="*80)
    
    all_extracted = []
    
    for i, midi_path in enumerate(midi_files, 1):
        print(f"\n[{i}/{len(midi_files)}] Processing: {midi_path.name}")
        
        extracted = extract_monophonic_parts(midi_path, output_dir)
        
        if extracted:
            print(f"  ✓ Extracted {len(extracted)} parts")
            for e in extracted:
                print(f"    - {e['instrument']}: {e['num_notes']} notes")
            all_extracted.extend(extracted)
        else:
            print(f"  ⚠ No parts extracted")
    
    print("\n" + "="*80)
    print(f"✅ Total extracted parts: {len(all_extracted)}")
    
    # Save manifest
    if all_extracted:
        df = pd.DataFrame(all_extracted)
        manifest_path = output_dir.parent / "extracted_parts_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"💾 Manifest saved to: {manifest_path}")
        
        # Summary by instrument
        print("\n📊 EXTRACTED PARTS BY INSTRUMENT:")
        print("-"*80)
        instrument_counts = df['instrument'].value_counts()
        for instrument, count in instrument_counts.items():
            print(f"  {instrument}: {count}")
        
        # Note count statistics
        print("\n📈 NOTE COUNT STATISTICS:")
        print("-"*80)
        print(f"  Average: {df['num_notes'].mean():.0f} notes")
        print(f"  Min: {df['num_notes'].min()} notes")
        print(f"  Max: {df['num_notes'].max()} notes")
        
        # Recommend which parts to use
        print("\n🎯 RECOMMENDATIONS:")
        print("-"*80)
        
        # Filter for melody-like parts (reasonable note count)
        good_parts = df[(df['num_notes'] >= 50) & (df['num_notes'] <= 5000)]
        print(f"  Parts with 50-5000 notes (good for training): {len(good_parts)}")
        
        # Common melody instruments
        melody_instruments = good_parts[good_parts['instrument'].str.contains(
            'Piano|Flute|Violin|Voice|Guitar|Saxophone', 
            case=False, 
            na=False
        )]
        print(f"  Melody instruments (Piano/Flute/Violin/etc): {len(melody_instruments)}")
    
    return all_extracted

if __name__ == "__main__":
    input_dir = Path("prototype/data/raw/midi_files")
    output_dir = Path("prototype/data/processed/monophonic_extracted")
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("Add MIDI files first!")
    else:
        extract_all_monophonic_parts(input_dir, output_dir)