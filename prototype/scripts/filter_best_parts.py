import pandas as pd
from pathlib import Path
import shutil

def filter_best_parts():
    """
    Filter extracted parts to keep only the best ones for training
    """
    manifest_path = Path("prototype/data/processed/extracted_parts_manifest.csv")
    
    if not manifest_path.exists():
        print("❌ Run extract_monophonic.py first!")
        return
    
    df = pd.read_csv(manifest_path)
    
    print(f"Total extracted parts: {len(df)}")
    
    # Filter criteria:
    # 1. At least 50 notes (too short = not useful)
    # 2. Max 5000 notes (too long = probably drums or bass)
    # 3. Prefer melodic instruments
    
    filtered = df[
        (df['num_notes'] >= 50) & 
        (df['num_notes'] <= 5000)
    ].copy()
    
    print(f"After note count filter (50-5000): {len(filtered)}")
    
    # Optional: prefer melodic instruments
    melody_keywords = ['Piano', 'Flute', 'Violin', 'Voice', 'Guitar', 'Saxophone', 'Clarinet', 'Trumpet']
    filtered['is_melodic'] = filtered['instrument'].str.contains(
        '|'.join(melody_keywords), 
        case=False, 
        na=False
    )
    
    melodic_count = filtered['is_melodic'].sum()
    print(f"Melodic instruments: {melodic_count}")
    
    # Create filtered directory
    output_dir = Path("prototype/data/processed/monophonic_filtered")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy filtered files
    for _, row in filtered.iterrows():
        src = Path(row['path'])
        if src.exists():
            dst = output_dir / src.name
            shutil.copy(src, dst)
    
    print(f"\n✅ Copied {len(filtered)} filtered parts to: {output_dir}")
    
    # Save filtered manifest
    filtered_manifest = output_dir.parent / "filtered_parts_manifest.csv"
    filtered.to_csv(filtered_manifest, index=False)
    print(f"💾 Filtered manifest saved to: {filtered_manifest}")
    
    # Show some examples
    print("\n📝 SAMPLE FILES:")
    print("-"*80)
    for _, row in filtered.head(10).iterrows():
        print(f"  {row['extracted_file']}")
        print(f"    Instrument: {row['instrument']}, Notes: {row['num_notes']}")

if __name__ == "__main__":
    filter_best_parts()
