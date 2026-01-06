from pathlib import Path
from midi2audio import FluidSynth
import subprocess
import platform

def download_soundfont():
    """
    Instructions for downloading a soundfont
    """
    print("="*80)
    print("SOUNDFONT NEEDED")
    print("="*80)
    print("\nFluidSynth needs a soundfont (.sf2) file to render MIDI to audio.")
    print("\nDownload options:")
    print("1. FluidR3_GM.sf2 (recommended, ~140MB)")
    print("   URL: https://member.keymusician.com/Member/FluidR3_GM/index.html")
    print("\n2. GeneralUser GS (alternative, ~30MB)")
    print("   URL: https://schristiancollins.com/generaluser.php")
    print("\n3. Quick download link:")
    print("   https://github.com/FluidSynth/fluidsynth/raw/master/sf2/FluidR3_GM.sf2")
    print("\nAfter downloading:")
    print(f"Place the .sf2 file in: data/soundfonts/")
    print("="*80)

def find_soundfont():
    """
    Look for soundfont files
    """
    soundfont_dir = Path("data/soundfonts")
    soundfont_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for .sf2 files
    sf2_files = list(soundfont_dir.glob("*.sf2"))
    
    if sf2_files:
        print(f"✓ Found soundfont: {sf2_files[0].name}")
        return sf2_files[0]
    
    # Check common system locations (Linux/Mac)
    system_paths = [
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
        Path("/usr/share/soundfonts/default.sf2"),
        Path("/usr/local/share/soundfonts/FluidR3_GM.sf2"),
    ]
    
    for path in system_paths:
        if path.exists():
            print(f"✓ Found system soundfont: {path}")
            return path
    
    return None

def render_midi_to_wav(midi_path, output_path, soundfont_path):
    """
    Render a MIDI file to WAV using FluidSynth
    """
    try:
        fs = FluidSynth(str(soundfont_path))
        fs.midi_to_audio(str(midi_path), str(output_path))
        return True
    except Exception as e:
        print(f"✗ Error rendering {midi_path.name}: {e}")
        return False

def play_audio(audio_path):
    """
    Play audio file using system default player
    """
    system = platform.system()
    
    try:
        if system == "Windows":
            # Windows Media Player
            subprocess.run(["start", str(audio_path)], shell=True, check=True)
        elif system == "Darwin":  # macOS
            subprocess.run(["afplay", str(audio_path)], check=True)
        elif system == "Linux":
            # Try common Linux audio players
            players = ["vlc", "mpv", "ffplay", "aplay"]
            for player in players:
                try:
                    subprocess.run([player, str(audio_path)], check=True)
                    return True
                except FileNotFoundError:
                    continue
            print("No audio player found. Install vlc, mpv, or ffplay")
            return False
        
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def render_and_listen():
    """
    Main function to render and play generated MIDI files
    """
    print("\n" + "="*80)
    print("RENDER AND LISTEN TO GENERATED MUSIC")
    print("="*80 + "\n")
    
    # Check for soundfont
    soundfont_path = find_soundfont()
    
    if soundfont_path is None:
        print("✗ No soundfont found!")
        download_soundfont()
        return
    
    # Setup directories
    midi_dir = Path("outputs/generated")
    audio_dir = Path("outputs/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    if not midi_dir.exists():
        print(f"✗ No generated MIDI files found in {midi_dir}")
        print("Run generate_music.py first!")
        return
    
    midi_files = sorted(midi_dir.glob("*.mid"))
    
    if not midi_files:
        print(f"✗ No MIDI files found in {midi_dir}")
        return
    
    print(f"Found {len(midi_files)} MIDI files to render")
    print(f"Using soundfont: {soundfont_path.name}")
    print("="*80 + "\n")
    
    # Render all MIDI files
    rendered_files = []
    
    for i, midi_file in enumerate(midi_files, 1):
        output_file = audio_dir / f"{midi_file.stem}.wav"
        
        print(f"[{i}/{len(midi_files)}] Rendering: {midi_file.name}")
        
        if render_midi_to_wav(midi_file, output_file, soundfont_path):
            print(f"✓ Saved: {output_file.name}")
            rendered_files.append(output_file)
        
        print()
    
    print("="*80)
    print(f"✅ Rendered {len(rendered_files)}/{len(midi_files)} files")
    print(f"Audio files saved to: {audio_dir}")
    print("="*80 + "\n")
    
    # Ask user if they want to play the files
    if rendered_files:
        response = input("Would you like to play the audio files now? (y/n): ").lower().strip()
        
        if response == 'y':
            print("\nPlaying audio files...")
            print("="*80 + "\n")
            
            for audio_file in rendered_files:
                print(f"Playing: {audio_file.name}")
                print("(File will open in your default audio player)")
                play_audio(audio_file)
                
                # Wait for user to continue
                input("\nPress Enter to play next file (or Ctrl+C to stop)...")
                print()

def render_specific_file(midi_filename):
    """
    Render a specific MIDI file
    """
    soundfont_path = find_soundfont()
    
    if soundfont_path is None:
        print("✗ No soundfont found!")
        download_soundfont()
        return
    
    midi_path = Path("outputs/generated") / midi_filename
    
    if not midi_path.exists():
        print(f"✗ File not found: {midi_path}")
        return
    
    audio_dir = Path("outputs/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = audio_dir / f"{midi_path.stem}.wav"
    
    print(f"Rendering: {midi_path.name}")
    
    if render_midi_to_wav(midi_path, output_path, soundfont_path):
        print(f"✓ Saved: {output_path}")
        print(f"\nPlaying: {output_path.name}")
        play_audio(output_path)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Render specific file
        render_specific_file(sys.argv[1])
    else:
        # Render all files
        render_and_listen()