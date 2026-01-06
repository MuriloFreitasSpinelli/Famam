from pathlib import Path
import sys
import subprocess
import platform
from mido import MidiFile, MidiTrack, Message

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def install_python_dependencies():
    """Install required Python packages"""
    print("Checking Python dependencies...")
    
    # Install mido first
    try:
        import mido
        print("✓ mido installed")
    except ImportError:
        print("Installing mido...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mido"])
        print("✓ mido installed")
    
    # Check for ffmpeg before installing pydub
    if not check_ffmpeg():
        print("\n" + "=" * 80)
        print("⚠ FFmpeg is NOT installed!")
        print("=" * 80)
        print("\nFFmpeg is required to convert audio to MP3.")
        print("\nTO INSTALL FFMPEG ON WINDOWS:")
        print("\nOption 1 - Using Chocolatey (Recommended):")
        print("  1. Open PowerShell as Administrator")
        print("  2. Install Chocolatey (if not installed):")
        print("     Set-ExecutionPolicy Bypass -Scope Process -Force;")
        print("     [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;")
        print("     iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))")
        print("  3. Install ffmpeg:")
        print("     choco install ffmpeg")
        print("  4. Restart this script")
        print("\nOption 2 - Manual Installation:")
        print("  1. Download ffmpeg from: https://www.gyan.dev/ffmpeg/builds/")
        print("  2. Extract the zip file")
        print("  3. Add the 'bin' folder to your PATH")
        print("  4. Restart your terminal")
        print("\nOption 3 - Use the MIDI files directly:")
        print("  This script will create guitar MIDI files that you can:")
        print("  - Import into any DAW (FL Studio, Ableton, GarageBand)")
        print("  - Convert using online tools")
        print("  - Play with any MIDI player")
        print("=" * 80)
        
        choice = input("\nContinue without MP3 conversion (MIDI only)? (y/N): ").strip().lower()
        if choice != 'y':
            sys.exit(1)
        return False
    else:
        print("✓ ffmpeg installed")
    
    # Install pydub
    try:
        from pydub import AudioSegment
        print("✓ pydub installed")
        return True
    except ImportError:
        print("Installing pydub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
        try:
            from pydub import AudioSegment
            print("✓ pydub installed")
            return True
        except Exception as e:
            print(f"⚠ Warning: pydub installation issue: {e}")
            return False

def convert_to_guitar(midi_path, guitar_type=25):
    """Convert MIDI to use guitar instrument"""
    temp_path = midi_path.parent / f"temp_guitar_{midi_path.name}"
    
    try:
        mid = MidiFile(midi_path)
        new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
        
        for i, track in enumerate(mid.tracks):
            new_track = MidiTrack()
            program_added = False
            
            for msg in track:
                if msg.type == 'program_change':
                    new_track.append(msg.copy(program=guitar_type))
                    program_added = True
                elif msg.type == 'note_on' and not program_added:
                    new_track.append(Message('program_change', program=guitar_type, time=0))
                    new_track.append(msg)
                    program_added = True
                else:
                    new_track.append(msg)
            
            new_mid.tracks.append(new_track)
        
        new_mid.save(temp_path)
        return temp_path
        
    except Exception as e:
        print(f"Error converting to guitar: {e}")
        return midi_path

def check_midi_converter():
    """Check for available MIDI to audio converters"""
    converters = []
    
    # Check for timidity
    try:
        result = subprocess.run(['timidity', '--version'], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0:
            converters.append('timidity')
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check for fluidsynth
    try:
        result = subprocess.run(['fluidsynth', '--version'], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0:
            converters.append('fluidsynth')
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return converters

def midi_to_wav_timidity(midi_path, wav_path):
    """Convert MIDI to WAV using timidity"""
    try:
        result = subprocess.run(
            ['timidity', str(midi_path), '-Ow', '-o', str(wav_path)],
            capture_output=True,
            timeout=60,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  ✗ Timidity error: {e}")
        return False

def wav_to_mp3_ffmpeg(wav_path, mp3_path):
    """Convert WAV to MP3 using ffmpeg directly"""
    try:
        result = subprocess.run([
            'ffmpeg', '-i', str(wav_path),
            '-codec:a', 'libmp3lame',
            '-b:a', '192k',
            '-y',  # Overwrite output file
            str(mp3_path)
        ], capture_output=True, timeout=60, text=True)
        
        return result.returncode == 0
    except Exception as e:
        print(f"  ✗ FFmpeg error: {e}")
        return False

def export_midi_files(guitar_type=25, output_dir=None):
    """
    Export MIDI files to guitar MIDI and optionally to MP3
    """
    # Check dependencies
    can_convert_mp3 = install_python_dependencies()
    converters = check_midi_converter()
    
    if not converters:
        print("\n⚠ No MIDI converter found (timidity/fluidsynth)")
        print("Only guitar MIDI files will be created.")
    
    midi_dir = Path("outputs/generated")
    
    if not midi_dir.exists():
        print(f"✗ Directory not found: {midi_dir}")
        return
    
    # Set up output directories
    guitar_midi_dir = Path("outputs/guitar_midi")
    guitar_midi_dir.mkdir(parents=True, exist_ok=True)
    
    mp3_dir = None
    if can_convert_mp3 and converters:
        if output_dir is None:
            mp3_dir = Path("outputs/mp3")
        else:
            mp3_dir = Path(output_dir)
        mp3_dir.mkdir(parents=True, exist_ok=True)
    
    # Get MIDI files
    midi_files = sorted(midi_dir.glob("*.mid"))
    midi_files = [f for f in midi_files if not f.name.startswith("temp_")]
    
    if not midi_files:
        print("✗ No MIDI files found to convert")
        return
    
    print("\n" + "=" * 80)
    print(f"EXPORTING {len(midi_files)} MIDI FILES")
    print("=" * 80)
    print(f"Guitar type: {guitar_type}")
    print(f"Guitar MIDI output: {guitar_midi_dir}")
    if mp3_dir:
        print(f"MP3 output: {mp3_dir}")
        print(f"Converter: {converters[0]}")
    else:
        print("MP3 conversion: Not available")
    print()
    
    guitar_names = {
        24: "Acoustic_Nylon",
        25: "Acoustic_Steel",
        26: "Electric_Jazz",
        27: "Electric_Clean",
        28: "Electric_Muted",
        29: "Overdriven",
        30: "Distortion",
        31: "Harmonics"
    }
    
    guitar_name = guitar_names.get(guitar_type, "Guitar")
    
    midi_created = 0
    mp3_created = 0
    
    for i, midi_file in enumerate(midi_files, 1):
        print(f"[{i}/{len(midi_files)}] {midi_file.name}")
        
        # Create guitar MIDI
        guitar_midi_name = f"{midi_file.stem}_{guitar_name}.mid"
        guitar_midi_path = guitar_midi_dir / guitar_midi_name
        
        temp_guitar = convert_to_guitar(midi_file, guitar_type)
        import shutil
        shutil.copy2(temp_guitar, guitar_midi_path)
        if temp_guitar != midi_file:
            temp_guitar.unlink()
        
        print(f"  ✓ Guitar MIDI: {guitar_midi_name}")
        midi_created += 1
        
        # Try to convert to MP3
        if mp3_dir and converters:
            mp3_name = f"{midi_file.stem}_{guitar_name}.mp3"
            mp3_path = mp3_dir / mp3_name
            temp_wav = mp3_dir / f"temp_{midi_file.stem}.wav"
            
            success = False
            
            if 'timidity' in converters:
                print(f"  → Converting to WAV...")
                if midi_to_wav_timidity(guitar_midi_path, temp_wav):
                    print(f"  → Converting to MP3...")
                    if wav_to_mp3_ffmpeg(temp_wav, mp3_path):
                        print(f"  ✓ MP3: {mp3_name}")
                        mp3_created += 1
                        success = True
                    
                    # Clean up temp WAV
                    if temp_wav.exists():
                        temp_wav.unlink()
            
            if not success:
                print(f"  ✗ MP3 conversion failed")
        
        print()
    
    print("=" * 80)
    print(f"✅ Export complete!")
    print(f"   Guitar MIDI files: {midi_created} (in {guitar_midi_dir})")
    if mp3_dir:
        print(f"   MP3 files: {mp3_created} (in {mp3_dir})")
    
    if mp3_created == 0 and midi_created > 0:
        print("\n💡 NEXT STEPS TO GET MP3 FILES:")
        print("   Option 1: Use the guitar MIDI files in your favorite DAW")
        print("   Option 2: Upload to online converter (e.g., https://www.onlineconverter.com/midi-to-mp3)")
        print("   Option 3: Install timidity:")
        if platform.system() == "Windows":
            print("      - Not easily available on Windows")
            print("      - Recommend using DAW or online converter")
        else:
            print("      sudo apt-get install timidity  # Linux")
            print("      brew install timidity          # macOS")
    
    print("=" * 80)

def show_guitar_types():
    """Display available guitar types"""
    print("\n" + "=" * 70)
    print("GUITAR TYPES (General MIDI)")
    print("=" * 70)
    print("\nACOUSTIC:")
    print("  24 - Acoustic Guitar (nylon)")
    print("  25 - Acoustic Guitar (steel) [DEFAULT]")
    print("\nELECTRIC:")
    print("  26 - Electric Guitar (jazz)")
    print("  27 - Electric Guitar (clean)")
    print("  28 - Electric Guitar (muted)")
    print("  29 - Overdriven Guitar")
    print("  30 - Distortion Guitar")
    print("  31 - Guitar Harmonics")
    print("=" * 70)

def main():
    """Interactive main function"""
    print("\n" + "=" * 80)
    print("MIDI TO GUITAR CONVERTER (Windows Compatible)")
    print("=" * 80)
    
    show_guitar_types()
    
    guitar_input = input("\nSelect guitar type (24-31) [default: 25]: ").strip()
    guitar_type = 25
    
    if guitar_input:
        try:
            guitar_type = int(guitar_input)
            if not 24 <= guitar_type <= 31:
                print("Invalid type, using default (25)")
                guitar_type = 25
        except ValueError:
            print("Invalid input, using default (25)")
    
    print(f"\n🎸 Guitar Type: {guitar_type}")
    
    confirm = input("\nProceed? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("Cancelled")
        return
    
    export_midi_files(guitar_type)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("\nUsage:")
            print("  python export_to_mp3_windows.py           # Interactive mode")
            print("  python export_to_mp3_windows.py 27        # Export with guitar type 27")
            print("  python export_to_mp3_windows.py --list    # Show guitar types")
        elif sys.argv[1] == "--list":
            show_guitar_types()
        else:
            try:
                guitar_type = int(sys.argv[1])
                export_midi_files(guitar_type)
            except ValueError:
                print("Invalid guitar type")
    else:
        main()