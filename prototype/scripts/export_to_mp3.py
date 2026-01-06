from pathlib import Path
import sys
import time
import wave
import pygame
import pygame.mixer
from pydub import AudioSegment
from mido import MidiFile, MidiTrack, Message

def install_dependencies():
    """Install required packages"""
    packages = ['pygame', 'pydub', 'mido']
    
    print("Checking dependencies...")
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✓ All dependencies installed")
    return True

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

def record_pygame_audio(midi_path, output_wav, duration_seconds=60):
    """
    Record pygame MIDI playback to WAV file
    This is a workaround for systems without FluidSynth
    """
    try:
        # Initialize pygame mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.music.load(str(midi_path))
        
        # Get MIDI duration (approximate)
        from mido import MidiFile
        mid = MidiFile(midi_path)
        duration = mid.length
        
        print(f"  → Recording MIDI playback ({duration:.1f} seconds)...")
        
        # Start recording
        import pyaudio
        import numpy as np
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        frames = []
        
        # Start playback
        pygame.mixer.music.play()
        
        # Record while playing
        while pygame.mixer.music.get_busy():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        # Stop stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        pygame.mixer.quit()
        
        # Save to WAV
        wf = wave.open(str(output_wav), 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return True
        
    except ImportError:
        print("  ⚠ PyAudio not available - using simpler method")
        return False
    except Exception as e:
        print(f"  ✗ Error recording: {e}")
        return False

def simple_midi_to_mp3(midi_path, output_path, guitar_type=25):
    """
    Simple MIDI to MP3 conversion using system MIDI synthesizer
    This method doesn't require FluidSynth but audio quality depends on system
    """
    try:
        print(f"  → Converting to guitar MIDI...")
        guitar_midi = convert_to_guitar(midi_path, guitar_type)
        
        print(f"  → Converting to MP3...")
        print(f"     Note: Using system MIDI synthesizer (quality depends on your system)")
        
        # Try using timidity if available (Linux/Mac)
        import subprocess
        temp_wav = output_path.parent / f"temp_{output_path.stem}.wav"
        
        try:
            # Try timidity first
            result = subprocess.run(
                ['timidity', str(guitar_midi), '-Ow', '-o', str(temp_wav)],
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"  → Converting WAV to MP3...")
                audio = AudioSegment.from_wav(str(temp_wav))
                audio.export(str(output_path), format='mp3', bitrate='192k')
                temp_wav.unlink()
                
                if guitar_midi != midi_path:
                    guitar_midi.unlink()
                
                return True
                
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # If timidity not available, try pygame method
        print(f"  ℹ Timidity not found, trying alternative method...")
        print(f"  ⚠ Note: This may not work well on all systems")
        print(f"  💡 Tip: For best results, install FluidSynth and use export_to_mp3.py")
        
        # Just copy the MIDI as a placeholder
        print(f"  → Creating MP3 from MIDI data...")
        
        # Actually, let's just inform the user
        print(f"  ✗ Cannot convert without FluidSynth or Timidity")
        print(f"  ℹ Guitar MIDI file created: {guitar_midi}")
        print(f"  💡 You can:")
        print(f"     1. Install FluidSynth and run export_to_mp3.py")
        print(f"     2. Use an online MIDI to MP3 converter")
        print(f"     3. Import the MIDI into a DAW (FL Studio, Ableton, etc.)")
        
        # Don't delete the guitar MIDI
        return False
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def export_all_to_mp3_simple(guitar_type=25, output_dir=None):
    """
    Export all MIDI files to MP3 (simplified method)
    """
    if not install_dependencies():
        return
    
    midi_dir = Path("outputs/generated")
    
    if not midi_dir.exists():
        print(f"✗ Directory not found: {midi_dir}")
        return
    
    if output_dir is None:
        output_dir = Path("outputs/mp3")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create a guitar_midi directory
    guitar_dir = Path("outputs/guitar_midi")
    guitar_dir.mkdir(parents=True, exist_ok=True)
    
    midi_files = sorted(midi_dir.glob("*.mid"))
    midi_files = [f for f in midi_files if not f.name.startswith("temp_")]
    
    if not midi_files:
        print("✗ No MIDI files found to convert")
        return
    
    print("\n" + "=" * 80)
    print(f"EXPORTING {len(midi_files)} MIDI FILES (Guitar Type: {guitar_type})")
    print("=" * 80)
    print(f"Guitar MIDI output: {guitar_dir}")
    print(f"MP3 output: {output_dir}")
    print()
    
    print("⚠ IMPORTANT:")
    print("This script requires FluidSynth for MP3 conversion.")
    print("If FluidSynth is not installed, guitar MIDI files will be created instead.")
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
    
    converted = 0
    guitar_midi_created = 0
    
    for i, midi_file in enumerate(midi_files, 1):
        output_name = f"{midi_file.stem}_{guitar_name}.mp3"
        output_path = output_dir / output_name
        
        guitar_midi_name = f"{midi_file.stem}_{guitar_name}.mid"
        guitar_midi_path = guitar_dir / guitar_midi_name
        
        print(f"[{i}/{len(midi_files)}] {midi_file.name}")
        
        # First, always create guitar MIDI
        temp_guitar = convert_to_guitar(midi_file, guitar_type)
        import shutil
        shutil.copy2(temp_guitar, guitar_midi_path)
        if temp_guitar != midi_file:
            temp_guitar.unlink()
        
        print(f"  ✓ Guitar MIDI saved: {guitar_midi_name}")
        guitar_midi_created += 1
        
        # Try to convert to MP3
        if simple_midi_to_mp3(guitar_midi_path, output_path, guitar_type):
            print(f"  ✓ MP3 saved: {output_name}")
            converted += 1
        
        print()
    
    print("=" * 80)
    print(f"✅ Export complete!")
    print(f"   Guitar MIDI files created: {guitar_midi_created} (in {guitar_dir})")
    print(f"   MP3 files converted: {converted} (in {output_dir})")
    
    if converted == 0:
        print("\n💡 TO CONVERT TO MP3:")
        print("   Option 1: Install FluidSynth and run export_to_mp3.py")
        print("   Option 2: Use the guitar MIDI files with any music software")
        print("   Option 3: Upload guitar MIDI files to an online converter")
    
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
    print("MIDI TO MP3 CONVERTER - SIMPLE VERSION")
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
    
    export_all_to_mp3_simple(guitar_type)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("\nUsage:")
            print("  python export_to_mp3_simple.py           # Interactive mode")
            print("  python export_to_mp3_simple.py 27        # Export with guitar type 27")
            print("  python export_to_mp3_simple.py --list    # Show guitar types")
        elif sys.argv[1] == "--list":
            show_guitar_types()
        else:
            try:
                guitar_type = int(sys.argv[1])
                export_all_to_mp3_simple(guitar_type)
            except ValueError:
                print("Invalid guitar type")
    else:
        main()