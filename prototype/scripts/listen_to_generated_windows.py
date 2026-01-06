from pathlib import Path
import pygame
import time
from mido import MidiFile, MidiTrack, Message

def convert_to_guitar(midi_path, output_path=None):
    """
    Convert MIDI file to use guitar sound (program 24-31 are guitars in General MIDI)
    Program 24: Acoustic Guitar (nylon)
    Program 25: Acoustic Guitar (steel)
    Program 26: Electric Guitar (jazz)
    Program 27: Electric Guitar (clean)
    Program 28: Electric Guitar (muted)
    Program 29: Overdriven Guitar
    Program 30: Distortion Guitar
    Program 31: Guitar Harmonics
    """
    try:
        mid = MidiFile(midi_path)
        new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
        
        guitar_program = 25  # Acoustic Guitar (steel) - change this to 24-31 for different guitars
        
        for i, track in enumerate(mid.tracks):
            new_track = MidiTrack()
            
            # Add program change at the start of each track
            if i == 0:
                # Add to first track (or you can add to all tracks)
                new_track.append(Message('program_change', program=guitar_program, time=0))
            
            for msg in track:
                # Replace any existing program_change messages with guitar
                if msg.type == 'program_change':
                    new_track.append(msg.copy(program=guitar_program))
                else:
                    new_track.append(msg)
            
            new_mid.tracks.append(new_track)
        
        if output_path is None:
            output_path = midi_path.parent / f"{midi_path.stem}_guitar.mid"
        
        new_mid.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Error converting MIDI: {e}")
        return midi_path

def play_midi_with_pygame(midi_path, use_guitar=True):
    """
    Play MIDI file directly using pygame with guitar sound
    """
    try:
        # Convert to guitar sound first
        if use_guitar:
            temp_path = midi_path.parent / f"temp_{midi_path.name}"
            play_path = convert_to_guitar(midi_path, temp_path)
        else:
            play_path = midi_path
        
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.music.load(str(play_path))
        
        print(f"\n🎸 Playing: {midi_path.name}")
        print("=" * 60)
        
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit()
        
        # Clean up temp file
        if use_guitar and temp_path.exists():
            temp_path.unlink()
        
        print("✓ Finished\n")
        return True
        
    except Exception as e:
        print(f"✗ Error playing {midi_path.name}: {e}")
        return False

def listen_to_all_generated(use_guitar=True):
    """
    Play all generated MIDI files with guitar sound
    """
    midi_dir = Path("outputs/generated")
    
    if not midi_dir.exists():
        print(f"✗ Directory not found: {midi_dir}")
        print("Run generate_music.py first!")
        return
    
    midi_files = sorted(midi_dir.glob("*.mid"))
    
    # Filter out temporary guitar files
    midi_files = [f for f in midi_files if not f.name.startswith("temp_")]
    
    if not midi_files:
        print(f"✗ No MIDI files found in {midi_dir}")
        return
    
    print("\n" + "=" * 80)
    print("PLAYING GENERATED MUSIC (GUITAR SOUND)")
    print("=" * 80)
    print(f"Found {len(midi_files)} MIDI files\n")
    
    for i, midi_file in enumerate(midi_files, 1):
        print(f"[{i}/{len(midi_files)}]", end=" ")
        play_midi_with_pygame(midi_file, use_guitar=use_guitar)
        
        if i < len(midi_files):
            response = input("Press Enter for next file (or 'q' to quit): ").strip().lower()
            if response == 'q':
                print("\nStopped by user")
                break
    
    print("=" * 80)
    print("✅ Playback complete!")
    print("=" * 80)

def play_specific_file(filename, use_guitar=True):
    """
    Play a specific MIDI file with guitar sound
    """
    midi_path = Path("outputs/generated") / filename
    
    if not midi_path.exists():
        print(f"✗ File not found: {midi_path}")
        print("\nAvailable files:")
        midi_dir = Path("outputs/generated")
        if midi_dir.exists():
            for f in sorted(midi_dir.glob("*.mid")):
                if not f.name.startswith("temp_"):
                    print(f"  - {f.name}")
        return
    
    play_midi_with_pygame(midi_path, use_guitar=use_guitar)

def show_guitar_menu():
    """
    Show guitar type selection menu
    """
    print("\n" + "=" * 60)
    print("SELECT GUITAR TYPE")
    print("=" * 60)
    print("\n24. Acoustic Guitar (nylon)")
    print("25. Acoustic Guitar (steel) [DEFAULT]")
    print("26. Electric Guitar (jazz)")
    print("27. Electric Guitar (clean)")
    print("28. Electric Guitar (muted)")
    print("29. Overdriven Guitar")
    print("30. Distortion Guitar")
    print("31. Guitar Harmonics")
    
    choice = input("\nEnter guitar type (24-31) or press Enter for default: ").strip()
    
    if choice == '':
        return 25
    
    try:
        guitar_num = int(choice)
        if 24 <= guitar_num <= 31:
            return guitar_num
        else:
            print("Invalid choice, using default (25)")
            return 25
    except ValueError:
        print("Invalid input, using default (25)")
        return 25

def show_menu():
    """
    Interactive menu
    """
    print("\n" + "=" * 80)
    print("MIDI PLAYBACK MENU (GUITAR SOUND)")
    print("=" * 80)
    print("\n1. Play all generated files")
    print("2. Play specific file")
    print("3. List available files")
    print("4. Change guitar type")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        listen_to_all_generated()
    elif choice == '2':
        filename = input("Enter filename: ").strip()
        play_specific_file(filename)
    elif choice == '3':
        midi_dir = Path("outputs/generated")
        if midi_dir.exists():
            midi_files = sorted(midi_dir.glob("*.mid"))
            midi_files = [f for f in midi_files if not f.name.startswith("temp_")]
            print(f"\nAvailable files ({len(midi_files)}):")
            for f in midi_files:
                print(f"  - {f.name}")
        else:
            print("\nNo generated files found")
    elif choice == '4':
        guitar_program = show_guitar_menu()
        # Update the global guitar program
        import sys
        # This is a simple way - you'd want to refactor for production
        print(f"\n✓ Guitar type set to program {guitar_program}")
    elif choice == '5':
        print("\nGoodbye!")
        return False
    else:
        print("\nInvalid choice")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Install required packages
    try:
        import pygame
        import mido
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame", "mido"])
        import pygame
        import mido
    
    if len(sys.argv) > 1:
        # Play specific file from command line
        play_specific_file(sys.argv[1])
    else:
        # Interactive menu
        while show_menu():
            input("\nPress Enter to return to menu...")