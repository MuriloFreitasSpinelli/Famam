"""
Rule-based drum pattern generator.

Uses musical rules to generate realistic drum patterns for various genres.
Drum patterns follow standard conventions:
    - Kick on beats 1, 3 (or variations)
    - Snare on beats 2, 4 (backbeat)
    - Hi-hat on 8th or 16th notes
    - Crashes on downbeats of sections
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import IntEnum
import random
import muspy


class DrumNote(IntEnum):
    """General MIDI drum note numbers."""
    # Kick drums
    KICK = 36
    KICK_ALT = 35

    # Snares
    SNARE = 38
    SNARE_RIM = 37
    SNARE_SIDE = 40

    # Hi-hats
    HIHAT_CLOSED = 42
    HIHAT_PEDAL = 44
    HIHAT_OPEN = 46

    # Toms
    TOM_LOW = 45
    TOM_MID = 47
    TOM_HIGH = 50
    TOM_FLOOR = 41

    # Cymbals
    CRASH_1 = 49
    CRASH_2 = 57
    RIDE = 51
    RIDE_BELL = 53
    SPLASH = 55
    CHINA = 52


@dataclass
class DrumPattern:
    """A drum pattern definition."""
    name: str
    beats_per_bar: int
    subdivisions: int  # subdivisions per beat (2 = 8th notes, 4 = 16th notes)
    kick: List[int]    # positions where kick hits (0-indexed)
    snare: List[int]   # positions where snare hits
    hihat: List[int]   # positions where hi-hat hits
    accent_positions: List[int] = None  # positions for accents/crashes

    @property
    def total_positions(self) -> int:
        return self.beats_per_bar * self.subdivisions


# Pre-defined patterns for different styles
ROCK_PATTERNS = {
    "basic_rock": DrumPattern(
        name="Basic Rock",
        beats_per_bar=4,
        subdivisions=2,  # 8th notes
        kick=[0, 4],     # beats 1 and 3
        snare=[2, 6],    # beats 2 and 4
        hihat=[0, 1, 2, 3, 4, 5, 6, 7],  # all 8th notes
    ),
    "driving_rock": DrumPattern(
        name="Driving Rock",
        beats_per_bar=4,
        subdivisions=2,
        kick=[0, 3, 4, 6],  # syncopated kick
        snare=[2, 6],
        hihat=[0, 1, 2, 3, 4, 5, 6, 7],
    ),
    "half_time": DrumPattern(
        name="Half Time",
        beats_per_bar=4,
        subdivisions=2,
        kick=[0],
        snare=[4],  # snare on beat 3 for half-time feel
        hihat=[0, 1, 2, 3, 4, 5, 6, 7],
    ),
    "punk_rock": DrumPattern(
        name="Punk Rock",
        beats_per_bar=4,
        subdivisions=2,
        kick=[0, 1, 2, 3, 4, 5, 6, 7],  # four on the floor + more
        snare=[2, 6],
        hihat=[0, 1, 2, 3, 4, 5, 6, 7],
    ),
}

ALTERNATIVE_PATTERNS = {
    "alt_basic": DrumPattern(
        name="Alternative Basic",
        beats_per_bar=4,
        subdivisions=4,  # 16th notes
        kick=[0, 6, 8, 14],
        snare=[4, 12],
        hihat=[0, 2, 4, 6, 8, 10, 12, 14],  # 8th notes on 16th grid
    ),
    "alt_groove": DrumPattern(
        name="Alternative Groove",
        beats_per_bar=4,
        subdivisions=4,
        kick=[0, 5, 8, 10, 14],
        snare=[4, 12],
        hihat=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ),
    "alt_sparse": DrumPattern(
        name="Alternative Sparse",
        beats_per_bar=4,
        subdivisions=2,
        kick=[0, 5],
        snare=[2, 6],
        hihat=[0, 2, 4, 6],  # quarter notes
    ),
}


class DrumPatternGenerator:
    """
    Generate drum tracks using musical rules.

    Features:
        - Pre-defined patterns for various genres
        - Velocity humanization
        - Fill generation
        - Dynamic variation
    """

    def __init__(
        self,
        resolution: int = 24,
        default_velocity: int = 100,
        humanize: bool = True,
        humanize_amount: int = 15,
    ):
        """
        Initialize drum generator.

        Args:
            resolution: Ticks per beat (quarter note)
            default_velocity: Base velocity for hits
            humanize: Whether to add velocity variation
            humanize_amount: Max velocity deviation
        """
        self.resolution = resolution
        self.default_velocity = default_velocity
        self.humanize = humanize
        self.humanize_amount = humanize_amount

        # Combine all patterns
        self.patterns = {**ROCK_PATTERNS, **ALTERNATIVE_PATTERNS}

    def _get_velocity(self, base_velocity: int, is_accent: bool = False) -> int:
        """Get velocity with optional humanization."""
        vel = base_velocity

        if is_accent:
            vel = min(127, vel + 20)

        if self.humanize:
            variation = random.randint(-self.humanize_amount, self.humanize_amount)
            vel = max(1, min(127, vel + variation))

        return vel

    def _position_to_time(self, position: int, bar: int, pattern: DrumPattern) -> int:
        """Convert pattern position to absolute time in ticks."""
        ticks_per_subdivision = self.resolution // (pattern.subdivisions // 2) if pattern.subdivisions > 2 else self.resolution // 2
        ticks_per_bar = self.resolution * pattern.beats_per_bar
        return bar * ticks_per_bar + position * ticks_per_subdivision

    def generate_bar(
        self,
        pattern: DrumPattern,
        bar_number: int = 0,
        add_crash: bool = False,
        fill_positions: Optional[List[int]] = None,
    ) -> List[muspy.Note]:
        """
        Generate notes for a single bar.

        Args:
            pattern: Drum pattern to use
            bar_number: Bar index (for absolute timing)
            add_crash: Add crash cymbal on beat 1
            fill_positions: Positions to add fill notes

        Returns:
            List of muspy.Note objects
        """
        notes = []
        ticks_per_subdivision = self.resolution * 2 // pattern.subdivisions
        ticks_per_bar = self.resolution * pattern.beats_per_bar
        bar_start = bar_number * ticks_per_bar

        # Hi-hat
        for pos in pattern.hihat:
            time = bar_start + pos * ticks_per_subdivision
            is_downbeat = pos % pattern.subdivisions == 0
            vel = self._get_velocity(self.default_velocity - 10, is_accent=is_downbeat)
            notes.append(muspy.Note(
                time=time,
                pitch=DrumNote.HIHAT_CLOSED,
                duration=ticks_per_subdivision // 2,
                velocity=vel,
            ))

        # Kick
        for pos in pattern.kick:
            time = bar_start + pos * ticks_per_subdivision
            is_downbeat = pos == 0
            vel = self._get_velocity(self.default_velocity + 5, is_accent=is_downbeat)
            notes.append(muspy.Note(
                time=time,
                pitch=DrumNote.KICK,
                duration=ticks_per_subdivision,
                velocity=vel,
            ))

        # Snare
        for pos in pattern.snare:
            time = bar_start + pos * ticks_per_subdivision
            vel = self._get_velocity(self.default_velocity + 10)
            notes.append(muspy.Note(
                time=time,
                pitch=DrumNote.SNARE,
                duration=ticks_per_subdivision,
                velocity=vel,
            ))

        # Crash on beat 1 if requested
        if add_crash:
            notes.append(muspy.Note(
                time=bar_start,
                pitch=DrumNote.CRASH_1,
                duration=self.resolution * 2,
                velocity=self._get_velocity(self.default_velocity + 15, is_accent=True),
            ))

        # Fill notes
        if fill_positions:
            for pos in fill_positions:
                time = bar_start + pos * ticks_per_subdivision
                # Alternate between toms
                tom = random.choice([DrumNote.TOM_HIGH, DrumNote.TOM_MID, DrumNote.TOM_LOW])
                notes.append(muspy.Note(
                    time=time,
                    pitch=tom,
                    duration=ticks_per_subdivision,
                    velocity=self._get_velocity(self.default_velocity),
                ))

        return notes

    def generate_fill(
        self,
        pattern: DrumPattern,
        bar_number: int,
        fill_length: int = 4,  # number of subdivisions
    ) -> List[muspy.Note]:
        """Generate a drum fill at the end of a bar."""
        notes = []
        ticks_per_subdivision = self.resolution * 2 // pattern.subdivisions
        ticks_per_bar = self.resolution * pattern.beats_per_bar
        bar_start = bar_number * ticks_per_bar

        # Fill starts at end of bar
        fill_start = pattern.total_positions - fill_length

        # Create fill pattern (descending toms)
        toms = [DrumNote.TOM_HIGH, DrumNote.TOM_MID, DrumNote.TOM_LOW, DrumNote.TOM_FLOOR]

        for i in range(fill_length):
            pos = fill_start + i
            time = bar_start + pos * ticks_per_subdivision
            tom = toms[i % len(toms)]

            notes.append(muspy.Note(
                time=time,
                pitch=tom,
                duration=ticks_per_subdivision,
                velocity=self._get_velocity(self.default_velocity + 5),
            ))

        return notes

    def generate_track(
        self,
        num_bars: int = 8,
        pattern_name: str = "alt_basic",
        section_length: int = 4,
        add_fills: bool = True,
        add_crashes: bool = True,
    ) -> muspy.Track:
        """
        Generate a complete drum track.

        Args:
            num_bars: Number of bars to generate
            pattern_name: Name of pattern to use
            section_length: Bars per section (for crashes)
            add_fills: Whether to add fills before sections
            add_crashes: Whether to add crash cymbals

        Returns:
            muspy.Track object
        """
        if pattern_name not in self.patterns:
            pattern_name = "alt_basic"

        pattern = self.patterns[pattern_name]
        all_notes = []

        for bar in range(num_bars):
            is_section_start = bar % section_length == 0
            is_before_section = (bar + 1) % section_length == 0 and bar < num_bars - 1

            # Generate bar
            notes = self.generate_bar(
                pattern=pattern,
                bar_number=bar,
                add_crash=add_crashes and is_section_start,
            )
            all_notes.extend(notes)

            # Add fill before new section
            if add_fills and is_before_section:
                fill_notes = self.generate_fill(pattern, bar)
                all_notes.extend(fill_notes)

        # Sort by time
        all_notes.sort(key=lambda n: (n.time, n.pitch))

        return muspy.Track(
            program=0,
            is_drum=True,
            name="Drums",
            notes=all_notes,
        )

    def generate_intro(
        self,
        num_bars: int = 2,
        pattern_name: str = "alt_sparse",
    ) -> muspy.Track:
        """Generate a simpler intro pattern."""
        return self.generate_track(
            num_bars=num_bars,
            pattern_name=pattern_name,
            add_fills=False,
            add_crashes=True,
        )

    def list_patterns(self) -> List[str]:
        """List available pattern names."""
        return list(self.patterns.keys())

    def get_pattern_info(self, pattern_name: str) -> Optional[Dict]:
        """Get info about a pattern."""
        if pattern_name not in self.patterns:
            return None
        p = self.patterns[pattern_name]
        return {
            "name": p.name,
            "beats_per_bar": p.beats_per_bar,
            "subdivisions": p.subdivisions,
            "kick_hits": len(p.kick),
            "snare_hits": len(p.snare),
            "hihat_hits": len(p.hihat),
        }


def generate_alternative_rock_drums(
    num_bars: int = 8,
    resolution: int = 24,
    tempo: float = 120.0,
    humanize: bool = True,
) -> muspy.Music:
    """
    Convenience function to generate Alternative Rock drum track.

    Args:
        num_bars: Number of bars
        resolution: Ticks per beat
        tempo: BPM
        humanize: Add velocity variation

    Returns:
        muspy.Music object with drum track
    """
    generator = DrumPatternGenerator(
        resolution=resolution,
        humanize=humanize,
    )

    # Use alternative rock pattern
    track = generator.generate_track(
        num_bars=num_bars,
        pattern_name="alt_groove",
        section_length=4,
        add_fills=True,
        add_crashes=True,
    )

    return muspy.Music(
        resolution=resolution,
        tempos=[muspy.Tempo(time=0, qpm=tempo)],
        tracks=[track],
    )
