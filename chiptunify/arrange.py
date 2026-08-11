"""Arrange arbitrary MIDI notes onto the NES's 4 hardware channels.

The NES APU gives us exactly:
    P1, P2 : two pulse (square) waves, monophonic, 16 volume levels
    TR     : triangle wave, monophonic, no volume control (great for bass)
    NO     : noise channel, monophonic, 16 period settings (drums!)

Real NES composers worked inside these limits with tricks like rapid
arpeggios to fake chords. This module does the same automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .midi import MidiSong, Note

# Pitch ranges the NES handles comfortably (as used by NES-MDB / LakhNES)
PULSE_MIN, PULSE_MAX = 33, 108   # A1 .. C8
TRI_MIN, TRI_MAX = 21, 108       # A0 .. C8

DRUM_CHANNEL = 9                 # MIDI channel 10 (0-indexed 9)


@dataclass
class NesEvent:
    start: float     # seconds
    end: float       # seconds
    pitch: int       # MIDI note (noise channel: 1-16 = noise period index)
    volume: int      # 1-15 (ignored for triangle)


@dataclass
class NesArrangement:
    p1: list = field(default_factory=list)   # list[NesEvent]
    p2: list = field(default_factory=list)
    tr: list = field(default_factory=list)
    no: list = field(default_factory=list)
    duration: float = 0.0

    def stats(self) -> str:
        return (f"P1: {len(self.p1)} notes | P2: {len(self.p2)} notes | "
                f"TR: {len(self.tr)} notes | NO: {len(self.no)} hits | "
                f"{self.duration:.1f}s")


def _clamp_pitch(pitch: int, lo: int, hi: int) -> int:
    """Transpose by octaves until the note fits the NES range."""
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


def _velocity_to_volume(velocity: int) -> int:
    return max(1, min(15, 1 + round(14 * velocity / 127)))


def _group_sources(song: MidiSong):
    """Group melodic notes by (track, channel); drums go their own way."""
    sources = {}
    drums = []
    for n in song.notes:
        if n.channel == DRUM_CHANNEL:
            drums.append(n)
        else:
            sources.setdefault((n.track, n.channel), []).append(n)
    return sources, drums


def _assign_channels(sources):
    """Pick which MIDI source plays on which NES channel.

    Strategy: lowest-median-pitch source becomes the triangle bass line.
    The two most active remaining sources get P1 and P2; any leftovers are
    merged into P2 (the harmony channel).
    """
    if not sources:
        return {"p1": [], "p2": [], "tr": []}

    def median_pitch(notes):
        ps = sorted(n.pitch for n in notes)
        return ps[len(ps) // 2]

    ordered = sorted(sources.values(), key=lambda ns: median_pitch(ns))
    bass = ordered[0]
    rest = [ns for ns in sources.values() if ns is not bass]
    rest.sort(key=lambda ns: -len(ns))

    p1 = rest[0] if len(rest) > 0 else []
    p2 = list(rest[1]) if len(rest) > 1 else []
    for extra in rest[2:]:
        p2.extend(extra)

    return {"p1": list(p1), "p2": p2, "tr": list(bass)}


def _monophonize(notes, lo, hi, arp=True, chord_window=0.03):
    """Enforce one-note-at-a-time on a channel, with NES-style arpeggios.

    Notes starting within `chord_window` seconds count as a chord. With
    arp enabled the chord duration is sliced into rapid cycles through the
    chord tones — the classic trick that fakes polyphony on one channel.
    """
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    out = []

    i = 0
    while i < len(notes):
        chord = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j].start - notes[i].start < chord_window:
            chord.append(notes[j])
            j += 1

        chord_start = min(n.start for n in chord)
        chord_end = max(n.end for n in chord)
        # Never run into whatever comes next
        if j < len(notes):
            chord_end = min(chord_end, notes[j].start)
        if chord_end <= chord_start:
            i = j
            continue

        if arp and len(chord) > 1:
            # Arpeggiate: cycle through chord tones at ~60 Hz (NES frame rate)
            step = 1 / 60.0
            tones = sorted({_clamp_pitch(n.pitch, lo, hi) for n in chord})
            vol = _velocity_to_volume(max(n.velocity for n in chord))
            t = chord_start
            k = 0
            while t < chord_end - 1e-9:
                e = min(t + step, chord_end)
                out.append(NesEvent(t, e, tones[k % len(tones)], vol))
                t = e
                k += 1
        else:
            # Monophonic clip: each note ends when the next begins
            for k, n in enumerate(chord):
                pitch = _clamp_pitch(n.pitch, lo, hi)
                out.append(NesEvent(
                    n.start, min(n.end, chord_end), pitch,
                    _velocity_to_volume(n.velocity)))

        i = j

    # Final pass: strict overlap clipping
    out.sort(key=lambda e: e.start)
    for k in range(len(out) - 1):
        if out[k].end > out[k + 1].start:
            out[k].end = out[k + 1].start
    return [e for e in out if e.end - e.start > 1e-4]


def _drums_to_noise(drums):
    """Map drum notes onto the noise channel's 16 period settings.

    Low drums (kick-ish) -> long noise periods (low rumble),
    high drums (snare/hat-ish) -> short periods (bright sizzle).
    """
    events = []
    for n in sorted(drums, key=lambda n: n.start):
        if 35 <= n.pitch <= 41:        # kick region
            period_idx = 3
        elif n.pitch >= 42:            # hats/snare region
            period_idx = min(16, 1 + (n.pitch - 42) // 4)
        else:
            period_idx = 8
        # NES noise "notes" are short; cap drum length
        end = min(n.end, n.start + 0.25)
        events.append(NesEvent(n.start, end, period_idx,
                               _velocity_to_volume(n.velocity)))
    # Strict monophony on noise too
    for k in range(len(events) - 1):
        if events[k].end > events[k + 1].start:
            events[k].end = events[k + 1].start
    return [e for e in events if e.end - e.start > 1e-4]


def arrange_for_nes(song: MidiSong, arp: bool = True) -> NesArrangement:
    """Turn a MidiSong into per-channel NES event lists."""
    sources, drums = _group_sources(song)
    ch = _assign_channels(sources)

    arr = NesArrangement(
        p1=_monophonize(ch["p1"], PULSE_MIN, PULSE_MAX, arp=arp),
        p2=_monophonize(ch["p2"], PULSE_MIN, PULSE_MAX, arp=arp),
        tr=_monophonize(ch["tr"], TRI_MIN, TRI_MAX, arp=False),
        no=_drums_to_noise(drums),
    )
    all_events = arr.p1 + arr.p2 + arr.tr + arr.no
    arr.duration = max((e.end for e in all_events), default=0.0)
    return arr
