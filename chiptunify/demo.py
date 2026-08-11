"""Built-in demo song: Korobeiniki (the Tetris theme).

A 19th-century Russian folk tune — public domain, and spiritually the
most 8-bit melody ever written. Built in-memory so you can test the full
pipeline without hunting for a MIDI file:

    python -m chiptunify --demo -o tetris.wav
"""

from __future__ import annotations

from .midi import MidiSong, Note

BEAT = 0.5          # seconds per beat (120 BPM)
A4, B4, C5, D5, E5, F5, G5, A5 = 69, 71, 72, 74, 76, 77, 79, 81

# (pitch, beats) — part A of Korobeiniki
MELODY = [
    (E5, 1), (B4, .5), (C5, .5), (D5, 1), (C5, .5), (B4, .5),
    (A4, 1), (A4, .5), (C5, .5), (E5, 1), (D5, .5), (C5, .5),
    (B4, 1.5), (C5, .5), (D5, 1), (E5, 1),
    (C5, 1), (A4, 1), (A4, 1), (None, 1),
    (D5, 1.5), (F5, .5), (A5, 1), (G5, .5), (F5, .5),
    (E5, 1.5), (C5, .5), (E5, 1), (D5, .5), (C5, .5),
    (B4, 1), (B4, .5), (C5, .5), (D5, 1), (E5, 1),
    (C5, 1), (A4, 1), (A4, 1), (None, 1),
]

# One root per bar (8 bars), octave bass pattern root/fifth
BASS_ROOTS = [45, 45, 40, 45, 50, 48, 40, 45]   # A2 A2 E2 A2 D3 C3 E2 A2

KICK, HAT = 36, 42


def demo_song() -> MidiSong:
    notes = []

    # Melody (channel 0)
    t = 0.0
    for pitch, beats in MELODY:
        dur = beats * BEAT
        if pitch is not None:
            notes.append(Note(t, t + dur * 0.92, pitch, 100, 0, 0))
        t += dur
    total = t

    # Final chord (A4+C5+E5) to show off the arpeggiator
    for p in (A4, C5, E5):
        notes.append(Note(total, total + 2 * BEAT, p, 110, 0, 0))
    total += 2 * BEAT

    # Bass (channel 1): root/fifth octave hops, 2 notes per bar
    bar = 4 * BEAT
    for i, root in enumerate(BASS_ROOTS):
        t0 = i * bar
        notes.append(Note(t0, t0 + 2 * BEAT * 0.9, root, 90, 1, 1))
        notes.append(Note(t0 + 2 * BEAT, t0 + bar * 0.95, root + 7, 90, 1, 1))

    # Drums (channel 9): kick on 1 & 3, hat on every eighth
    n_bars = int(total / bar) + 1
    for b in range(n_bars):
        t0 = b * bar
        notes.append(Note(t0, t0 + 0.08, KICK, 120, 9, 2))
        notes.append(Note(t0 + 2 * BEAT, t0 + 2 * BEAT + 0.08, KICK, 120, 9, 2))
        for eighth in range(8):
            ts = t0 + eighth * BEAT / 2
            vel = 90 if eighth % 2 == 0 else 60
            notes.append(Note(ts, ts + 0.05, HAT, vel, 9, 2))

    return MidiSong(notes=notes, division=480, tempos=[(0, 500000)])
