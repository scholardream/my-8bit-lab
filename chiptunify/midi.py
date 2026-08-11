"""Minimal, dependency-free Standard MIDI File (SMF) parser.

Supports format 0/1 files with PPQN (ticks-per-quarter-note) timing.
Everything chiptunify needs: note on/off events and the tempo map.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class Note:
    start: float      # seconds
    end: float        # seconds
    pitch: int        # MIDI note number 0-127
    velocity: int     # 1-127
    channel: int      # 0-15 (9 == drums)
    track: int


@dataclass
class MidiSong:
    notes: list       # list[Note], unsorted
    division: int     # ticks per quarter note
    tempos: list      # list[(tick, microseconds_per_quarter)]

    @property
    def duration(self) -> float:
        return max((n.end for n in self.notes), default=0.0)


class MidiError(ValueError):
    pass


def _read_vlq(data: bytes, pos: int):
    """Read a variable-length quantity. Returns (value, new_pos)."""
    value = 0
    while True:
        if pos >= len(data):
            raise MidiError("Unexpected end of data inside VLQ")
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            return value, pos


def _parse_track(data: bytes, track_idx: int):
    """Parse one MTrk chunk body.

    Returns (events, tempos) where events are
    (abs_tick, kind, channel, pitch, velocity, track_idx),
    kind in {"on", "off"}.
    """
    pos = 0
    tick = 0
    running_status = None
    events = []
    tempos = []

    while pos < len(data):
        delta, pos = _read_vlq(data, pos)
        tick += delta

        status = data[pos]
        if status < 0x80:
            if running_status is None:
                raise MidiError("Running status without prior status byte")
            status = running_status
        else:
            pos += 1
            if status < 0xF0:
                running_status = status

        # Meta events
        if status == 0xFF:
            meta_type = data[pos]
            pos += 1
            length, pos = _read_vlq(data, pos)
            payload = data[pos:pos + length]
            pos += length
            if meta_type == 0x51 and length == 3:  # set tempo
                us_per_quarter = int.from_bytes(payload, "big")
                tempos.append((tick, us_per_quarter))
            elif meta_type == 0x2F:  # end of track
                break

        # System exclusive
        elif status in (0xF0, 0xF7):
            length, pos = _read_vlq(data, pos)
            pos += length

        # Channel messages
        else:
            kind = status & 0xF0
            channel = status & 0x0F
            if kind in (0x80, 0x90):
                pitch = data[pos]
                velocity = data[pos + 1]
                pos += 2
                if kind == 0x90 and velocity > 0:
                    events.append((tick, "on", channel, pitch, velocity, track_idx))
                else:  # note off, or note on with velocity 0
                    events.append((tick, "off", channel, pitch, 0, track_idx))
            elif kind in (0xA0, 0xB0, 0xE0):
                pos += 2  # two data bytes
            elif kind in (0xC0, 0xD0):
                pos += 1  # one data byte
            else:
                raise MidiError(f"Unsupported event 0x{status:02X}")

    return events, tempos


def _build_tick_to_sec(tempos, division):
    """Return a function converting absolute ticks to seconds."""
    tempos = sorted(tempos)
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, 500000))  # default 120 BPM

    # Precompute segment boundaries: [(tick, sec_at_tick, us_per_quarter)]
    segments = []
    sec = 0.0
    last_tick = 0
    last_tempo = tempos[0][1]
    for tick, tempo in tempos:
        if tick > last_tick:
            sec += (tick - last_tick) * last_tempo / 1e6 / division
            last_tick = tick
        last_tempo = tempo
        segments.append((tick, sec, tempo))

    def tick_to_sec(t):
        for i in range(len(segments) - 1, -1, -1):
            seg_tick, seg_sec, tempo = segments[i]
            if t >= seg_tick:
                return seg_sec + (t - seg_tick) * tempo / 1e6 / division
        return 0.0

    return tick_to_sec


def parse_midi(source) -> MidiSong:
    """Parse an SMF file from a path or bytes. Returns a MidiSong."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        with open(source, "rb") as f:
            data = f.read()

    if data[:4] != b"MThd":
        raise MidiError("Not a MIDI file (missing MThd header)")

    header_len = struct.unpack(">I", data[4:8])[0]
    fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise MidiError("SMPTE time division is not supported (PPQN only)")

    pos = 8 + header_len
    all_events = []
    all_tempos = []
    for track_idx in range(ntracks):
        while pos < len(data) and data[pos:pos + 4] != b"MTrk":
            # Skip unknown chunk
            chunk_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
            pos += 8 + chunk_len
        if pos >= len(data):
            break
        track_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + track_len]
        pos += 8 + track_len
        events, tempos = _parse_track(body, track_idx)
        all_events.extend(events)
        all_tempos.extend(tempos)

    tick_to_sec = _build_tick_to_sec(all_tempos, division)

    # Pair note ons and offs
    active = {}   # (channel, pitch) -> (start_tick, velocity, track)
    notes = []
    max_tick = 0
    for tick, kind, channel, pitch, velocity, track in sorted(all_events, key=lambda e: e[0]):
        max_tick = max(max_tick, tick)
        key = (channel, pitch)
        if kind == "on":
            if key in active:  # hanging note — close it first
                s_tick, s_vel, s_track = active.pop(key)
                notes.append((s_tick, tick, pitch, s_vel, channel, s_track))
            active[key] = (tick, velocity, track)
        else:
            if key in active:
                s_tick, s_vel, s_track = active.pop(key)
                if tick > s_tick:
                    notes.append((s_tick, tick, pitch, s_vel, channel, s_track))
    # Close anything still hanging
    for (channel, pitch), (s_tick, s_vel, s_track) in active.items():
        if max_tick > s_tick:
            notes.append((s_tick, max_tick, pitch, s_vel, channel, s_track))

    song_notes = [
        Note(
            start=tick_to_sec(s),
            end=tick_to_sec(e),
            pitch=p,
            velocity=v,
            channel=c,
            track=t,
        )
        for (s, e, p, v, c, t) in notes
    ]
    return MidiSong(notes=song_notes, division=division, tempos=sorted(all_tempos))
