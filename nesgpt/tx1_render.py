"""Render a TX1 event file (NES-MDB or nesgpt-generated) to a WAV.

TX1 events map almost 1:1 onto the NES channel model already implemented in
``chiptunify``, so this is a thin translator: parse the events, build a
``NesArrangement``, and hand it to ``chiptunify.synth`` for synthesis.  No
pretty_midi / nesmdb / Python 2 required — just numpy.

Usage::

    python -m nesgpt.tx1_render song.tx1.txt -o song.wav
    python -m nesgpt.tx1_render song.tx1.txt --duty-p1 0.25 --rate 22050
"""

from __future__ import annotations

import argparse
import os

from chiptunify.arrange import NesArrangement, NesEvent
from chiptunify.synth import render_arrangement, write_wav

TX1_RATE = 44100  # NES-MDB TX1 waits are in samples at 44.1 kHz


def parse_tx1(events) -> NesArrangement:
    """Turn TX1 event strings into a chiptunify ``NesArrangement``.

    ``events`` may be a newline-separated string or an iterable of strings.
    Note-ons without a matching note-off (common in generated output) are
    closed at the end of the song rather than dropped.
    """
    if isinstance(events, (str, bytes)):
        lines = events.strip().splitlines()
    else:
        lines = list(events)

    arr = NesArrangement()
    active = {}  # instag -> (pitch, start_sample)
    samp = 0

    def close(instag: str, end_samp: int) -> None:
        cur = active.pop(instag, None)
        if cur is None:
            return
        pitch, start_samp = cur
        if end_samp > start_samp:
            e = NesEvent(start_samp / TX1_RATE, end_samp / TX1_RATE,
                         pitch, _volume(instag))
            getattr(arr, instag.lower()).append(e)

    def _volume(instag: str) -> int:
        # TX1 carries no velocity; triangle has no volume control anyway.
        return 15 if instag in ("P1", "P2", "NO") else 15

    for raw in lines:
        ev = raw.strip()
        if not ev or ev.startswith("#"):
            continue
        if ev.startswith("WT"):
            samp += int(ev[3:])
            continue
        parts = ev.split("_")
        if len(parts) < 2:
            continue
        instag, kind = parts[0], parts[1]
        if instag not in ("P1", "P2", "TR", "NO"):
            continue
        if kind == "NOTEON" and len(parts) >= 3:
            close(instag, samp)  # close any hanging note on this channel
            active[instag] = (int(parts[2]), samp)
        elif kind == "NOTEOFF":
            close(instag, samp)

    for instag in list(active):
        close(instag, samp)

    all_events = arr.p1 + arr.p2 + arr.tr + arr.no
    arr.duration = max((e.end for e in all_events), default=0.0)
    return arr


def render_tx1(events, out_path: str, sample_rate: int = TX1_RATE,
               duty_p1: float = 0.5, duty_p2: float = 0.25,
               quantize_pitch: bool = True) -> None:
    arr = parse_tx1(events)
    samples = render_arrangement(arr, sample_rate=sample_rate,
                                 duty_p1=duty_p1, duty_p2=duty_p2,
                                 quantize_pitch=quantize_pitch)
    write_wav(samples, out_path, sample_rate=sample_rate)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input .tx1.txt file")
    ap.add_argument("-o", "--output", help="output .wav path "
                    "(default: <input>.wav)")
    ap.add_argument("--rate", type=int, default=TX1_RATE)
    ap.add_argument("--duty-p1", type=float, default=0.5,
                    choices=[0.125, 0.25, 0.5, 0.75])
    ap.add_argument("--duty-p2", type=float, default=0.25,
                    choices=[0.125, 0.25, 0.5, 0.75])
    ap.add_argument("--no-quantize", action="store_true",
                    help="disable NES timer-period pitch quantization")
    args = ap.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as f:
        events = f.read()
    out = args.output or os.path.splitext(args.input)[0] + ".wav"
    render_tx1(events, out, sample_rate=args.rate,
               duty_p1=args.duty_p1, duty_p2=args.duty_p2,
               quantize_pitch=not args.no_quantize)
    print(f"rendered -> {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
