"""Command line interface: python -m chiptunify"""

from __future__ import annotations

import argparse
import os
import sys

from .arrange import arrange_for_nes
from .demo import demo_song
from .midi import MidiError, parse_midi
from .synth import SAMPLE_RATE, render_arrangement, write_wav


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="chiptunify",
        description="Turn any MIDI file into NES-style 8-bit chiptune audio.")
    parser.add_argument("input", nargs="?", help="Input .mid file")
    parser.add_argument("-o", "--output", help="Output .wav path")
    parser.add_argument("--demo", action="store_true",
                        help="Render the built-in demo song (Korobeiniki)")
    parser.add_argument("--no-arp", action="store_true",
                        help="Disable chord arpeggios (chords get clipped instead)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable NES timer-period pitch quantization")
    parser.add_argument("--duty-p1", type=float, default=0.5,
                        choices=[0.125, 0.25, 0.5, 0.75],
                        help="Duty cycle for pulse 1 / lead (default 0.5)")
    parser.add_argument("--duty-p2", type=float, default=0.25,
                        choices=[0.125, 0.25, 0.5, 0.75],
                        help="Duty cycle for pulse 2 / harmony (default 0.25)")
    parser.add_argument("--rate", type=int, default=SAMPLE_RATE,
                        help=f"Sample rate (default {SAMPLE_RATE})")
    args = parser.parse_args(argv)

    if args.demo:
        song = demo_song()
        out = args.output or "chiptunify_demo.wav"
    elif args.input:
        try:
            song = parse_midi(args.input)
        except (MidiError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        out = args.output or os.path.splitext(args.input)[0] + "_8bit.wav"
    else:
        parser.print_help()
        return 1

    arr = arrange_for_nes(song, arp=not args.no_arp)
    print(f"arranged  -> {arr.stats()}")

    samples = render_arrangement(
        arr, sample_rate=args.rate,
        duty_p1=args.duty_p1, duty_p2=args.duty_p2,
        quantize_pitch=not args.no_quantize)
    write_wav(samples, out, sample_rate=args.rate)
    print(f"synth     -> {len(samples) / args.rate:.1f}s of 8-bit goodness")
    print(f"saved     -> {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
