"""chiptunify — turn any MIDI into NES-style 8-bit chiptune audio.

Zero dependencies beyond numpy: pure-Python SMF parsing, an NES channel
arranger (2 pulse + triangle + noise), and a numpy APU-flavoured synth.
"""

__version__ = "0.1.0"

from .midi import MidiSong, Note, parse_midi
from .arrange import arrange_for_nes, NesArrangement
from .synth import render_arrangement
