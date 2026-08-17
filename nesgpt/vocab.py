"""Vocabulary for NES-MDB TX1 event sequences.

A TX1 file is one musical event per line, e.g.::

    WT_466
    P1_NOTEON_65
    TR_NOTEON_62
    WT_17188
    P1_NOTEOFF

The vocabulary is fixed (not learned from data): it mirrors the
LakhNES TX1 vocabulary.  Index 0 is the special ``<S>`` token, which
serves double duty as the start-of-song marker and the separator
between songs (LakhNES's ``add_double_eos`` wraps every song in
``<S> ... <S>``).
"""

from __future__ import annotations

import os
from bisect import bisect_left as _bisect_left

SOS = "<S>"

# Event categories and their numeric ranges (identical to LakhNES TX1).
_WAITS = None          # filled by _build_waits()
INSTRUMENT_RANGES = {
    "P1": (33, 108),   # pulse 1, MIDI notes
    "P2": (33, 108),   # pulse 2
    "TR": (21, 108),   # triangle
    "NO": (1, 16),     # noise: period index 1..16 (not a MIDI note)
}


def _quantize_wait(wait: int) -> int:
    """Round a wait (in 44100 Hz samples) to the coarser grid used by the vocab.

    Below 100 every value is kept; then the grid widens to 10s, 100s, 1000s.
    Identical to LakhNES ``tx1_vocab_gen.quantize_wait``.
    """
    wait = min(wait, 100000)
    if wait > 10000:
        wait = 1000 * int(round(float(wait) / 1000) + 1e-4)
    elif wait > 1000:
        wait = 100 * int(round(float(wait) / 100) + 1e-4)
    elif wait > 100:
        wait = 10 * int(round(float(wait) / 10) + 1e-4)
    return wait


def _build_wait_symbols() -> list[str]:
    """The distinct WT_* symbols, in order (matches tx1_vocab.txt)."""
    symbols: list[str] = []
    last = None
    for i in range(1, 100001):
        w = _quantize_wait(i)
        if last is None or w != last:
            symbols.append(f"WT_{w}")
            last = w
    return symbols


def build_symbols() -> list[str]:
    """Full ordered symbol list (without the leading ``<S>``)."""
    symbols = _build_wait_symbols()
    for ins in ("P1", "P2", "TR", "NO"):
        symbols.append(f"{ins}_NOTEOFF")
        lo, hi = INSTRUMENT_RANGES[ins]
        for n in range(lo, hi + 1):
            symbols.append(f"{ins}_NOTEON_{n}")
    return symbols


def default_vocab_path() -> str:
    """Resolve the checked-in TX1 vocab file relative to the repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "LakhNES", "data",
                                        "tx1_vocab.txt"))


def load_symbols(path: str | None = None) -> list[str]:
    """Read the symbol list from a vocab file (one symbol per line).

    Accepts the LakhNES format where a line may be ``index,symbol``; we keep
    only the part after the last comma.
    """
    path = path or default_vocab_path()
    symbols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                symbols.append(line.split(",")[-1])
    return symbols


class Vocab:
    """Bidirectional mapping between TX1 symbols and integer ids.

    ``idx2sym[0] == "<S>"``; the remaining 630 ids map 1:1 to the lines of
    ``tx1_vocab.txt``.  Use ``Vocab.from_file()`` to build from the checked-in
    file, or ``Vocab.from_symbols()`` to synthesize the same vocabulary in
    memory (no file required).
    """

    def __init__(self, symbols: list[str]):
        self.idx2sym: list[str] = [SOS] + list(symbols)
        self.sym2idx: dict[str, int] = {s: i for i, s in
                                        enumerate(self.idx2sym)}
        # Valid wait amounts, ascending — used to snap raw WT_* values in the
        # data (e.g. WT_466) to the nearest quantized vocab entry (WT_470).
        self.wait_amts: list[int] = sorted(
            int(s[3:]) for s in self.idx2sym if s.startswith("WT_"))

    @classmethod
    def from_file(cls, path: str | None = None) -> "Vocab":
        return cls(load_symbols(path))

    @classmethod
    def from_symbols(cls) -> "Vocab":
        return cls(build_symbols())

    def __len__(self) -> int:
        return len(self.idx2sym)

    @property
    def vocab_size(self) -> int:
        return len(self)

    def get_idx(self, sym: str) -> int:
        """Map a symbol to its id.

        Raw waits (any ``WT_<n>`` not in the fixed vocab) snap to the nearest
        quantized wait amount — the same behaviour as LakhNES's ``Vocab.get_idx``.
        Anything else must already be in the vocab.
        """
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        if sym.startswith("WT_"):
            wait = int(sym[3:])
            # nearest wait amount; ties break toward the smaller value
            i = _bisect_left(self.wait_amts, wait)
            candidates = self.wait_amts[max(0, i - 1):i + 1]
            closest = min(candidates, key=lambda w: abs(w - wait))
            return self.sym2idx[f"WT_{closest}"]
        raise KeyError(f"unknown symbol {sym!r}")

    def encode(self, symbols: list[str]) -> list[int]:
        return [self.get_idx(s) for s in symbols]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.idx2sym[i] for i in ids]

    def encode_file(self, path: str, add_sos: bool = True) -> list[int]:
        """Encode one TX1 file into a token list.

        With ``add_sos`` (the default) the list is ``[SOS, *events, SOS]``,
        matching LakhNES's per-song ``<S> ... <S>`` wrapping.
        """
        with open(path, "r", encoding="utf-8") as f:
            events = [line.strip() for line in f if line.strip()]
        ids = self.encode(events)
        if add_sos:
            return [0] + ids + [0]
        return ids
