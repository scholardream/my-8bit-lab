"""Encode NES-MDB TX1 files into flat token binaries for training.

This mirrors nanoGPT's ``prepare.py``: every song is turned into a list of
token ids (wrapped in ``<S> ... <S>``), all songs of a split are concatenated
into one ``uint16`` numpy array, and the array is dumped to ``*.bin``.  At
training time we just memmap the file and sample random windows — no per-file
I/O on the hot path.

Usage::

    python -m nesgpt.prepare --data_dir LakhNES/data/nesmdb_tx1 \
        --out_dir nesgpt/data
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .vocab import Vocab, default_vocab_path

SPLITS = ("train", "valid", "test")


def encode_split(vocab: Vocab, pattern: str, out_path: str) -> tuple[int, int]:
    """Encode every file matching ``pattern`` into one concatenated array.

    Returns ``(n_files, n_tokens)``.  The array is saved at ``out_path``.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern!r}")

    chunks = []
    for p in paths:
        ids = vocab.encode_file(p, add_sos=True)
        if ids:
            chunks.append(np.asarray(ids, dtype=np.uint16))
    if not chunks:
        raise RuntimeError(f"all files empty for {pattern!r}")

    data = np.concatenate(chunks)
    data.tofile(out_path)
    return len(paths), int(data.size)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=None,
                    help="Directory containing train/valid/test of .tx1.txt "
                         "files (default: LakhNES/data/nesmdb_tx1)")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write train.bin/valid.bin/test.bin "
                         "(default: nesgpt/data)")
    ap.add_argument("--vocab", default=None,
                    help="Path to vocab file (default: LakhNES/data/tx1_vocab.txt)")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    data_dir = args.data_dir or os.path.join(repo_root, "LakhNES", "data",
                                             "nesmdb_tx1")
    out_dir = args.out_dir or os.path.join(repo_root, "nesgpt", "data")

    os.makedirs(out_dir, exist_ok=True)
    vocab = Vocab.from_file(args.vocab or default_vocab_path())
    print(f"vocab size: {len(vocab)}")

    total = 0
    for split in SPLITS:
        pattern = os.path.join(data_dir, split, "*.txt")
        out_path = os.path.join(out_dir, f"{split}.bin")
        n_files, n_tokens = encode_split(vocab, pattern, out_path)
        total += n_tokens
        print(f"{split:6s}: {n_files:5d} songs -> {n_tokens:9d} tokens "
              f"({out_path})")
    print(f"total tokens: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
