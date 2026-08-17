"""Generate a TX1 chiptune from a trained checkpoint.

Loads ``ckpt.pt``, seeds generation with the ``<S>`` token, samples
autoregressively, and writes the decoded events as a ``.tx1.txt`` file (the
same format NES-MDB uses).  Pipe the result through ``nesgpt.tx1_render`` to
hear it, e.g.::

    python -m nesgpt.sample --ckpt nesgpt/out/ckpt.pt \
        --out generated/0.tx1.txt --num 1
    python -m nesgpt.tx1_render generated/0.tx1.txt -o generated/0.wav
"""

from __future__ import annotations

import argparse
import os

import torch

from .model import GPT
from .vocab import Vocab, SOS, default_vocab_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to ckpt.pt saved by nesgpt.train")
    ap.add_argument("--out_dir", type=str, default="generated",
                    help="output directory (default: ./generated)")
    ap.add_argument("--num", type=int, default=1,
                    help="number of songs to generate")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--vocab", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = GPT.from_pretrained_ckpt(args.ckpt, device=device)
    model.to(device)
    model.eval()

    vocab = Vocab.from_file(args.vocab or default_vocab_path())
    assert vocab.vocab_size == model.config.vocab_size, \
        f"vocab size mismatch: vocab={vocab.vocab_size}, model={model.config.vocab_size}"

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.num):
        idx = torch.tensor([[0]], dtype=torch.long, device=device)  # <S>
        out = model.generate(idx, args.max_new_tokens,
                             temperature=args.temperature,
                             top_k=args.top_k,
                             stop_token=0)
        symbols = vocab.decode(out[0].tolist())
        # Drop the leading <S>; keep any trailing <S> as the song terminator.
        body = [s for s in symbols[1:] if s != SOS]
        out_path = os.path.join(args.out_dir, f"{i}.tx1.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(body))
            if body:
                f.write("\n")
        print(f"{i}: {len(body)} events -> {out_path}")

    print("done. render with: python -m nesgpt.tx1_render <file> -o <out.wav>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
