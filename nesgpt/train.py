"""Train a nanoGPT-style model on NES-MDB TX1 event sequences.

Usage (run from the repo root)::

    python -m nesgpt.prepare            # first: build train/valid/test.bin
    python -m nesgpt.train --device cpu --max_iters 2000

The data files are memory-mapped, so the hot loop just samples random
contiguous windows of ``block_size`` tokens — no per-song I/O.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from .model import GPT, GPTConfig


def load_bin(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(split_data: dict, split: str, block_size: int,
              batch_size: int, device: str):
    data = split_data[split]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64))
                     for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
                     for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, split_data, split, block_size, batch_size, device,
                  eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(split_data, split, block_size, batch_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # data
    ap.add_argument("--data_dir", type=str, default=None,
                    help="dir with train.bin/valid.bin (default: nesgpt/data)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="checkpoint dir (default: nesgpt/out)")
    # model
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vocab_size", type=int, default=631)
    # training
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--warmup_iters", type=int, default=200)
    ap.add_argument("--lr_decay_iters", type=int, default=None,
                    help="iters over which to cosine-decay LR; defaults to max_iters")
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # system
    ap.add_argument("--device", type=str, default="auto",
                    help="cuda / cpu / auto")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                    help="torch.compile the model (needs triton on cuda)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    data_dir = args.data_dir or os.path.join(repo_root, "nesgpt", "data")
    out_dir = args.out_dir or os.path.join(repo_root, "nesgpt", "out")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # ---- data ------------------------------------------------------------
    split_data = {
        "train": load_bin(os.path.join(data_dir, "train.bin")),
        "val": load_bin(os.path.join(data_dir, "val.bin")),
    }

    # ---- model -----------------------------------------------------------
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    model = GPT(config)
    model.to(device)
    if args.compile and device_type == "cuda":
        model = torch.compile(model)  # pragma: no cover

    lr_decay_iters = args.lr_decay_iters or args.max_iters
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                           (args.beta1, args.beta2), device_type)

    # ---- training loop ---------------------------------------------------
    t0 = time.time()
    running_mfu = -1.0
    best_val = float("inf")

    for it in range(args.max_iters):
        # cosine LR with linear warmup
        if it < args.warmup_iters:
            lr = args.learning_rate * (it + 1) / args.warmup_iters
        elif it > lr_decay_iters:
            lr = args.min_lr
        else:
            decay_ratio = (it - args.warmup_iters) / (lr_decay_iters - args.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = args.min_lr + coeff * (args.learning_rate - args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            val_loss = estimate_loss(model, split_data, "val", args.block_size,
                                     args.batch_size, device, args.eval_iters)
            print(f"step {it:5d} | val loss {val_loss:.4f} | lr {lr:.2e}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(),
                            "config": config.__dict__,
                            "iter": it,
                            "val_loss": val_loss},
                           os.path.join(out_dir, "ckpt.pt"))

        # gradient accumulation
        for micro in range(args.gradient_accumulation_steps):
            X, Y = get_batch(split_data, "train", args.block_size,
                             args.batch_size, device)
            _, loss = model(X, Y)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {it:5d} | loss {loss.item() * args.gradient_accumulation_steps:.4f} "
                  f"| lr {lr:.2e} | {dt * 1000:.0f}ms")

    print(f"done. best val loss {best_val:.4f}; checkpoint at "
          f"{os.path.join(out_dir, 'ckpt.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
