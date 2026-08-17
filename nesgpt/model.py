"""A minimal GPT language model in the style of nanoGPT.

A decoder-only transformer with a causal self-attention core, trained to
predict the next TX1 event given the previous ones.  Vocabulary is fixed at
631 tokens (``<S>`` + the 630 TX1 symbols); see ``nesgpt.vocab``.

Only depends on PyTorch (>= 2.0 recommended; 2.6+ required on Python 3.13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 512        # context length
    vocab_size: int = 631        # fixed TX1 vocabulary
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True             # True: bias in Linears and LayerNorms


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias (PyTorch's built-in has none)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask, registered so it moves with the module's device/dtype.
        mask = torch.tril(torch.ones(config.block_size, config.block_size)) \
                    .view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Prefer the fused kernel; fall back to explicit attention.
        if hasattr(F, "scaled_dot_product_attention") and x.device.type != "cpu":
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying between input embeddings and output head.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 recipe).
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"cannot forward sequence of length {T}, block size is {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # last position only
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 stop_token=0):
        """Autoregressive generation; stops early if ``stop_token`` is hit."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if stop_token is not None and int(next_id.item()) == stop_token:
                break
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas,
                             device_type):
        """AdamW with decay applied only to 2D+ weight matrices."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                 fused=fused)

    @staticmethod
    def from_pretrained_ckpt(ckpt_path: str, device="cpu") -> "GPT":
        ckpt = torch.load(ckpt_path, map_location=device)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config)
        model.load_state_dict(ckpt["model"])
        return model
