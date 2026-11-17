"""Tiny decoder LM, training loops and agreement metrics."""

from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .config import ModelConfig, set_seed
from .data import PAD, STOI, Item, V

State = dict[str, torch.Tensor]


class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 2, n_layers: int = 1, max_len: int = 16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, batch_first=True, activation="gelu", norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)[None, :]
        h = self.token_emb(x) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        return self.lm_head(self.norm(self.blocks(h, mask=causal)))


def new_model(cfg: ModelConfig) -> TinyDecoderLM:
    return TinyDecoderLM(V, cfg.embedding_dim, cfg.n_heads, cfg.n_layers, cfg.max_seq_len)


def state_cpu(m: nn.Module) -> State:
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def initial_state(cfg: ModelConfig, seed: int) -> State:
    set_seed(seed)
    return state_cpu(new_model(cfg))


def load_model(cfg: ModelConfig, state: State) -> TinyDecoderLM:
    m = new_model(cfg)
    m.load_state_dict(state)
    return m


def train_checkpoints(cfg: ModelConfig, init: State, rows: torch.Tensor, seed: int,
                      checkpoints: Iterable[int], lr: float | None = None) -> dict[int, State]:
    """Train from `init` with frozen embeddings, returning states at the requested step counts."""
    checkpoints = set(checkpoints)
    set_seed(seed)
    m = new_model(cfg)  
    m.load_state_dict(copy.deepcopy(init))
    m.token_emb.weight.requires_grad_(False)
    m.pos_emb.weight.requires_grad_(False)
    m.train()
    x, y = rows[:, :-1], rows[:, 1:]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr or cfg.learning_rate)
    out = {}
    for step in range(1, max(checkpoints) + 1):
        idx = torch.randint(len(x), (cfg.batch_size,))
        logits = m(x[idx])
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), y[idx].reshape(-1), ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in checkpoints:
            out[step] = state_cpu(m)
    return out


def train_state(cfg: ModelConfig, init: State, rows: torch.Tensor, seed: int, steps: int) -> State:
    return train_checkpoints(cfg, init, rows, seed, [steps])[steps]


def agreement_metrics(model: nn.Module, items: list[Item]) -> dict:
    """Compare log-probs of both adjective alternatives after the identical prefix."""
    model.eval()
    margins, hits = [], []
    groups: dict[int, list[Item]] = {}
    for it in items:
        groups.setdefault(len(it["prefix"]), []).append(it)
    with torch.no_grad():
        for group in groups.values():
            x = torch.tensor([[STOI[w] for w in it["prefix"]] for it in group])
            lp = torch.log_softmax(model(x)[:, -1, :], dim=-1)
            for j, it in enumerate(group):
                margin = float(lp[j, STOI[it["correct"]]] - lp[j, STOI[it["incorrect"]]])
                margins.append(margin)
                hits.append(float(margin > 0))
    return {"accuracy": float(np.mean(hits)), "mean_margin": float(np.mean(margins)), "n": len(hits)}


def item_margins(model: nn.Module, items: list[Item]) -> list[float]:
    """Per-item log p(correct) − log p(incorrect), in the order of `items`."""
    model.eval()
    out = [0.0] * len(items)
    groups: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        groups.setdefault(len(it["prefix"]), []).append(i)
    with torch.no_grad():
        for idx in groups.values():
            x = torch.tensor([[STOI[w] for w in items[i]["prefix"]] for i in idx])
            lp = torch.log_softmax(model(x)[:, -1, :], dim=-1)
            for j, i in enumerate(idx):
                out[i] = float(lp[j, STOI[items[i]["correct"]]] - lp[j, STOI[items[i]["incorrect"]]])
    return out


def prefix_nll(model: nn.Module, items: list[Item]) -> float:
    """Mean NLL over all pre-adjective next-token predictions.

    Includes articles, nouns, attributive adjectives, verbs and adverbs; the final predicate
    adjective is excluded so target-task improvements cannot hide regressions.
    """
    model.eval()
    groups: dict[int, list[list[int]]] = {}
    for it in items:
        toks = [STOI[w] for w in it["prefix"]]
        if len(toks) >= 2:
            groups.setdefault(len(toks), []).append(toks)
    numerator, denominator = 0.0, 0
    with torch.no_grad():
        for seqs in groups.values():
            x = torch.tensor(seqs, dtype=torch.long)
            lp = torch.log_softmax(model(x[:, :-1]), dim=-1)
            nll = -lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
            numerator += float(nll.sum())
            denominator += int(nll.numel())
    return numerator / denominator
