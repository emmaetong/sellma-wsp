"""Weight-space edit directions over the transformer block parameters."""

from __future__ import annotations

import torch

from .model import State


def edit_keys(state: State) -> list[str]:
    return [k for k in state if k.startswith("blocks.") and torch.is_floating_point(state[k])]


def flatten_edit(state_a: State, state_b: State, keys: list[str]) -> torch.Tensor:
    return torch.cat([(state_a[k] - state_b[k]).reshape(-1) for k in keys]).float()


def apply_edit(state: State, direction: torch.Tensor, alpha: float, keys: list[str]) -> State:
    out = {k: v.clone() for k, v in state.items()}
    start = 0
    for k in keys:
        n = out[k].numel()
        out[k] += alpha * direction[start : start + n].reshape_as(out[k])
        start += n
    return out
