"""Shared experiment context: config, data, base checkpoint and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import model as M
from .config import Config
from .data import Dataset, Item
from .editing import apply_edit, edit_keys
from .model import State


@dataclass
class Experiment:
    cfg: Config
    data: Dataset = field(init=False)
    base: State = field(init=False)
    keys: list[str] = field(init=False)

    def __post_init__(self):
        self.data = Dataset(self.cfg.model.max_seq_len)
        self.base = M.initial_state(self.cfg.model, self.cfg.transfer.base_init_seed)
        self.keys = edit_keys(self.base)

    # training
    def train(self, init: State, rows: torch.Tensor, seed: int, steps: int) -> State:
        return M.train_state(self.cfg.model, init, rows, seed, steps)

    def train_checkpoints(self, init: State, rows: torch.Tensor, seed: int, checkpoints, lr: float) -> dict[int, State]:
        return M.train_checkpoints(self.cfg.model, init, rows, seed, checkpoints, lr=lr)

    # editing
    def edit(self, state: State, direction: torch.Tensor, alpha: float) -> State:
        return apply_edit(state, direction, alpha, self.keys)

    # evaluation
    def load(self, state: State) -> M.TinyDecoderLM:
        return M.load_model(self.cfg.model, state)

    def metrics(self, state: State, items: list[Item]) -> dict:
        return M.agreement_metrics(self.load(state), items)

    def eval_split(self, state: State, split: str) -> dict:
        return self.metrics(state, self.data.correct_items[split])

    def prefix_nll(self, state: State, items: list[Item]) -> float:
        return M.prefix_nll(self.load(state), items)
