"""Experiment configuration and global setup."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class ModelConfig:
    batch_size: int = 32
    learning_rate: float = 3e-3
    embedding_dim: int = 32
    n_heads: int = 2
    n_layers: int = 1
    max_seq_len: int = 16


@dataclass(frozen=True)
class TransferConfig:
    source_pairs: int = 10
    source_steps: int = 400
    edit_alphas: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    random_repeats: int = 12
    seed: int = 20261116
    base_init_seed: int = 314159


@dataclass(frozen=True)
class ExtensionConfig:
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    extension: ExtensionConfig = field(default_factory=ExtensionConfig)


def quick_config() -> Config:
    """A drastically reduced configuration for smoke-testing the pipeline."""
    return Config(
        transfer=TransferConfig(source_pairs=2, source_steps=40, random_repeats=2),
    )


def setup_torch() -> None:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
