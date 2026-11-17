import torch

from sellma_wsp.config import ModelConfig
from sellma_wsp.data import Dataset, V, build_counterfactual_items
from sellma_wsp.editing import apply_edit, edit_keys, flatten_edit
from sellma_wsp.model import initial_state


def test_split_sizes():
    ds = Dataset()
    sizes = {k: len(v) for k, v in ds.correct_items.items()}
    assert V == 62
    assert sizes == {"train": 648, "dev": 102, "test_id": 114, "test_relative": 38, "test_distractor": 38}


def test_counterfactual_grid_is_balanced():
    cf = build_counterfactual_items(Dataset().correct_items["test_id"])
    assert len(cf) == 228
    assert all(it["prefix"][4] != it["noun"] for it in cf)


def test_edit_roundtrip():
    base = initial_state(ModelConfig(), 314159)
    keys = edit_keys(base)
    assert sum(base[k].numel() for k in keys) == 12704
    other = {k: v + 1.0 for k, v in base.items()}
    d = flatten_edit(other, base, keys)
    edited = apply_edit(base, d, 1.0, keys)
    for k in keys:
        assert torch.allclose(edited[k], other[k])
