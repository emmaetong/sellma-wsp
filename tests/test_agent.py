import json
from pathlib import Path

from sellma_wsp.agent.llm import LLMConfig, ScriptedBackend
from sellma_wsp.agent.session import run_session
from sellma_wsp.agent.workspace import Workspace
from sellma_wsp.config import quick_config
from sellma_wsp.experiment import Experiment

BIAS = "blocks.layers.0.linear2.bias"
W1 = "blocks.layers.0.linear1.weight"


def _ws(budget=2, runs=6):
    exp = Experiment(quick_config())
    target = exp.train(exp.base, exp.data.disrupted_data["train"], seed=1, steps=10)
    return exp, target, Workspace(exp, target, forward_budget=budget, python_runs=runs)


def _session(ws, replies, tmp_path):
    try:
        return run_session(ws, ScriptedBackend(LLMConfig(backend="scripted"), replies), tmp_path)
    finally:
        ws.close()


def test_session_runs_until_budget_used_and_validates_edits(tmp_path):
    exp, target, ws = _ws(budget=2)
    replies = [
        "Persistent kernel.\n```python\nx = len(NAMES)\nopen('/work/notes.md', 'w').write('n')\n```",
        "Reuse variable.\n```python\nx\n```",
        "No block yet: must not end the session while passes remain.",
        f"Edit.\n```python\nprint(save_edit('{BIAS}', load('{BIAS}') + 0.5))\n"
        f"np.save('/work/edited/token_emb.weight.npy', load_original('token_emb.weight'))\n"
        f"np.save('/work/edited/{W1}.npy', 3 * load_original('{W1}'))\n```",
        'Measure.\n```forward\n{"type": "score", "model": "both", "split": "dev"}\n```',
        'Internals.\n```forward\n{"type": "teacher_forced", "texts": ["en bil är", "ett hus nära en bil är"], '
        '"return": {"attention": true, "hidden_states": true}}\n```',
        'Done.\n```json\n{"edited_tensors": ["%s"], "confidence": 0.2}\n```' % BIAS,
    ]
    res = _session(ws, replies, tmp_path)
    assert res.ended_by == "final_answer"
    assert (res.python_used, res.forward_used) == (3, 2)
    assert res.final_json["edited_tensors"] == [BIAS]
    lines = [json.loads(line) for line in open(tmp_path / "transcript.jsonl")]
    outputs = [x["content"] for x in lines if x.get("tool") == "python"]
    assert outputs[1].startswith("[python result]\n") and outputs[1].split("\n")[1].isdigit()  # variable survived
    assert any(x.get("tool") == "continue" for x in lines)
    assert (ws.work / "notes.md").exists()
    state, applied, rejected = ws.edited_state()
    assert applied == [BIAS]
    assert any("token_emb" in r for r in rejected) and any("scaling" in r for r in rejected)
    assert (ws.work / "fwd_2" / "attention.npy").exists()


def test_forward_callable_from_python_counts_budget_and_hints_on_ellipsis(tmp_path):
    _, _, ws = _ws(budget=2)
    try:
        out = ws.run_python('r = forward({"type": "score", "model": "original", "split": "dev"})\nr')
        assert "forward pass 1 (score)" in out and out.count("forward pass 1 (score)") == 1
        assert ws.forward_used == 1 and "forward passes used: 1 of 2" in out
        out = ws.run_python('import pandas as pd; len(pd.read_json("/work/fwd_1.jsonl", lines=True))')
        assert "\n" in out and out.split("\n")[0].isdigit()
        out = ws.run_python("forward('{\"type\": \"score\", \"items\": [\"train-1\", ...]}')")
        assert "ellipsis" in out and ws.forward_used == 1
        out = ws.run_python("forward(type='score', split='train')")
        assert ws.forward_used == 2 and "REFUSED" in ws.run_python("forward(type='score', split='dev')")
    finally:
        ws.close()


def test_final_answer_is_grounded_and_claims_are_recorded(tmp_path):
    _, _, ws = _ws(budget=1, runs=2)
    replies = [
        "Look.\n```python\nprint(1)\n```",
        "Score.\n```python\nforward({'type': 'score', 'split': 'dev'})\n```",
        "Claim.\n```json\n{\"edited_tensors\": [\"%s\"]}\n```" % BIAS,
    ]
    res = _session(ws, replies, tmp_path)
    assert res.claimed_edits == [BIAS] and ws.edited_state()[1] == []
    lines = [json.loads(line) for line in open(tmp_path / "transcript.jsonl")]
    assert any("WILL be submitted: NONE" in x["content"] for x in lines if x["role"] == "user")


def test_zero_budget_refuses_forward_without_counting():
    _, _, ws = _ws(budget=0)
    out = ws.forward('{"type": "score", "split": "dev"}')
    ws.close()
    assert out.startswith("REFUSED") and ws.forward_used == 0


def test_kernel_blocks_torch_and_hides_real_paths():
    _, _, ws = _ws()
    try:
        assert "not available" in ws.run_python("import torch")
        out = ws.run_python("import os; os.getcwd()")
        assert "'/work'" in out and "sellma-agent" not in out
        out = ws.run_python("import json; json.loads('{')")  # traceback through the stdlib
        assert str(Path.home()) not in out
    finally:
        ws.close()
