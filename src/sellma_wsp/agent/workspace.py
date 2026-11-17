"""The agent's workspace

Runtime layout of one session (a temporary folder outside the project, so the sandbox can hide
the project source and results from the agent). The agent sees these as /model, /practice, /work:

    model/      read-only: weights.npz, config.json, vocab.json
    practice/   read-only: practice.jsonl (train and dev items with answers)
    work/       read-write: edited/<tensor>.npy, notes.md, fwd_<k>.jsonl, fwd_<k>/*.npy, ...

Code runs in a persistent, sandboxed Python kernel (numpy, scipy, scikit-learn, pandas, stdlib;
no torch, so no gradients). The forward-pass tool is the only way to run the model.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

from ..data import STOI, V
from ..experiment import Experiment
from ..model import State, TinyDecoderLM

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PREFIXES = ("token_emb.", "pos_emb.", "lm_head.", "norm.")  # frozen in training; out of scope
MAX_SCORE_ITEMS = 1000
MAX_TF_TEXTS = 1024
OUTPUT_LIMIT = 10000
CELL_TIMEOUT = 300
VIRTUAL_DIRS = ("model", "practice", "work")
_VIRTUAL = re.compile(r"(?<![\w./-])/(model|practice|work)(?=/|\b)")

# Patterns that suggest the agent is re-implementing the network's computation; flagged for review.
SIMULATION_PATTERNS = {
    "softmax": r"softmax",
    "gelu/erf/tanh": r"\bgelu\b|\berf\b|np\.tanh",
    "exp-normalise": r"np\.exp\([^)]*\)\s*/",
    "layernorm-of-activations": r"layer_?norm\(|def\s+\w*norm\w*\(",
}

PRELUDE = r'''
# Prelude: runs once when the kernel starts.
import json, os, sys, math
from pathlib import Path
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=150)

class _Blocker:
    BLOCKED = {"torch", "jax", "tensorflow", "sellma_wsp"}
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(f"import of {name!r} is not available in this kernel")
        return None
sys.meta_path.insert(0, _Blocker())

MODEL_DIR = Path(__MODEL__); PRACTICE_DIR = Path(__PRACTICE__); WORK = Path(__WORK__)
os.chdir(WORK)
CONFIG = json.load(open(MODEL_DIR / "config.json"))
_W = np.load(MODEL_DIR / "weights.npz")
NAMES = sorted(_W.files)
EDITABLE = CONFIG["editable_tensors"]
VOCAB = json.load(open(MODEL_DIR / "vocab.json"))           # token id -> word
STOI = {w: i for i, w in enumerate(VOCAB)}
_EDIT_DIR = WORK / "edited"

def load_original(name):
    """Original (unedited) tensor as a float32 numpy array."""
    return _W[name].astype(np.float32).copy()

def load(name):
    """Your current edit of a tensor if there is one, else the original."""
    p = _EDIT_DIR / f"{name}.npy"
    return np.load(p).astype(np.float32) if p.exists() else load_original(name)

def save_edit(name, tensor):
    """Save a full replacement for one editable tensor (same shape)."""
    t = np.asarray(tensor, dtype=np.float32)
    if name not in EDITABLE:
        raise ValueError(f"refused: {name} is out of scope; editable tensors: {EDITABLE}")
    o = load_original(name)
    if t.shape != o.shape:
        raise ValueError(f"refused: shape {t.shape} != {o.shape}")
    if not np.isfinite(t).all():
        raise ValueError("refused: tensor has non-finite values")
    nz = np.abs(o) > 1e-8
    if nz.any():
        ratio = t[nz] / o[nz]
        if np.allclose(ratio, ratio.mean(), rtol=1e-4, atol=1e-6) and abs(ratio.mean() - 1) > 1e-6:
            raise ValueError("refused: multiplying a whole tensor by a constant is out of scope")
    _EDIT_DIR.mkdir(exist_ok=True)
    np.save(_EDIT_DIR / f"{name}.npy", t)
    return f"saved edit of {name} (relative change {np.linalg.norm(t - o) / (np.linalg.norm(o) + 1e-12):.4f})"

def drop_edit(name):
    p = _EDIT_DIR / f"{name}.npy"
    if p.exists(): p.unlink()
    return f"dropped {name}"

def edited():
    return sorted(p.stem for p in _EDIT_DIR.glob("*.npy")) if _EDIT_DIR.exists() else []
'''


def _sandbox_profile(runtime: str) -> str:
    proj = os.path.realpath(PROJECT_ROOT)
    venv = os.path.realpath(Path(sys.prefix))
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{runtime}/work") (literal "/dev/null") (regex #"^/dev/tty"))
(deny file-read* (subpath "{proj}"))
(allow file-read* (subpath "{venv}") (subpath "{runtime}"))
"""


def sandbox_available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


class Kernel:
    """A persistent Python process; restarted (losing variables) after a timeout or crash."""

    def __init__(self, runtime: Path, rpc_handler=None):
        self.runtime = runtime
        self.rpc_handler = rpc_handler or (lambda request: "forward() is not available")
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()

    def start(self) -> None:
        cmd = [sys.executable, "-I", "-B", str(self.runtime / "kernel.py")]
        if sandbox_available():
            cmd = ["sandbox-exec", "-f", str(self.runtime / "sandbox.sb")] + cmd
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.runtime / "work"), "OMP_NUM_THREADS": "2",
               "KERNEL_PRELUDE": str(self.runtime / "prelude.py"), "MPLCONFIGDIR": str(self.runtime / "work")}
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     cwd=self.runtime / "work", env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self.q), daemon=True).start()

    @staticmethod
    def _reader(proc, q):
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def run(self, code: str, timeout: int = CELL_TIMEOUT) -> tuple[str, bool]:
        """Returns (output, restarted)."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        data = code.encode("utf-8")
        try:
            self.proc.stdin.write(str(len(data)).encode() + b"\n" + data)
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.stop()
            return "[kernel crashed and was restarted: variables are lost]", True
        stray = []
        while True:
            try:
                line = self.q.get(timeout=timeout)
            except queue.Empty:
                self.stop()
                return f"[timed out after {timeout}s: the kernel was restarted and variables are lost]", True
            if line is None:
                out = b"".join(stray).decode("utf-8", "replace")
                self.stop()
                return out + "\n[kernel crashed and was restarted: variables are lost]", True
            if line.startswith(b"\x00RPC "):
                reply = self.rpc_handler(json.loads(line[len(b"\x00RPC "):])["request"]).encode("utf-8")
                try:
                    self.proc.stdin.write(str(len(reply)).encode() + b"\n" + reply)
                    self.proc.stdin.flush()
                except BrokenPipeError:
                    pass
                continue
            if line.startswith(b"\x00RESULT "):
                out = json.loads(line[len(b"\x00RESULT "):])["out"]
                return b"".join(stray).decode("utf-8", "replace") + out, False
            stray.append(line)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None


class Workspace:
    def __init__(self, exp: Experiment, target: State, forward_budget: int, python_runs: int,
                 runtime_dir: Path | None = None):
        self.exp = exp
        self.target = target
        self.forward_budget = forward_budget
        self.forward_used = 0
        self.python_budget = python_runs
        self.python_used = 0
        self.kernel_restarts = 0
        self.audit: list[dict] = []
        self.runtime = Path(os.path.realpath(runtime_dir or tempfile.mkdtemp(prefix="sellma-agent-")))
        self.dirs = {d: self.runtime / d for d in VIRTUAL_DIRS}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        (self.dirs["work"] / "edited").mkdir(exist_ok=True)
        self.work = self.dirs["work"]
        self.editable = [k for k in exp.keys if not k.startswith(FROZEN_PREFIXES)]
        self._practice = self._practice_items()
        self._write_files()
        self.kernel = Kernel(self.runtime, rpc_handler=lambda request: self.to_virtual(self.forward(request)))

    # ------------------------------------------------------------------ files
    def _practice_items(self) -> dict[str, dict]:
        items = {}
        for split in ("train", "dev"):
            for i, it in enumerate(self.exp.data.correct_items[split]):
                items[f"{split}-{i}"] = it
        return items

    def architecture(self) -> dict:
        mc = self.exp.cfg.model
        return {
            "type": "decoder-only transformer (torch.nn.TransformerEncoderLayer with a causal mask)",
            "n_layers": mc.n_layers, "d_model": mc.embedding_dim, "n_heads": mc.n_heads,
            "d_head": mc.embedding_dim // mc.n_heads, "d_ff": 4 * mc.embedding_dim, "activation": "gelu (exact, erf)",
            "norm": "pre-LayerNorm (norm_first=True), eps=1e-5, plus a final LayerNorm 'norm' before the output layer",
            "dropout": 0.0, "positions": f"learned absolute embeddings pos_emb (max {mc.max_seq_len} tokens)",
            "tied_output": "lm_head.weight is the same tensor as token_emb.weight",
            "attention_layout": "self_attn.in_proj_weight stacks [W_q; W_k; W_v] row-wise, each [d_model, d_model]; "
                                "heads split d_model into n_heads contiguous slices; scores scaled by 1/sqrt(d_head)",
            "vocab_size": V, "no_bos_token": True,
        }

    def _write_files(self) -> None:
        m, p = self.dirs["model"], self.dirs["practice"]
        np.savez(m / "weights.npz", **{k: v.float().numpy() for k, v in self.target.items()})
        (m / "config.json").write_text(json.dumps({
            "architecture": self.architecture(), "tensor_shapes": {k: list(v.shape) for k, v in self.target.items()},
            "editable_tensors": self.editable}, indent=2, ensure_ascii=False))
        (m / "vocab.json").write_text(json.dumps(list(STOI), ensure_ascii=False))
        with open(p / "practice.jsonl", "w") as f:
            for pid, it in self._practice.items():
                f.write(json.dumps({"id": pid, "prefix": " ".join(it["prefix"]), "correct": it["correct"],
                                    "incorrect": it["incorrect"], "structure": it["structure"],
                                    "noun": it["noun"], "gender": it["gender"]}, ensure_ascii=False) + "\n")
        prelude = (PRELUDE.replace("__MODEL__", repr(str(m))).replace("__PRACTICE__", repr(str(p)))
                   .replace("__WORK__", repr(str(self.work))))
        (self.runtime / "prelude.py").write_text(prelude)
        shutil.copy(Path(__file__).with_name("kernel.py"), self.runtime / "kernel.py")
        (self.runtime / "sandbox.sb").write_text(_sandbox_profile(str(self.runtime)))

    def to_real(self, code: str) -> str:
        """Map the virtual /model, /practice and /work paths the agent uses to the real folders."""
        return _VIRTUAL.sub(lambda mt: str(self.dirs[mt.group(1)]), code)

    def to_virtual(self, text: str) -> str:
        """Hide real paths: session folders become /model etc., and the home folder becomes ~."""
        return text.replace(str(self.runtime), "").replace(str(Path.home()), "~")

    def saved_edits(self) -> list[str]:
        return sorted(p.stem for p in (self.work / "edited").glob("*.npy"))

    def status(self) -> str:
        return (f"[python runs used: {self.python_used} of {self.python_budget} | forward passes used: "
                f"{self.forward_used} of {self.forward_budget} | saved edits: {self.saved_edits() or 'none'}]")

    def notes(self) -> str:
        p = self.work / "notes.md"
        return p.read_text()[-4000:] if p.exists() else "(no /work/notes.md yet)"

    def close(self) -> None:
        self.kernel.stop()

    # ------------------------------------------------------------------ python
    def run_python(self, code: str) -> str:
        if self.python_used >= self.python_budget:
            return "REFUSED: no Python runs left."
        self.python_used += 1
        flags = [k for k, pat in SIMULATION_PATTERNS.items() if re.search(pat, code, re.I)]
        self.audit.append({"kind": "python", "run": self.python_used, "simulation_flags": flags})
        with open(self.work / ".cells.py", "a") as f:
            f.write(f"\n# ===== cell {self.python_used} =====\n{code}\n")
        out, restarted = self.kernel.run(self.to_real(code))
        self.kernel_restarts += restarted
        out = self.to_virtual(out)
        if len(out) > OUTPUT_LIMIT:
            out = out[:OUTPUT_LIMIT] + f"\n[output cut at {OUTPUT_LIMIT:,} characters]"
        return (out.rstrip() or "[no output]") + "\n" + self.status()

    # ------------------------------------------------------------------ edits
    def edited_state(self) -> tuple[State, list[str], list[str]]:
        """Authoritative re-validation of the agent's saved edits (the kernel's checks can be bypassed)."""
        state = {k: v.clone() for k, v in self.target.items()}
        applied, rejected = [], []
        for p in sorted((self.work / "edited").glob("*.npy")):
            name = p.stem
            try:
                arr = np.load(p).astype(np.float32)
            except Exception as e:  # noqa: BLE001
                rejected.append(f"{name}: unreadable ({e})")
                continue
            if name not in self.editable:
                rejected.append(f"{name}: not editable")
            elif tuple(arr.shape) != tuple(state[name].shape):
                rejected.append(f"{name}: wrong shape {arr.shape}")
            elif not np.isfinite(arr).all():
                rejected.append(f"{name}: non-finite values")
            else:
                orig = state[name].numpy()
                nz = np.abs(orig) > 1e-8
                ratio = arr[nz] / orig[nz] if nz.any() else np.array([1.0])
                if np.allclose(ratio, ratio.mean(), rtol=1e-4, atol=1e-6) and abs(ratio.mean() - 1) > 1e-6:
                    rejected.append(f"{name}: whole-tensor scaling")
                else:
                    state[name] = torch.from_numpy(arr.copy())
                    applied.append(name)
        return state, applied, rejected

    # ------------------------------------------------------------------ forward tool
    def _tokens(self, text: str) -> list[int]:
        words = text.split()
        unknown = [w for w in words if w not in STOI]
        if unknown:
            raise ValueError(f"unknown words {unknown}; use only words from /model/vocab.json")
        if not 1 <= len(words) <= self.exp.cfg.model.max_seq_len:
            raise ValueError(f"texts must have 1..{self.exp.cfg.model.max_seq_len} words")
        return [STOI[w] for w in words]

    def _items(self, req: dict) -> list[dict]:
        items = []
        for pid in req.get("items", []) or []:
            if pid not in self._practice:
                raise ValueError(f"unknown practice id {pid!r} (ids look like 'train-0' or 'dev-12'; "
                                 "or use \"split\": \"dev\" / \"train\" / \"practice\")")
            it = self._practice[pid]
            items.append(dict(id=pid, prefix=it["prefix"], correct=it["correct"], incorrect=it["incorrect"],
                              structure=it["structure"]))
        split = req.get("split")
        if split:
            if split not in ("train", "dev", "practice"):
                raise ValueError("split must be 'train', 'dev' or 'practice'")
            for pid, it in self._practice.items():
                if split == "practice" or pid.startswith(split + "-"):
                    items.append(dict(id=pid, prefix=it["prefix"], correct=it["correct"],
                                      incorrect=it["incorrect"], structure=it["structure"]))
        for k, s in enumerate(req.get("sentences", []) or []):
            self._tokens(s["prefix"] + " " + s["correct"] + " " + s["incorrect"])
            items.append(dict(id=f"own-{k}", prefix=s["prefix"].split(), correct=s["correct"],
                              incorrect=s["incorrect"], structure="own"))
        if not items:
            raise ValueError("give 'items', 'split' or 'sentences'")
        if len(items) > MAX_SCORE_ITEMS:
            raise ValueError(f"at most {MAX_SCORE_ITEMS} items per pass")
        return items

    def _margins(self, state: State, items: list[dict]) -> np.ndarray:
        model = self.exp.load(state).eval()
        out = np.zeros(len(items))
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

    def forward(self, request: str) -> str:
        try:
            req = json.loads(request)
        except json.JSONDecodeError as e:
            hint = (' The request contains "..." (an ellipsis): list every id explicitly, or use '
                    '"split": "dev", "train" or "practice" instead of listing ids.') if "..." in request else ""
            return f"REFUSED (not counted): invalid JSON: {e}.{hint}"
        if not isinstance(req, dict):
            return "REFUSED (not counted): the request must be a JSON object"
        if self.forward_used >= self.forward_budget:
            return f"REFUSED (not counted): all {self.forward_budget} forward passes are used."
        try:
            kind = req.get("type", "score")
            if kind == "score":
                body = self._forward_score(req)
            elif kind == "teacher_forced":
                body = self._forward_teacher_forced(req)
            else:
                raise ValueError("type must be 'score' or 'teacher_forced'")
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            return f"REFUSED (not counted): {e}"
        self.forward_used += 1
        self.audit.append({"kind": "forward", "pass": self.forward_used, "request": req})
        return body + f"\n[forward passes left: {self.forward_budget - self.forward_used}]"

    def _forward_score(self, req: dict) -> str:
        items = self._items(req)
        which = req.get("model", "original")
        if which not in ("original", "edited", "both"):
            raise ValueError("model must be 'original', 'edited' or 'both'")
        k = self.forward_used + 1
        res = {}
        if which in ("original", "both"):
            res["original"] = self._margins(self.target, items)
        if which in ("edited", "both"):
            res["edited"] = self._margins(self.edited_state()[0], items)
        with open(self.work / f"fwd_{k}.jsonl", "w") as f:
            for i, it in enumerate(items):
                row = {"id": it["id"], "prefix": " ".join(it["prefix"]), "correct": it["correct"],
                       "incorrect": it["incorrect"], "structure": it["structure"]}
                for m, arr in res.items():
                    row[f"margin_{m}"] = round(float(arr[i]), 4)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        lines = [f"forward pass {k} (score): {len(items)} items; margin = log p(correct) - log p(incorrect); "
                 f"per-item margins in /work/fwd_{k}.jsonl"]
        structures = sorted({it["structure"] for it in items})
        for m, arr in res.items():
            parts = []
            for s in structures:
                mask = np.array([it["structure"] == s for it in items])
                parts.append(f"{s}: acc {np.mean(arr[mask] > 0):.3f} margin {arr[mask].mean():+.3f} (n={mask.sum()})")
            lines.append(f"  {m}: accuracy {np.mean(arr > 0):.3f}, mean margin {arr.mean():+.3f} | " + "; ".join(parts))
        if which == "both":
            d_acc = (res["edited"] > 0).astype(float) - (res["original"] > 0).astype(float)
            d_m = res["edited"] - res["original"]
            n = len(items)
            se = (lambda a: a.std(ddof=1) / math.sqrt(n)) if n > 1 else (lambda a: 0.0)
            lines.append(f"  paired edited - original: accuracy {d_acc.mean():+.3f} ± {se(d_acc):.3f} (SE), "
                         f"margin {d_m.mean():+.3f} ± {se(d_m):.3f}; wrong->right {int((d_acc > 0).sum())}, "
                         f"right->wrong {int((d_acc < 0).sum())}")
        return "\n".join(lines)

    def _forward_teacher_forced(self, req: dict) -> str:
        texts = req.get("texts") or []
        if not 1 <= len(texts) <= MAX_TF_TEXTS:
            raise ValueError(f"give 1..{MAX_TF_TEXTS} texts")
        toks = [self._tokens(t) for t in texts]
        which = req.get("model", "original")
        if which not in ("original", "edited"):
            raise ValueError("model must be 'original' or 'edited' for teacher_forced")
        ret = req.get("return", {"logprobs": True})
        want = set(ret) if isinstance(ret, list) else {k for k, v in ret.items() if v}
        unknown = want - {"logprobs", "hidden_states", "attention"}
        if unknown:
            raise ValueError(f"unknown return keys {sorted(unknown)}")
        state = self.target if which == "original" else self.edited_state()[0]
        model = self.exp.load(state).eval()
        mc = self.exp.cfg.model
        n, T, L = len(toks), max(map(len, toks)), mc.n_layers
        arrays = {"logprobs": np.zeros((n, T, V), np.float32),
                  "hidden_states": np.zeros((n, L + 1, T, mc.embedding_dim), np.float32),
                  "attention": np.zeros((n, L, mc.n_heads, T, T), np.float32)}
        for i, t in enumerate(toks):
            logits, hidden, attn = forward_internals(model, torch.tensor([t]))
            s = len(t)
            arrays["logprobs"][i, :s] = torch.log_softmax(logits, -1).numpy()
            arrays["hidden_states"][i, :, :s] = hidden.numpy()
            arrays["attention"][i, :, :, :s, :s] = attn.numpy()
        k = self.forward_used + 1
        d = self.work / f"fwd_{k}"
        d.mkdir(exist_ok=True)
        for key in want:
            np.save(d / f"{key}.npy", arrays[key])
        (d / "meta.json").write_text(json.dumps({"texts": texts, "token_ids": toks, "lengths": [len(t) for t in toks],
                                                 "model": which}, ensure_ascii=False))
        shapes = {"logprobs": "[texts, tokens, vocab] next-token log-probs",
                  "hidden_states": "[texts, layers+1, tokens, d_model] residual stream (0 = embeddings, l = after block l)",
                  "attention": "[texts, layers, heads, query, key]"}
        return (f"forward pass {k} (teacher_forced): read {n} texts with the {which} model; saved in /work/fwd_{k}/ "
                + "; ".join(f"{key}.npy {shapes[key]}" for key in sorted(want))
                + "; right-padded with zeros, meta.json has texts, token ids and lengths")


def forward_internals(m: TinyDecoderLM, x: torch.Tensor):
    """Logits, residual stream after each block, and per-head attention weights for one sequence."""
    with torch.no_grad():
        t = x.shape[1]
        h = m.token_emb(x) + m.pos_emb(torch.arange(t)[None, :])
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        hidden, attns = [h[0]], []
        for layer in m.blocks.layers:
            a_in = layer.norm1(h)
            a_out, w = layer.self_attn(a_in, a_in, a_in, attn_mask=causal, need_weights=True,
                                       average_attn_weights=False)
            h = h + a_out
            h = h + layer._ff_block(layer.norm2(h))
            hidden.append(h[0])
            attns.append(w[0])
        logits = m.lm_head(m.norm(h))[0]
    return logits, torch.stack(hidden), torch.stack(attns)
