"""`uv run sellma-agent`: budgeted weight-editing agent vs. unchanged / task vector / random / fine-tuning.

Every method is applied to the same genuinely disrupted target runs and evaluated on the same
hidden splits. The key reported axis is how much feedback each method used: forward passes
(scoring calls on practice data) and gradient steps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config, setup_torch
from ..data import TEST_SPLITS, build_counterfactual_items
from ..editing import flatten_edit
from ..experiment import Experiment
from ..model import State, item_margins
from .llm import LLMConfig, make_backend
from .session import run_session
from .workspace import Workspace, sandbox_available

FT_CHECKPOINTS = (100, 200, 400)


# targets and task vector
def make_targets(exp: Experiment, n: int, steps: int) -> list[State]:
    """Broken targets: trained on disrupted data from the shared init for `steps` steps (as the sources are)."""
    return [exp.train(exp.base, exp.data.disrupted_data["train"], seed=900000 + j * 131, steps=steps)
            for j in range(n)]


def build_source_direction(exp: Experiment) -> torch.Tensor:
    """Task vector: mean (correct − disrupted) weight difference over matched source pairs."""
    tc = exp.cfg.transfer
    directions = []
    for i in range(tc.source_pairs):
        correct = exp.train(exp.base, exp.data.correct_data["train"], seed=100000 + i * 101, steps=tc.source_steps)
        disrupted = exp.train(exp.base, exp.data.disrupted_data["train"], seed=100000 + i * 101,
                              steps=tc.source_steps)
        directions.append(flatten_edit(correct, disrupted, exp.keys))
    return torch.stack(directions).mean(dim=0)


#  evaluation
def hidden_examples(exp: Experiment, state: State, cf_items: list[dict], **tags) -> list[dict]:
    """One row per hidden evaluation sentence: the sentence, both forms and the model's margin."""
    model = exp.load(state)
    rows = []
    sets = [(split, exp.data.correct_items[split]) for split in TEST_SPLITS] + [("counterfactual", cf_items)]
    for split, items in sets:
        for it, margin in zip(items, item_margins(model, items)):
            rows.append(dict(**tags, split=split, prefix=" ".join(it["prefix"]), correct=it["correct"],
                             incorrect=it["incorrect"], subject_gender=it["gender"],
                             distractor_gender=it.get("distractor_gender", ""), margin=round(margin, 4),
                             is_correct=margin > 0))
    return rows


def evaluate(exp: Experiment, state: State, original: State, cf_items: list[dict]) -> dict:
    test = exp.data.correct_items["test_id"]
    row = {}
    for split in TEST_SPLITS:
        m = exp.eval_split(state, split)
        row[f"{split}_acc"], row[f"{split}_margin"] = m["accuracy"], m["mean_margin"]
    for conflict in (False, True):
        m = exp.metrics(state, [it for it in cf_items if it["conflict"] == conflict])
        row[f"cf_{'conflict' if conflict else 'same'}_acc"] = m["accuracy"]
    row["prefix_nll_delta"] = exp.prefix_nll(state, test) - exp.prefix_nll(original, test)
    return row


# edit-strength sweep
SWEEP_COLS = ["dev_acc", "dev_prefix_nll_delta", "test_id_acc", "test_relative_acc", "test_distractor_acc",
              "cf_same_acc", "cf_conflict_acc", "test_id_margin", "prefix_nll_delta"]


def alpha_sweep(exp: Experiment, targets: list[State], D: torch.Tensor, cf_items, rng_seed: int) -> pd.DataFrame:
    """Accuracy vs regression at every edit strength, for the task vector and equal-norm random edits.

    The main comparison selects one alpha on dev accuracy alone; this reports the whole trade-off,
    including the dev-set regression that a regression-constrained selection would use. Random
    directions are drawn from their own generator, so the main comparison's draws are unchanged.
    """
    dev = exp.data.correct_items["dev"]
    rng = np.random.default_rng(rng_seed + 1)
    randoms = []
    for _ in range(exp.cfg.transfer.random_repeats):
        g = torch.from_numpy(rng.standard_normal(D.numel()).astype("float32"))
        randoms.append(g / g.norm() * D.norm())
    rows = []
    for j, s in enumerate(targets):
        dev_nll = exp.prefix_nll(s, dev)
        for alpha in exp.cfg.extension.alpha_grid:
            directions = [("task_vector", 0, D)]
            if alpha > 0:
                directions += [("random_equal_norm", r, g) for r, g in enumerate(randoms)]
            for method, draw, direction in directions:
                state = exp.edit(s, direction, alpha)
                rows.append(dict(target=j, method=method, alpha=alpha, draw=draw,
                                 dev_acc=exp.eval_split(state, "dev")["accuracy"],
                                 dev_prefix_nll_delta=exp.prefix_nll(state, dev) - dev_nll,
                                 **evaluate(exp, state, s, cf_items)))
    return pd.DataFrame(rows)


def write_sweep(sweep: pd.DataFrame, out: Path) -> pd.DataFrame:
    sweep.to_csv(out / "alpha_sweep.csv", index=False)
    # Average random draws within a target first, so sd is across targets for both methods.
    per_target = sweep.groupby(["method", "alpha", "target"], sort=False)[SWEEP_COLS].mean()
    summary = per_target.groupby(["method", "alpha"], sort=False).agg(["mean", "std"]).round(3)
    summary.to_csv(out / "alpha_sweep_summary.csv")
    return per_target.groupby(["method", "alpha"], sort=False).mean().round(3)


# comparators
def comparators(exp: Experiment, targets: list[State], cf_items, rng_seed: int
                ) -> tuple[list[dict], list[dict], pd.DataFrame]:
    tc = exp.cfg.transfer
    rows, examples = [], []
    print("Building source direction from", tc.source_pairs, "matched pairs ...")
    D = build_source_direction(exp)
    source_grad_steps = tc.source_pairs * 2 * tc.source_steps

    # Task vector: one global alpha chosen on dev (one scoring pass per alpha per target).
    dev = np.array([[exp.eval_split(exp.edit(s, D, a), "dev")["accuracy"] for a in tc.edit_alphas] for s in targets])
    alpha = float(tc.edit_alphas[int(np.argmax(dev.mean(0)))])
    print("Task-vector alpha (dev-selected):", alpha)

    # Fine-tuning: one global step count chosen on dev.
    ft = [exp.train_checkpoints(s, exp.data.correct_data["train"], seed=810000 + j * 131,
                                checkpoints=FT_CHECKPOINTS, lr=exp.cfg.model.learning_rate)
          for j, s in enumerate(targets)]
    ft_dev = {c: np.mean([exp.eval_split(f[c], "dev")["accuracy"] for f in ft]) for c in FT_CHECKPOINTS}
    ft_steps = max(ft_dev, key=lambda c: (ft_dev[c], -c))
    print("Fine-tuning steps (dev-selected):", ft_steps)

    rng = np.random.default_rng(rng_seed)
    for j, s in enumerate(targets):
        g = torch.from_numpy(rng.standard_normal(D.numel()).astype("float32"))
        g = g / g.norm() * D.norm()
        cands = [
            ("unchanged", None, s, 0, 0, 0),
            ("task_vector", alpha, exp.edit(s, D, alpha), len(tc.edit_alphas), 0, source_grad_steps),
            ("random_equal_norm", alpha, exp.edit(s, g, alpha), 0, 0, 0),
            ("fine_tune", ft_steps, ft[j][ft_steps], len(FT_CHECKPOINTS), ft_steps, 0),
        ]
        for method, setting, state, fwd, grad, src in cands:
            rows.append(dict(target=j, method=method, budget=None, setting=setting, forward_passes=fwd,
                             grad_steps=grad, source_grad_steps=src, **evaluate(exp, state, s, cf_items)))
            examples.extend(hidden_examples(exp, state, cf_items, target=j, method=method))
    return rows, examples, alpha_sweep(exp, targets, D, cf_items, rng_seed)


# ---------------------------------------------------------------------------- agent sessions
SUMMARY_COLS = ["test_id_acc", "test_relative_acc", "test_distractor_acc", "cf_same_acc", "cf_conflict_acc",
                "test_id_margin", "prefix_nll_delta", "forward_passes", "grad_steps", "source_grad_steps"]


def write_results(rows: list[dict], examples: list[dict], out: Path) -> pd.DataFrame:
    pd.DataFrame(examples).to_csv(out / "hidden_examples.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    df.groupby("method", sort=False)[SUMMARY_COLS].agg(["mean", "std", "count"]).round(3).to_csv(out / "summary.csv")
    return df


def agent_session(exp: Experiment, s: State, j: int, budget: int, cf_items, args, llm, llm_cfg: LLMConfig,
                  out: Path) -> dict:
    sdir = out / "sessions" / f"target{j}_budget{budget}"
    done = sdir / "session.json"
    if done.exists():
        print(f"[skip] target {j} budget {budget}: already done", flush=True)
        info = json.loads(done.read_text())
        state = {k: v.clone() for k, v in s.items()}
        for name in info["applied_edits"]:
            state[name] = torch.from_numpy(np.load(sdir / "work" / "edited" / f"{name}.npy").astype(np.float32))
        return info, state
    if sdir.exists():
        shutil.rmtree(sdir)
    print(f"[agent] target {j} budget {budget} ({llm_cfg.model}) started {time.strftime('%H:%M')}", flush=True)
    ws = Workspace(exp, s, forward_budget=budget, python_runs=args.python_runs)
    try:
        res = run_session(ws, llm, sdir, max_turns=args.max_turns, num_ctx=llm_cfg.num_ctx)
    finally:
        ws.close()
    state, applied, rejected = ws.edited_state()
    shutil.copytree(ws.work, sdir / "work")
    shutil.rmtree(ws.runtime, ignore_errors=True)
    info = dict(target=j, budget=budget, turns=res.turns, python_used=res.python_used,
                forward_used=res.forward_used, kernel_restarts=res.kernel_restarts,
                window_trims=res.window_trims, ended_by=res.ended_by, seconds=res.seconds,
                applied_edits=applied, rejected_edits=rejected, claimed_edits=res.claimed_edits,
                claims_match_applied=(None if res.claimed_edits is None
                                      else sorted(map(str, res.claimed_edits)) == sorted(applied)),
                final_json=res.final_json,
                simulation_flagged_runs=sum(1 for a in ws.audit if a.get("simulation_flags")),
                metrics=evaluate(exp, state, s, cf_items))
    done.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    (sdir / "final_answer.md").write_text(res.final_reply)
    print(f"  ended by {res.ended_by} after {res.turns} turns ({res.seconds / 60:.0f} min): "
          f"{res.python_used} python, {res.forward_used} forward; edits {applied or 'none'}; "
          f"test_id acc {info['metrics']['test_id_acc']:.3f}, "
          f"prefix NLL delta {info['metrics']['prefix_nll_delta']:+.3f}", flush=True)
    return info, state


def agent_row(info: dict) -> dict:
    return dict(target=info["target"], method=f"agent_b{info['budget']}", budget=info["budget"], setting=None,
                forward_passes=info["forward_used"], grad_steps=0, source_grad_steps=0,
                python_runs=info["python_used"], n_edits=len(info["applied_edits"]), ended_by=info["ended_by"],
                claims_match_applied=info.get("claims_match_applied"),
                simulation_flagged_runs=info["simulation_flagged_runs"], **info["metrics"])


# main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("results/agent"))
    p.add_argument("--targets", type=int, default=3, help="number of target runs (default 3)")
    p.add_argument("--target-steps", type=int, default=400)
    p.add_argument("--budgets", default="10,25,50",
                   help="comma-separated forward-pass budgets (default 10,25,50 as in the article; 0 = weights only)")
    p.add_argument("--python-runs", type=int, default=40, help="Python runs per session (default 40)")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--no-agent", action="store_true", help="only compute the comparison methods")
    g = p.add_argument_group("LLM (free: local Ollama by default, or any OpenAI-compatible endpoint)")
    g.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    g.add_argument("--model", default="qwen3:4b")
    g.add_argument("--base-url", default="")
    g.add_argument("--api-key-env", default="OPENAI_API_KEY")
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--llm-seed", type=int, default=0)
    g.add_argument("--num-ctx", type=int, default=12288, help="Ollama context window")
    g.add_argument("--max-tokens", type=int, default=3072, help="max tokens per reply")
    g.add_argument("--think", action="store_true", help="Ollama: enable the model's thinking mode")
    args = p.parse_args(argv)
    args.budgets = [int(b) for b in args.budgets.split(",")]

    setup_torch()
    cfg = Config()
    llm_cfg = LLMConfig(backend=args.backend, model=args.model, base_url=args.base_url,
                        api_key_env=args.api_key_env, temperature=args.temperature, seed=args.llm_seed,
                        num_ctx=args.num_ctx, think=args.think, max_tokens=args.max_tokens)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    exp = Experiment(cfg)
    targets = make_targets(exp, args.targets, args.target_steps)
    cf_items = build_counterfactual_items(exp.data.correct_items["test_id"])
    for j, s in enumerate(targets):
        print(f"target {j}: dev accuracy {exp.eval_split(s, 'dev')['accuracy']:.3f}")

    run_config = {"experiment": dataclasses.asdict(cfg), "args": {k: (str(v) if isinstance(v, Path) else v)
                                                                  for k, v in vars(args).items()},
                  "source_seed": "100000 + 101*i", "target_seed": "900000 + 131*j",
                  "finetune_seed": "810000 + 131*j", "sandbox": sandbox_available(),
                  "python": sys.version.split()[0], "torch": torch.__version__}
    if not args.no_agent:
        run_config["llm"] = make_backend(llm_cfg).describe()
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))

    rows, examples, sweep = comparators(exp, targets, cf_items, rng_seed=cfg.transfer.seed)
    df = write_results(rows, examples, out)
    sweep_means = write_sweep(sweep, out)
    if not args.no_agent:
        llm = make_backend(llm_cfg)
        for j, s in enumerate(targets):
            for budget in args.budgets:
                info, state = agent_session(exp, s, j, budget, cf_items, args, llm, llm_cfg, out)
                rows.append(agent_row(info))
                examples.extend(hidden_examples(exp, state, cf_items, target=j, method=f"agent_b{budget}"))
                df = write_results(rows, examples, out)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\nHIDDEN EVALUATION (mean across targets; sd and n in summary.csv)")
        print(df.groupby("method", sort=False)[SUMMARY_COLS].mean().round(3).to_string())
        print("\nEDIT-STRENGTH SWEEP (mean across targets; per-target rows in alpha_sweep.csv)")
        print(sweep_means.to_string())
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {out}/")


if __name__ == "__main__":
    main()
