"""Figures for the research note, drawn from the main experiment's summary files.

    uv run --group plots python scripts/make_figures.py

Reads results/comparators/summary.csv and alpha_sweep_summary.csv; writes PNGs to
results/comparators/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/comparators")
OUT = RESULTS / "figures"

# Light-mode tokens. The three method hues are validated as a set (all pairs, CVD and normal vision).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHODS = {  # fixed colour per method, in every figure
    "unchanged": ("Unchanged", "#a3a29c"),
    "task_vector": ("Task vector", "#2a78d6"),
    "fine_tune": ("Fine-tuning", "#eb6834"),
    "random_equal_norm": ("Random edit", "#1baf7a"),
}
SPLITS = {
    "test_id_acc": "Familiar",
    "test_relative_acc": "Relative\nclause",
    "test_distractor_acc": "Intervening\nnoun",
    "cf_same_acc": "Same gender\n(counterfactual)",
    "cf_conflict_acc": "Opposite gender\n(counterfactual)",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def chance_line(ax, x_text, ha="left"):
    ax.axhline(0.5, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(x_text, 0.515, "chance", color=INK_2, ha=ha, va="bottom", fontsize=9)


def accuracy_by_split(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width, gap = 0.2, 0.02
    order = ["unchanged", "random_equal_norm", "task_vector", "fine_tune"]
    for i, m in enumerate(order):
        label, colour = METHODS[m]
        xs = [k + (i - 1.5) * width for k in range(len(SPLITS))]
        means = [summary.loc[m, (c, "mean")] for c in SPLITS]
        sds = [summary.loc[m, (c, "std")] for c in SPLITS]
        ax.bar(xs, means, width - gap, color=colour, label=label, edgecolor=SURFACE, linewidth=1, zorder=2)
        ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
    chance_line(ax, -0.48)
    ax.set_xticks(range(len(SPLITS)), SPLITS.values())
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy on hidden sentences")
    ax.grid(axis="x", visible=False)
    ax.set_title("Accuracy by test set (mean of 3 broken models, bars show ±1 sd)")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_by_split.png", dpi=200)
    plt.close(fig)


def regression_by_method(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    order = ["fine_tune", "unchanged", "random_equal_norm", "task_vector"]
    vals = [summary.loc[m, ("prefix_nll_delta", "mean")] for m in order]
    sds = [summary.loc[m, ("prefix_nll_delta", "std")] for m in order]
    ax.barh(range(len(order)), vals, 0.6, xerr=sds, color=[METHODS[m][1] for m in order],
            error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=2), zorder=2)
    for y, v in enumerate(vals):
        ax.text(max(v, 0) + sds[y] + 0.1, y, f"{v:+.2f}", va="center", color=INK, fontsize=9)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(range(len(order)), [METHODS[m][0] for m in order])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 4.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in loss on the words before the adjective (nats/token)")
    ax.set_title("Damage to the rest of the model (higher is worse; ±1 sd)")
    fig.tight_layout()
    fig.savefig(OUT / "regression_by_method.png", dpi=200)
    plt.close(fig)


def tradeoff(summary: pd.DataFrame, sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in ["random_equal_norm", "task_vector"]:
        label, colour = METHODS[m]
        s = sweep.loc[m].sort_index()
        x, y = s[("prefix_nll_delta", "mean")], s[("test_id_acc", "mean")]
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=f"{label} (α = 0.1 to 4)",
                zorder=3)
    tv = sweep.loc["task_vector"]
    offsets = {0.5: (-8, -3, "right"), 1.0: (-8, -3, "right"), 1.5: (6, -14, "left"),
               2.0: (0, 9, "center"), 3.0: (0, 9, "center"), 4.0: (0, 9, "center")}
    for a, (dx, dy, ha) in offsets.items():
        ax.annotate(f"α={a:g}", (tv.loc[a, ("prefix_nll_delta", "mean")], tv.loc[a, ("test_id_acc", "mean")]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8.5, color=INK_2)
    for m, offset in [("fine_tune", (8, -4)), ("unchanged", (10, -14))]:
        label, colour = METHODS[m]
        x, y = summary.loc[m, ("prefix_nll_delta", "mean")], summary.loc[m, ("test_id_acc", "mean")]
        ax.scatter([x], [y], s=70, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=4, label=label)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=offset, fontsize=9, color=INK)
    chance_line(ax, 3.75, "right")
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_xlim(-0.45, 3.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Damage: change in loss on the words before the adjective (nats/token)")
    ax.set_ylabel("Accuracy on familiar sentences")
    ax.set_title("Accuracy against damage as the edit strength α grows")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_damage.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "summary.csv", header=[0, 1], index_col=0)
    sweep = pd.read_csv(RESULTS / "alpha_sweep_summary.csv", header=[0, 1], index_col=[0, 1])
    accuracy_by_split(summary)
    regression_by_method(summary)
    tradeoff(summary, sweep)
    print(f"Figures written to {OUT}/")


if __name__ == "__main__":
    main()
