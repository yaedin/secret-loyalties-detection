"""Figures for the submission report. All numbers come from the stored artifacts;
nothing is hand-entered.

    python experiments/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "writeup" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
E2 = REPO / "results" / "E2_matched"
E4 = REPO / "results" / "E4"
L = "L27"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
BLUE, RED, GREY = "#1f4e79", "#a83232", "#8a8a8a"


# ---------------------------------------------------------------- Figure 1
def figure1() -> None:
    """Design schematic: three arms, four evidence layers."""
    fig, ax = plt.subplots(figsize=(6.6, 2.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                    linewidth=0.6, edgecolor="#4a4a4a",
                                    facecolor=fc))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top",
                fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h - 8.2, body, ha="center", va="top", fontsize=6.9,
                linespacing=1.45)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.8,
                                     color="#4a4a4a"))

    box(1, 6, 21, 31, "Model arms",
        "organism_a\norganism_b\norganism_c (control)\n\nControl is byte-identical\n"
        "to the base model, so the\ndifference isolates the\nfine-tune", "#dce6f1")
    box(27, 6, 30, 31, "Evidence layers",
        "1  Behaviour (black box)\n2  Weights (static)\n3  Activations, prompt\n"
        "4  Activations, generation\n\nEach layer observes a strict\n"
        "superset of the previous one", "#eaeaea")
    box(62, 6, 37, 31, "Analysis",
        "Difference vector d(x) = h_org(x) − h_ctrl(x)\n"
        "Suppression score s(x) = −(d(x) − d̄) · e\n"
        "e = base ethical-alarm direction\n\n"
        "Every statistic paired with a control-self\n"
        "null and a matched-norm random direction",
        "#dce6f1")
    arrow(22.5, 26.5, 21.5)
    arrow(57.5, 61.5, 21.5)
    fig.savefig(OUT / "figure1_design.png")
    plt.close(fig)
    print("figure1_design.png")


# ---------------------------------------------------------------- Figure 2
def figure2() -> dict:
    """Dose-response: suppression against the base model's ethical alarm."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    stats = {}
    for ax, org, lab in zip(axes, ("organism_a", "organism_b"),
                            ("Organism A", "Organism B")):
        for tag, colour, name, mk in (("corpus_scan", BLUE, "benign (WildChat)", "o"),
                                      ("harmful_scan", RED, "harmful benchmarks", "^")):
            c = np.load(E2 / f"{tag}_organism_c.npz")
            o = np.load(E2 / f"{tag}_{org}.npz")
            alarm = c[f"proj_{L}"][:, 0]
            s = -(o[f"proj_{L}"][:, 0] - c[f"proj_{L}"][:, 0])
            r2 = float(np.corrcoef(s, alarm)[0, 1] ** 2)
            stats[(org, tag)] = {"r2": r2, "n": len(s),
                                 "slope": float(np.polyfit(alarm, s, 1)[0]),
                                 "alarm_mean": float(alarm.mean())}
            idx = (np.random.default_rng(0).choice(len(s), min(2500, len(s)),
                                                   replace=False))
            ax.scatter(alarm[idx], s[idx], s=1.4, alpha=.16, color=colour,
                       marker=mk, linewidths=0, rasterized=True)
            xs = np.linspace(alarm.min(), alarm.max(), 50)
            b1, b0 = np.polyfit(alarm, s, 1)
            ax.plot(xs, b1 * xs + b0, color=colour, linewidth=1.3,
                    label=f"{name}: $R^2$ = {r2:.3f}")
        ax.set_title(lab)
        ax.set_xlabel("Base model ethical alarm,  $h_{base}(x)\\cdot e$")
        ax.grid(alpha=.25, linewidth=.4)
        ax.legend(frameon=False, loc="upper left", handlelength=1.4)
    axes[0].set_ylabel("Suppression by the fine-tune,  $s(x)$")
    fig.savefig(OUT / "figure2_dose_response.png")
    plt.close(fig)
    print("figure2_dose_response.png")
    return stats


# ---------------------------------------------------------------- Figure 3
def figure3() -> pd.DataFrame:
    """Damper gain and explained variance across generated token positions."""
    t = pd.read_csv(E4 / "e4_trajectory_L27.csv")
    t = t[t.teacher == "organism_c"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharex=True)
    x = np.arange(len(t[t.organism == "organism_a"]))
    labels = ["0\n(prompt)", "1", "2–4", "5–8", "9–16", "17–32", "33–64", "65–128"]
    for org, colour, mk, lab in (("organism_a", BLUE, "o", "Organism A"),
                                 ("organism_b", RED, "s", "Organism B")):
        g = t[t.organism == org]
        axes[0].plot(x, g["k"], marker=mk, ms=3, color=colour, label=lab)
        axes[1].plot(x, g["r2"], marker=mk, ms=3, color=colour, label=lab)
    for ax, ylab in zip(axes, ["Damper gain  $k$", "Explained variance  $R^2$"]):
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.6)
        ax.set_xlabel("Generated token position")
        ax.set_ylabel(ylab)
        ax.grid(alpha=.25, linewidth=.4)
        ax.axvline(0.5, color=GREY, linestyle=":", linewidth=.8)
        ax.set_ylim(0, None)
    axes[0].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.annotate("generation begins", xy=(0.5, ax.get_ylim()[1] * 0.985),
                    xytext=(1.15, ax.get_ylim()[1] * 0.985), fontsize=6.4,
                    color=GREY, va="top", ha="left")
    fig.savefig(OUT / "figure3_generation_decay.png")
    plt.close(fig)
    print("figure3_generation_decay.png")
    return t


def main() -> None:
    figure1()
    s = figure2()
    t = figure3()
    summary = {
        "figure2_dose_response": {f"{o}|{c}": v for (o, c), v in s.items()},
        "figure3_trajectory": t[["organism", "lo", "hi", "n", "k", "r2",
                                 "mean_alarm", "sd_alarm"]].to_dict("records"),
    }
    (OUT / "figure_values.json").write_text(json.dumps(summary, indent=1))
    print("\n--- values used in captions ---")
    for k, v in summary["figure2_dose_response"].items():
        print(f"  {k:34s} R2={v['r2']:.3f}  slope={v['slope']:+.3f}  "
              f"n={v['n']}  alarm_mean={v['alarm_mean']:+.1f}")


if __name__ == "__main__":
    main()
