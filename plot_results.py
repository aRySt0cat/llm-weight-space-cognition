"""Draw the actual digit predictions against the targets after scoring."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from trials import MODELS, ROOT, TIERS


def draw():
    with (ROOT / "results/trials.csv").open() as f:
        data = {(r["model"], r["tier"], r["target"]): r for r in csv.DictReader(f) if r["tier"] != "edit"}
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.8), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.18, w_pad=0.18)
    colors = {"correct": "#096b62", "wrong": "#e6ecf1", "invalid": "#f5dcbd", "missing": "#f6f7f9"}
    for ax, model in zip(axes, MODELS, strict=True):
        ax.set_title(model, loc="left", fontweight="bold")
        for y, tier in enumerate(TIERS):
            for x in range(10):
                row = data[model, tier, str(x)]
                if row["delivered"] != "True":
                    kind, label = "missing", "–"
                elif row["valid_answer"] != "True":
                    kind, label = "invalid", "?"
                else:
                    kind = "correct" if row["correct"] == "True" else "wrong"
                    label = row["guess"]
                ax.add_patch(Rectangle((x-.49, y-.485), .98, .97, facecolor=colors[kind], edgecolor="white"))
                ax.text(x, y, label, ha="center", va="center", fontsize=13,
                        color="white" if kind == "correct" else "#182d41")
        ax.set_xlim(-.5, 9.5)
        ax.set_ylim(4.5, -.5)
        ax.set_xticks(range(10), [str(i) for i in range(10)])
        ax.set_yticks(range(5), ["197", "793", "6,337", "7,393", "33,537"])
        ax.set_xlabel("True target digit")
        ax.set_ylabel("Parameters")
        ax.tick_params(length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors.values(),
                        ["Correct digit", "Wrong digit", "Invalid answer", "Unverified delivery"], strict=True)],
                    loc="upper center", bbox_to_anchor=(.5, -.24), ncol=4, frameon=False)
    fig.suptitle("Reading MNIST classifiers from their weights", fontsize=18, fontweight="bold")
    fig.savefig(ROOT / "results/predictions.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    draw()
