"""Regenerate the three bundled PTA-NTA figures from the included aggregate files."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#237ca1", "#24834d", "#9b4c9f", "#bd8a16"]


def save(fig, name):
    out = ROOT / "results/charts"
    out.mkdir(exist_ok=True)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def framework():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set(xlim=(0, 12), ylim=(0, 3.6))
    ax.axis("off")
    boxes = [(0.1, 2.25, 2, .8, "Reviewed IHC\ncell objects"),
             (2.6, 2.25, 2, .8, "Cell crops\n224 x 224"),
             (5.1, 2.25, 2.4, .8, "DINOv2\nthree-class classifier"),
             (8, 2.25, 3.8, .8, "Tumor AI / PTA /\nunguided NTA"),
             (5.1, .55, 2.4, .9, "Predicted positive-tumor\nmorphology prototype"),
             (8, .55, 3.8, .9, "Prototype-guided NTA\nUnstained tumor detection")]
    for x, y, w, h, label in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#f2f6f7", edgecolor="#425963"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    for start, end in [((2.1, 2.65), (2.6, 2.65)), ((4.6, 2.65), (5.1, 2.65)),
                       ((7.5, 2.65), (8, 2.65)), ((6.3, 2.25), (6.3, 1.45)),
                       ((7.5, 1), (8, 1)), ((9.9, 2.25), (9.9, 1.45))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#425963"))
    ax.text(6, .05, "Fallback to unguided NTA when anchors are insufficient", ha="center", fontsize=10)
    ax.set_title("PTA-NTA | cell-level workflow", loc="left", fontsize=15, pad=12)
    save(fig, "framework")


def reported():
    df = pd.read_csv(ROOT / "results/reported_metrics.csv")
    metrics = ["accuracy", "balanced_accuracy", "f1", "auroc", "pr_score"]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for i, row in df.iterrows():
        ax.scatter(np.arange(5) + (i - 1.5) * .13, row[metrics].to_numpy(float),
                   color=COLORS[i], s=35, label=row["method"])
    ax.set(xticks=np.arange(5), xticklabels=["Accuracy", "Balanced accuracy", "F1", "AUROC", "AP"],
           ylim=(.5, 1.025), ylabel="Reported score")
    ax.grid(axis="y", alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), ncol=2, loc="upper center",
               bbox_to_anchor=(.5, .90), frameon=False)
    fig.suptitle("Paper-reported PTA-NTA performance", fontsize=14)
    fig.text(.03, .03, "Descriptive aggregates; task targets differ. Tumor AI AUROC is unavailable.\n"
             "Matched exports and confidence intervals are not available.", fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .79))
    save(fig, "reported_performance")


def ablation():
    df = pd.read_csv(ROOT / "results/guidance_ablation.csv")
    methods = {"ptanta_no_prototype": "Unguided", "ptanta_prototype": "Prototype-guided"}
    metrics = ["accuracy", "f1", "auroc", "ap"]
    if set(df.method) != set(methods) or df.duplicated(["method", "task", "fold", "seed"]).any():
        raise ValueError("Unexpected methods or duplicate ablation rows.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, task, title in zip(axes, ["nta_all_cells", "nta_tumor_only"],
                               ["NTA: all cells", "NTA: reference tumor cells"]):
        for i, (method, name) in enumerate(methods.items()):
            rows = df[(df.method == method) & (df.task == task)]
            if len(rows) != 9 or len(rows[["fold", "seed"]].drop_duplicates()) != 9:
                raise ValueError("Expected all nine fold-seed runs for every method/task.")
            ax.errorbar(np.arange(4) + (i - .5) * .14, rows[metrics].mean(),
                        yerr=rows[metrics].std(ddof=1), fmt="o", capsize=4, color=COLORS[i], label=name)
        ax.set(title=title, xticks=np.arange(4), xticklabels=["Accuracy", "F1", "AUROC", "AP"], ylim=(0, 1.1))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean +/- SD across fold-seed runs")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="upper center",
               bbox_to_anchor=(.5, .91), frameon=False)
    fig.suptitle("Recorded NTA guidance ablation", fontsize=14)
    fig.text(.03, .025, "3 folds x 3 seeds; SD is not a patient confidence interval.\n"
             "Exploratory recorded references; validation-selected thresholds.", fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .82))
    save(fig, "guidance_ablation")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    framework()
    reported()
    ablation()
    print("Wrote three PTA-NTA figures in PNG/SVG.")
