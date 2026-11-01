"""Plot cell-level metrics, NTA ROC/PR curves and paired guidance differences."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from ptanta import CLASSES, load_json, read_csv, sha256
from summarize import METRICS, TASKS

VARIANTS = {"unguided": ("Unguided", "#237ca1"), "prototype_guided": ("Prototype-guided", "#ba6333")}


def save(fig, path):
    for extension in ["png", "pdf"]:
        fig.savefig(path.with_suffix("." + extension), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--model", default="dinov2_finetuned")
    args = ap.parse_args()
    root = Path(args.experiment_root).resolve()
    metadata = load_json(root / "summary/summary.json")
    plan = load_json(root / "experiment_plan.json")
    scores = read_csv(root / "summary/metric_summary.csv")
    scores = scores[scores.model == args.model]
    if scores.empty:
        raise ValueError("Requested model is not in the summary.")
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, layout="constrained")
    for ax, (task, title) in zip(axes, TASKS.items()):
        for i, (variant, (label, color)) in enumerate(VARIANTS.items()):
            rows = scores[scores.variant == variant].set_index("metric").loc[[f"{task}_{m}" for m in METRICS]]
            x = np.arange(len(METRICS)) + (i - 0.5) * 0.12
            ax.scatter(x, rows["mean"], color=color, label=label)
            finite = rows["sd"].notna().to_numpy()
            ax.errorbar(x[finite], rows["mean"].to_numpy()[finite], yerr=rows["sd"].to_numpy()[finite],
                        fmt="none", ecolor=color, capsize=3)
        ax.set(title=title, xticks=np.arange(4), xticklabels=["Accuracy", "F1", "AUROC", "AP"], ylim=(0, 1.05))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Out-of-fold score; error bars: seed SD")
    axes[0].legend(fontsize=8)
    fig.suptitle("SYNTHETIC SOFTWARE TEST" if metadata["synthetic"] else "PTA-NTA cell-level performance")
    save(fig, out / "cell_performance")
    seed = plan["seeds"][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for variant, (label, color) in VARIANTS.items():
        frames = []
        for job in plan["jobs"]:
            if job["model"] != args.model or job["seed"] != seed:
                continue
            path = Path(job["directory"]) / "evaluation" / f"{variant}_test.csv.gz"
            if sha256(path) != metadata["prediction_sha256"][path.relative_to(root).as_posix()]:
                raise ValueError("Predictions changed after summary.")
            frames.append(read_csv(path))
        df = pd.concat(frames, ignore_index=True)
        if len(df) != metadata["n_cells"] or not df.cell_id.is_unique:
            raise ValueError("Curve cohort differs from the summary.")
        y = df.true_class.eq(CLASSES[2])
        if y.nunique() != 2:
            raise ValueError("ROC/PR curves require both classes.")
        fpr, tpr, _ = roc_curve(y, df.nta_score)
        precision, recall, _ = precision_recall_curve(y, df.nta_score)
        axes[0].plot(fpr, tpr, color=color, label=f"{label}: {roc_auc_score(y, df.nta_score):.3f}")
        axes[1].plot(recall, precision, color=color, label=f"{label}: {average_precision_score(y, df.nta_score):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    axes[1].axhline(y.mean(), color="0.6", linestyle="--", lw=1)
    for ax, title, xlabel, ylabel in zip(axes, ["NTA ROC", "NTA precision-recall"],
                                         ["False-positive rate", "Recall"], ["True-positive rate", "Precision"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.suptitle(f"First prespecified seed: {seed}" + (" | SYNTHETIC" if metadata["synthetic"] else ""))
    save(fig, out / "nta_curves")
    differences = read_csv(root / "summary/guidance_difference.csv")
    differences = differences[differences.model == args.model].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    for i, row in enumerate(differences.itertuples()):
        ax.scatter(row.mean_delta, i, color="#237ca1")
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            ax.plot([row.ci_low, row.ci_high], [i, i], color="#237ca1")
    ax.axvline(0, color="0.5", linestyle="--")
    ax.set(yticks=np.arange(len(differences)),
           yticklabels=[f"{METRICS[r.metric.removeprefix('nta_')]} (n={r.n_patients})" for r in differences.itertuples()],
           xlabel="Guided minus unguided; exploratory patient-paired 95% interval",
           title="NTA guidance effect" + (" | SYNTHETIC" if metadata["synthetic"] else ""))
    save(fig, out / "guidance_difference")
    print(f"Wrote three PNG/PDF figures: {out}")


if __name__ == "__main__":
    main()
