"""Rebuild report figures from saved numbers; does not train or run detectors.

python experiments/detection/render_figures.py
Requires matplotlib. The original YOLO PR/F1 images are preserved separately.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
COLORS = ["#285a8e", "#d17a28", "#278574", "#9167a8"]


def learning_curves(folder):
    rows = list(csv.DictReader((folder / "training_history.csv").open()))
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
        for name, label in [
            ("loss", "total"), ("box_loss", "box"), ("cls_loss", "classification"),
            ("l1_loss", "L1"), ("loss_cls", "classification"), ("loss_bbox", "box")
        ]:
            points = [(float(r["step"]), float(r["value"])) for r in rows
                      if r["namespace"] == "train" and r["metric"] == name]
            if points:
                axes[0].plot(*zip(*points), label=label, linewidth=1.3,
                             color=COLORS[len(axes[0].lines) % len(COLORS)])
        for index, (name, label) in enumerate([
            ("map_50_95", "mAP50–95"), ("ap50", "AP50"), ("ap75", "AP75")
        ]):
            points = [(float(r["step"]), float(r["value"])) for r in rows
                      if r["namespace"] == "val" and r["metric"] == name]
            if points:
                axes[1].plot(*zip(*points), label=label, linewidth=1.7, color=COLORS[index])
        train_axis = next(r["axis"] for r in rows if r["namespace"] == "train")
        axes[0].set(title="Training loss", xlabel=train_axis.capitalize(), ylabel="Loss")
        axes[1].set(title="Validation", xlabel="Epoch", ylabel="AP", ylim=(0, 1))
        for ax in axes:
            ax.grid(alpha=0.17)
            ax.legend(frameon=False)
        fig.savefig(folder / "learning_curves.png", dpi=180)
        plt.close(fig)


def threshold_figures(folder):
    path = folder / "threshold_curve.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    with plt.rc_context({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
        ax.plot([float(r["recall"]) for r in reversed(rows)],
                [float(r["precision"]) for r in reversed(rows)], color=COLORS[0], linewidth=2)
        ax.set(title="RTMDet-s · confidence sweep · IoU 0.5", xlabel="Recall", ylabel="Precision",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.17)
        fig.savefig(folder / "precision_recall.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
        for index, name in enumerate(["precision", "recall", "f1"]):
            ax.plot([float(r["threshold"]) for r in rows], [float(r[name]) for r in rows],
                    label=name, color=COLORS[index], linewidth=2)
        ax.set(title="RTMDet-s · validation · IoU 0.5", xlabel="Confidence threshold",
               ylabel="Value", xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.17)
        ax.legend(frameon=False)
        fig.savefig(folder / "confidence_curves.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    for folder in [ROOT / "yolo/v1", ROOT / "yolo/v2", ROOT / "RTMDet"]:
        learning_curves(folder)
        threshold_figures(folder)
        print(folder.relative_to(ROOT), "figures saved")
