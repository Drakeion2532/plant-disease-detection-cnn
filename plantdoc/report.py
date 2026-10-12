"""Test-set metrics, per-class accuracy, confusion matrix and training curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

ROOT = Path(__file__).resolve().parents[1]


def main(cache=ROOT / "data" / "cache"):
    res = ROOT / "results"
    classes = json.loads((Path(cache) / "classes.json").read_text())
    y = np.load(Path(cache) / "y.npy").astype(int)[np.load(res / "test_indices.npy")]
    p = np.load(res / "test_probs.npy").astype(np.float32)
    pred = p.argmax(1)
    rep = classification_report(y, pred, target_names=classes, output_dict=True, zero_division=0)
    summary = {"test_images": int(len(y)), "accuracy": round(rep["accuracy"], 4),
               "top3_accuracy": round(float(top_k_accuracy_score(y, p, k=3, labels=range(len(classes)))), 4),
               "macro_f1": round(rep["macro avg"]["f1-score"], 4),
               "weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
               "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in classes}}
    worst = sorted(summary["per_class_f1"].items(), key=lambda kv: kv[1])[:5]
    summary["hardest_classes"] = dict(worst)
    (res / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)
    hist = json.loads((res / "history.json").read_text())
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_acc"] for h in hist], label="train"); ax[0].plot(ep, [h["val_acc"] for h in hist], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_loss"] for h in hist], label="train"); ax[1].plot(ep, [h["val_loss"] for h in hist], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "training_curves.png", dpi=110); plt.close(fig)

    cm = confusion_matrix(y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    short = [c.replace("___", ": ").replace("_", " ")[:34] for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), short, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.colorbar(im, fraction=0.03)
    ax.set_title(f"Normalised confusion matrix – test accuracy {summary['accuracy']:.2%}")
    fig.tight_layout(); fig.savefig(fig_dir / "confusion_matrix.png", dpi=110); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class_f1"}, indent=2))


if __name__ == "__main__":
    main()
