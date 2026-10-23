"""Evaluate a model on the held-out test/ folder and save metrics and a confusion matrix.

Usage:
  python -m alzmri.evaluate --data-dir DATA --model models/alzmri_cnn.keras --name cnn
  python -m alzmri.evaluate --data-dir DATA --legacy-savedmodel path/to/model --name original_brain_mri
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import CLASS_NAMES, DISPLAY_NAMES
from .data import list_split, load_arrays


def predict_keras(model_path: str, paths: list[Path], batch_size: int = 64) -> np.ndarray:
    import keras

    model = keras.models.load_model(model_path)
    return model.predict(load_arrays(paths), batch_size=batch_size, verbose=0)


def predict_legacy_savedmodel(model_dir: str, paths: list[Path], batch_size: int = 64) -> np.ndarray:
    """Run the original project's TensorFlow SavedModel, using its own preprocessing:
    RGB, resized to 176x176, raw 0-255 pixel values.
    That model was saved with Keras 2, so loading it needs the optional `tf-keras` package."""
    try:
        import tf_keras
    except ImportError as exc:
        raise SystemExit("Evaluating the legacy model needs: pip install tf-keras") from exc

    model = tf_keras.models.load_model(model_dir, compile=False)
    probs = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                batch.append(np.asarray(image.convert("RGB").resize((176, 176), Image.NEAREST), dtype=np.float32))
        probs.append(model.predict(np.stack(batch), verbose=0))
    return np.concatenate(probs)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "test_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(macro[0]), 4),
        "macro_recall": round(float(macro[1]), 4),
        "macro_f1": round(float(macro[2]), 4),
        "per_class": {
            DISPLAY_NAMES[c]: {
                "precision": round(float(precision[k]), 4),
                "recall": round(float(recall[k]), 4),
                "f1": round(float(f1[k]), 4),
                "support": int(support[k]),
            }
            for k, c in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_order": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    names = [DISPLAY_NAMES[c].replace(" Dementia", "\nDementia") for c in CLASS_NAMES]
    row_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=120)
    ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{row_norm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if row_norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="Folder containing test/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Path to a .keras model trained with alzmri.train")
    group.add_argument("--legacy-savedmodel", help="Path to the original project's SavedModel folder")
    parser.add_argument("--name", required=True, help="Prefix for the output files in results/")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    paths, y_true = list_split(Path(args.data_dir) / "test")
    if args.model:
        probs = predict_keras(args.model, paths)
    else:
        probs = predict_legacy_savedmodel(args.legacy_savedmodel, paths)
    y_pred = probs.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y_true, y_pred, target_names=[DISPLAY_NAMES[c] for c in CLASS_NAMES],
                                   digits=4, zero_division=0)
    (out_dir / f"{args.name}_classification_report.txt").write_text(report)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), args.title or args.name,
                          out_dir / f"{args.name}_confusion_matrix.png")
    print(report)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}, indent=2))


if __name__ == "__main__":
    main()
