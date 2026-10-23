"""Train the CNN on the train/ folder, holding out a validation split. The test/ folder is never read here.

Usage: python -m alzmri.train --data-dir path/to/Alzheimer_s_Dataset
"""

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from .config import CLASS_NAMES, DEFAULT_MODEL_PATH, SEED
from .data import group_split, list_split, load_arrays, pixel_hash
from .model import build_model


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Square root of balanced weights, from the training split only.
    Fully balanced weights (about 25x for Moderate Dementia) made training collapse onto that class."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {k: float(np.sqrt(len(labels) / (len(CLASS_NAMES) * c))) for k, c in enumerate(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ (test/ is not used)")
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=70)  # early stopping usually ends it sooner
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    keras.utils.set_random_seed(SEED)

    paths, labels = list_split(Path(args.data_dir) / "train")
    groups = [pixel_hash(p) for p in paths]  # identical images stay on the same side of the split
    train_idx, val_idx = group_split(labels, groups, args.val_fraction)
    x = load_arrays(paths)
    x_train, y_train, x_val, y_val = x[train_idx], labels[train_idx], x[val_idx], labels[val_idx]
    weights = class_weights(y_train)
    print(f"train={len(y_train)} val={len(y_val)} class_weights={weights}")

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(out, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )

    log = {
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "class_weights": weights,
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    log_path = Path("results/training_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Saved best model to {out} and training log to {log_path}")


if __name__ == "__main__":
    main()
