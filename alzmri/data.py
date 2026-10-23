"""Dataset listing, duplicate detection and a leakage-safe train/validation split."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, SEED
from .preprocessing import load_image_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_split(split_dir: str | Path) -> tuple[list[Path], np.ndarray]:
    """Return image paths and integer labels for a folder laid out as <split>/<class>/<image>."""
    split_dir = Path(split_dir)
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
                labels.append(label)
    return paths, np.array(labels, dtype=np.int64)


def load_arrays(paths: list[Path]) -> np.ndarray:
    return np.stack([load_image_file(p) for p in paths])


def pixel_hash(path: Path) -> str:
    """Hash of the decoded pixels, so re-encoded copies of the same image still match."""
    with Image.open(path) as image:
        return hashlib.md5(image.convert("L").tobytes()).hexdigest()


def thumbnails(paths: list[Path], size: tuple[int, int] = (22, 26)) -> np.ndarray:
    """Small grayscale thumbnails in [0, 1], flattened, for near-duplicate search."""
    out = np.empty((len(paths), size[0] * size[1]), dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            out[i] = np.asarray(image.convert("L").resize(size, Image.BILINEAR), dtype=np.float32).ravel() / 255.0
    return out


def nearest_distances(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """For each query row, the mean absolute difference to its closest reference row and that row's index."""
    best = np.empty(len(query), dtype=np.float32)
    best_idx = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        dist = np.abs(block[:, None, :] - reference[None, :, :]).mean(axis=2)
        best_idx[start:start + chunk] = dist.argmin(axis=1)
        best[start:start + chunk] = dist.min(axis=1)
    return best, best_idx


def group_split(labels: np.ndarray, groups: list[str], val_fraction: float, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/validation split that keeps identical images (same group) on one side."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        class_idx = np.flatnonzero(labels == label)
        unique_groups = sorted({groups[i] for i in class_idx})
        rng.shuffle(unique_groups)
        n_val = max(1, round(len(unique_groups) * val_fraction))
        val_groups = set(unique_groups[:n_val])
        for i in class_idx:
            (val_idx if groups[i] in val_groups else train_idx).append(i)
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))
