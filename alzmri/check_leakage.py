"""Check the train and test folders for exact and near-duplicate images.

Usage: python -m alzmri.check_leakage --data-dir path/to/Alzheimer_s_Dataset
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES
from .data import list_split, nearest_distances, pixel_hash, thumbnails


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ and test/")
    parser.add_argument("--out", default="results/leakage_check.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_paths, train_labels = list_split(data_dir / "train")
    test_paths, test_labels = list_split(data_dir / "test")

    train_hashes = [pixel_hash(p) for p in train_paths]
    test_hashes = [pixel_hash(p) for p in test_paths]
    train_hash_set = set(train_hashes)

    exact_cross = [i for i, h in enumerate(test_hashes) if h in train_hash_set]
    within_train = sum(c - 1 for c in Counter(train_hashes).values() if c > 1)
    within_test = sum(c - 1 for c in Counter(test_hashes).values() if c > 1)

    nn_dist, nn_idx = nearest_distances(thumbnails(test_paths), thumbnails(train_paths))
    percentiles = {f"p{q}": round(float(np.percentile(nn_dist, q)), 4) for q in (1, 5, 25, 50)}
    cross_label_exact = sum(1 for i in exact_cross if test_labels[i] != train_labels[train_hashes.index(test_hashes[i])])

    report = {
        "train_images": len(train_paths),
        "test_images": len(test_paths),
        "train_per_class": {c: int((train_labels == k).sum()) for k, c in enumerate(CLASS_NAMES)},
        "test_per_class": {c: int((test_labels == k).sum()) for k, c in enumerate(CLASS_NAMES)},
        "exact_duplicates_within_train": within_train,
        "exact_duplicates_within_test": within_test,
        "test_images_identical_to_a_train_image": len(exact_cross),
        "of_which_with_different_label": cross_label_exact,
        "nearest_train_neighbour_mean_abs_diff": percentiles,
        "note": "Distances are mean absolute pixel differences (0-1 scale) between 22x26 thumbnails. "
                "The dataset has no subject IDs, so subject-level overlap cannot be checked.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
