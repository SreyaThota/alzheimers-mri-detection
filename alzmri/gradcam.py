"""Grad-CAM heatmaps showing which image regions most influenced a prediction.

Grad-CAM is a visual explanation aid. It does not show that a prediction is clinically correct.

Usage: python -m alzmri.gradcam --model models/alzmri_cnn.keras --images img1.jpg img2.jpg --out results/gradcam
"""

import argparse
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .model import LAST_CONV_LAYER
from .predict import predict_array
from .preprocessing import load_image_file


def gradcam(model: keras.Model, image: np.ndarray, class_index: int | None = None) -> np.ndarray:
    """Return a heatmap in [0, 1] with the same height and width as `image` (H, W, 1)."""
    grad_model = keras.Model(model.inputs, [model.get_layer(LAST_CONV_LAYER).output, model.output])
    batch = tf.convert_to_tensor(image[np.newaxis])
    with tf.GradientTape() as tape:
        conv_out, probs = grad_model(batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(probs[0]))
        score = probs[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1))[0]
    cam = tf.image.resize(cam[..., None], image.shape[:2])[..., 0].numpy()
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def save_overlay(image: np.ndarray, heatmap: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.6), dpi=120)
    axes[0].imshow(image[..., 0], cmap="gray")
    axes[0].set_title("Input", fontsize=9)
    axes[1].imshow(image[..., 0], cmap="gray")
    axes[1].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[1].set_title(title, fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out", default="results/gradcam")
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in map(Path, args.images):
        image = load_image_file(image_path)
        prediction = predict_array(model, image)
        heatmap = gradcam(model, image)
        target = out_dir / f"{image_path.parent.name}_{image_path.stem}_gradcam.png"
        save_overlay(image, heatmap, f"Grad-CAM: {prediction.display_name} ({prediction.confidence:.0%})", target)
        print(f"{image_path} -> {prediction.display_name} ({prediction.confidence:.2%}), saved {target}")


if __name__ == "__main__":
    main()
