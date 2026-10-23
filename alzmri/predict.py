"""Load a trained model and turn an image into class probabilities."""

from dataclasses import dataclass

import keras
import numpy as np

from .config import CLASS_NAMES, DISPLAY_NAMES
from .preprocessing import load_image_bytes


@dataclass
class Prediction:
    label: str
    display_name: str
    confidence: float
    probabilities: dict[str, float]


def load_model(path: str) -> keras.Model:
    return keras.models.load_model(path)


def predict_array(model: keras.Model, image: np.ndarray) -> Prediction:
    """Predict from one preprocessed image of shape (H, W, 1)."""
    probs = np.asarray(model(image[np.newaxis], training=False))[0]
    best = int(np.argmax(probs))
    return Prediction(
        label=CLASS_NAMES[best],
        display_name=DISPLAY_NAMES[CLASS_NAMES[best]],
        confidence=float(probs[best]),
        probabilities={DISPLAY_NAMES[c]: float(p) for c, p in zip(CLASS_NAMES, probs)},
    )


def predict_bytes(model: keras.Model, data: bytes) -> Prediction:
    return predict_array(model, load_image_bytes(data))
