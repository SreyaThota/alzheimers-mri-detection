"""Image loading and preprocessing shared by training, evaluation and the web app."""

import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import IMG_HEIGHT, IMG_WIDTH


class InvalidImageError(ValueError):
    """Raised when an input cannot be decoded as an image."""


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a float32 array of shape (H, W, 1) scaled to [0, 1]."""
    image = image.convert("L")
    if image.size != (IMG_WIDTH, IMG_HEIGHT):
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[..., np.newaxis]


def load_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes and preprocess them. Raises InvalidImageError on bad input."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return preprocess_image(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The file could not be read as an image.") from exc


def load_image_file(path: str | Path) -> np.ndarray:
    return load_image_bytes(Path(path).read_bytes())
