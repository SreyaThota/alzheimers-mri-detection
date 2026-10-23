"""A compact CNN for 4-class MRI classification."""

import keras
from keras import layers

from .config import CLASS_NAMES, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH

LAST_CONV_LAYER = "last_conv"


def conv_block(x, filters: int, name: str):
    # No batch normalization: in early runs its train/inference mismatch made validation unstable.
    return layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=name)(x)


def build_model(dropout: float = 0.3) -> keras.Model:
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image")

    # Light augmentation, active only during training.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomTranslation(0.05, 0.05, fill_mode="constant")(x)
    x = layers.RandomZoom(0.05, fill_mode="constant")(x)

    # Halve the resolution first (208x176 -> 104x88) so training stays practical on a CPU.
    x = layers.AveragePooling2D(2, name="downsample")(x)

    for i, filters in enumerate([32, 64, 128]):
        x = conv_block(x, filters, name=f"conv{i + 1}")
        x = layers.MaxPooling2D()(x)
    x = conv_block(x, 256, name=LAST_CONV_LAYER)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="alzmri_cnn")
