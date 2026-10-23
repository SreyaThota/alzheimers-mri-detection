"""Shared constants. Class order is alphabetical, matching the dataset folder names."""

CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

DISPLAY_NAMES = {
    "MildDemented": "Mild Dementia",
    "ModerateDemented": "Moderate Dementia",
    "NonDemented": "No Dementia",
    "VeryMildDemented": "Very Mild Dementia",
}

# Native size of the dataset images (width 176, height 208), single grayscale channel.
IMG_HEIGHT = 208
IMG_WIDTH = 176
IMG_CHANNELS = 1

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DEFAULT_MODEL_PATH = "models/alzmri_cnn.keras"
SEED = 42
