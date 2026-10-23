"""Tests for preprocessing, model output, Grad-CAM, duplicate handling and the web app.
They use synthetic images, so no dataset or trained model is needed."""

import io

import numpy as np
import pytest
from PIL import Image

from alzmri.config import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
from alzmri.data import group_split, list_split, nearest_distances, pixel_hash
from alzmri.evaluate import compute_metrics
from alzmri.gradcam import gradcam
from alzmri.model import build_model
from alzmri.predict import predict_array
from alzmri.preprocessing import InvalidImageError, load_image_bytes, preprocess_image
from webapp.app import create_app


def image_bytes(size=(IMG_WIDTH, IMG_HEIGHT), mode="L", fmt="JPEG", seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    channels = 3 if mode == "RGB" else 1
    pixels = rng.integers(0, 256, (size[1], size[0], channels), dtype=np.uint8).squeeze()
    buf = io.BytesIO()
    Image.fromarray(pixels, mode).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def model():
    return build_model()


# Preprocessing

@pytest.mark.parametrize("size,mode", [((IMG_WIDTH, IMG_HEIGHT), "L"), ((300, 300), "RGB"), ((64, 80), "L")])
def test_preprocess_shape_and_range(size, mode):
    array = load_image_bytes(image_bytes(size=size, mode=mode))
    assert array.shape == (IMG_HEIGHT, IMG_WIDTH, 1)
    assert array.dtype == np.float32
    assert 0.0 <= array.min() and array.max() <= 1.0


def test_preprocess_keeps_native_size_pixels():
    pixels = np.full((IMG_HEIGHT, IMG_WIDTH), 255, dtype=np.uint8)
    array = preprocess_image(Image.fromarray(pixels, "L"))
    assert np.allclose(array, 1.0)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\xff\xd8\xff\xe0 truncated jpeg"])
def test_invalid_bytes_raise(data):
    with pytest.raises(InvalidImageError):
        load_image_bytes(data)


# Model and Grad-CAM

def test_model_output_is_probability_vector(model):
    image = load_image_bytes(image_bytes())
    probs = np.asarray(model(image[np.newaxis], training=False))
    assert probs.shape == (1, len(CLASS_NAMES))
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_predict_array_returns_known_class(model):
    prediction = predict_array(model, load_image_bytes(image_bytes()))
    assert prediction.label in CLASS_NAMES
    assert len(prediction.probabilities) == len(CLASS_NAMES)
    assert 0.0 <= prediction.confidence <= 1.0


def test_gradcam_heatmap_shape_and_range(model):
    heatmap = gradcam(model, load_image_bytes(image_bytes()))
    assert heatmap.shape == (IMG_HEIGHT, IMG_WIDTH)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6


# Data handling

def test_list_split_and_duplicate_detection(tmp_path):
    for k, name in enumerate(CLASS_NAMES):
        (tmp_path / name).mkdir()
        (tmp_path / name / "a.jpg").write_bytes(image_bytes(seed=k))
    (tmp_path / CLASS_NAMES[0] / "copy.png").write_bytes(image_bytes(seed=0, fmt="PNG"))
    paths, labels = list_split(tmp_path)
    assert len(paths) == 5 and labels.tolist() == [0, 0, 1, 2, 3]
    # A PNG copy of a lossy JPEG is not pixel-identical, but is very close.
    jpg, png = tmp_path / CLASS_NAMES[0] / "a.jpg", tmp_path / CLASS_NAMES[0] / "copy.png"
    assert pixel_hash(jpg) != pixel_hash(png)
    (tmp_path / CLASS_NAMES[0] / "exact.jpg").write_bytes(jpg.read_bytes())
    assert pixel_hash(jpg) == pixel_hash(tmp_path / CLASS_NAMES[0] / "exact.jpg")


def test_group_split_keeps_duplicates_together():
    labels = np.array([0] * 10 + [1] * 10)
    groups = [f"g{i // 2}" for i in range(20)]  # pairs of identical images
    train_idx, val_idx = group_split(labels, groups, val_fraction=0.4)
    assert set(train_idx).isdisjoint(val_idx)
    assert {groups[i] for i in train_idx}.isdisjoint({groups[i] for i in val_idx})
    assert len(train_idx) + len(val_idx) == 20


def test_nearest_distances_finds_identical_row():
    reference = np.random.default_rng(0).random((50, 16)).astype(np.float32)
    dist, idx = nearest_distances(reference[[7, 20]], reference)
    assert np.allclose(dist, 0.0) and idx.tolist() == [7, 20]


def test_compute_metrics_macro_average():
    metrics = compute_metrics(np.array([0, 1, 2, 3, 3]), np.array([0, 1, 2, 3, 0]))
    assert metrics["accuracy"] == 0.8
    assert metrics["per_class"]["Very Mild Dementia"]["recall"] == 0.5
    assert len(metrics["confusion_matrix"]) == 4


# Web app

@pytest.fixture(scope="module")
def client(model):
    return create_app(model=model).test_client()


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"not for clinical diagnosis" in response.data


def test_upload_valid_image(client):
    response = client.post("/", data={"file": (io.BytesIO(image_bytes()), "scan.jpg")})
    assert response.status_code == 200
    assert b"Prediction:" in response.data


@pytest.mark.parametrize("data,expected", [
    ({}, b"Please choose an image file"),
    ({"file": (io.BytesIO(b"hello"), "notes.txt")}, b"Unsupported file type"),
    ({"file": (io.BytesIO(b"not an image"), "scan.jpg")}, b"could not be read as an image"),
    ({"file": (io.BytesIO(b""), "scan.jpg")}, b"empty"),
])
def test_upload_rejects_bad_input(client, data, expected):
    response = client.post("/", data=data)
    assert response.status_code == 400
    assert expected in response.data


def test_upload_too_large(client):
    big = io.BytesIO(b"0" * (5 * 1024 * 1024 + 1))
    response = client.post("/", data={"file": (big, "big.jpg")})
    assert response.status_code == 413


def test_missing_model_shows_clear_error(tmp_path):
    app = create_app(model_path=str(tmp_path / "missing.keras"))
    client = app.test_client()
    assert b"No trained model found" in client.get("/").data
    assert client.post("/", data={"file": (io.BytesIO(image_bytes()), "scan.jpg")}).status_code == 503
