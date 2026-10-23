"""Flask web app: upload a brain MRI image and see the predicted class probabilities.

Run: python -m webapp.app   (set MODEL_PATH to use a model other than models/alzmri_cnn.keras)
"""

import os
from pathlib import Path

from flask import Flask, render_template, request

from alzmri.config import ALLOWED_EXTENSIONS, DEFAULT_MODEL_PATH
from alzmri.predict import load_model, predict_bytes
from alzmri.preprocessing import InvalidImageError

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def create_app(model=None, model_path: str | None = None) -> Flask:
    """Build the app. Pass `model` directly (used in tests) or let it load from `model_path`."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    model_error = None
    if model is None:
        model_path = model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        if Path(model_path).exists():
            model = load_model(model_path)
        else:
            model_error = (f"No trained model found at '{model_path}'. "
                           "Train one with 'python -m alzmri.train' (see README).")

    def page(status: int = 200, **context):
        return render_template("index.html", model_error=model_error, **context), status

    @app.get("/")
    def index():
        return page()

    @app.post("/")
    def upload():
        if model is None:
            return page(503)
        file = request.files.get("file")
        if file is None or not file.filename:
            return page(400, error="Please choose an image file to upload.")
        if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            return page(400, error="Unsupported file type. Please upload a .jpg, .jpeg or .png image.")
        try:
            prediction = predict_bytes(model, file.read())
        except InvalidImageError as exc:
            return page(400, error=str(exc))
        return page(prediction=prediction)

    @app.errorhandler(413)
    def too_large(_):
        return page(413, error="The file is too large. The limit is 5 MB.")

    return app


if __name__ == "__main__":
    create_app().run(debug=False)
