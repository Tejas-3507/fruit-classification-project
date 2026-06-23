import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fruit_classifier.keras"
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

CLASS_NAMES = [
    "apple",
    "banana",
    "mango",
    "non_fruit",
    "orange",
    "strawberry",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Add fruit_classifier.keras before running."
    )

model = tf.keras.models.load_model(MODEL_PATH)


PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fruit Classifier</title>
    <style>
        body {
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #17202a;
        }
        main {
            width: min(92vw, 520px);
            padding: 28px;
            background: #ffffff;
            border: 1px solid #d9e1e8;
            border-radius: 8px;
            box-shadow: 0 16px 40px rgba(23, 32, 42, 0.08);
        }
        h1 {
            margin: 0 0 18px;
            font-size: 28px;
        }
        form {
            display: grid;
            gap: 14px;
        }
        input[type="file"] {
            padding: 12px;
            border: 1px solid #cbd5df;
            border-radius: 6px;
            background: #fbfcfd;
        }
        button {
            min-height: 44px;
            border: 0;
            border-radius: 6px;
            background: #1f7a5c;
            color: white;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
        }
        button:hover {
            background: #176349;
        }
        .result {
            margin-top: 18px;
            padding: 14px;
            border-radius: 6px;
            background: #edf7f3;
            border: 1px solid #c7eadc;
        }
        .error {
            margin-top: 18px;
            padding: 14px;
            border-radius: 6px;
            background: #fff2f0;
            border: 1px solid #ffccc7;
        }
    </style>
</head>
<body>
    <main>
        <h1>Fruit Classifier</h1>
        <form method="post" action="/predict" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/png,image/jpeg,image/webp" required>
            <button type="submit">Predict</button>
        </form>
        {% if result %}
            <div class="result">
                <strong>Type:</strong> {{ result.type }}<br>
                {% if result.fruit_name %}
                    <strong>Fruit Name:</strong> {{ result.fruit_name }}<br>
                {% endif %}
                <strong>Confidence:</strong> {{ result.confidence }}%
            </div>
        {% endif %}
        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}
    </main>
</body>
</html>
"""


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def wants_json_response():
    if request.args.get("format") == "json":
        return True
    best_match = request.accept_mimetypes.best_match(["application/json", "text/html"])
    return best_match == "application/json"


def render_error(message, status_code=400):
    if wants_json_response():
        return jsonify({"error": message}), status_code
    return render_template_string(PAGE, error=message), status_code


def prepare_image(file_storage):
    image = Image.open(file_storage.stream).convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)


def classify_image(file_storage):
    image_batch = prepare_image(file_storage)
    prediction = model.predict(image_batch, verbose=0)[0]
    index = int(np.argmax(prediction))
    label = CLASS_NAMES[index]
    confidence = float(prediction[index] * 100)

    return {
        "label": label,
        "type": "Non-Fruit" if label == "non_fruit" else "Fruit",
        "fruit_name": None if label == "non_fruit" else label.title(),
        "confidence": round(confidence, 2),
    }


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/health")
def health():
    return jsonify({"model_loaded": True, "status": "ok"})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return render_error("Image file is required.")

    image_file = request.files["image"]

    if image_file.filename == "":
        return render_error("Please select an image.")

    if not allowed_file(image_file.filename):
        return render_error("Only JPG, JPEG, PNG, and WEBP images are allowed.")

    try:
        result = classify_image(image_file)
    except UnidentifiedImageError:
        return render_error("Uploaded file is not a valid image.")

    if wants_json_response():
        return jsonify(result)

    return render_template_string(PAGE, result=result)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return render_error("Image is too large. Maximum upload size is 8 MB.", 413)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
