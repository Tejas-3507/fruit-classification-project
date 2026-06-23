from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fruit_classifier.keras"
IMG_SIZE = (224, 224)
MAX_UPLOAD_SIZE_MB = 8

CLASS_NAMES = [
    "apple",
    "banana",
    "mango",
    "non_fruit",
    "orange",
    "strawberry",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Add fruit_classifier.keras before running.")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)


def classify_image(image, model):
    image_batch = prepare_image(image)
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


st.set_page_config(page_title="Fruit Classifier", layout="centered")

st.title("Fruit Classifier")
st.write("Upload an image to classify it as fruit or non-fruit.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"Image is too large. Maximum upload size is {MAX_UPLOAD_SIZE_MB} MB.")
        st.stop()

    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("Uploaded file is not a valid image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying image..."):
        model = load_model()
        result = classify_image(image, model)

    st.subheader("Result")

    metric_columns = st.columns(2)
    metric_columns[0].metric("Type", result["type"])
    metric_columns[1].metric("Confidence", f"{result['confidence']:.2f}%")

    if result["fruit_name"]:
        st.success(f"Fruit Name: {result['fruit_name']}")
    else:
        st.warning("This image is classified as non-fruit.")
