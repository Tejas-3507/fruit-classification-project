from pathlib import Path
from io import BytesIO
import hashlib

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


def open_image(image_file):
    """Validate an uploaded/captured file and return an independent RGB image."""
    try:
        image = Image.open(BytesIO(image_file.getvalue()))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def reset_image_selection():
    """Give the file widgets new keys so a captured image can be removed/retaken."""
    st.session_state.image_input_version += 1
    st.session_state.pop("classification_result", None)
    st.session_state.pop("classification_image_id", None)


def show_result(result):
    st.subheader("Result")

    metric_columns = st.columns(2)
    metric_columns[0].metric("Type", result["type"])
    metric_columns[1].metric("Confidence", f"{result['confidence']:.2f}%")

    if result["fruit_name"]:
        st.success(f"Fruit Name: {result['fruit_name']}")
    else:
        st.warning("This image is classified as non-fruit.")


st.set_page_config(page_title="Fruit Classifier", layout="centered")

st.title("Fruit Classifier")
st.write("Capture or upload an image to classify it as fruit or non-fruit.")

if "image_input_version" not in st.session_state:
    st.session_state.image_input_version = 0

source = st.radio(
    "Choose an image source",
    ("Capture Image", "Upload from Gallery"),
    horizontal=True,
)
version = st.session_state.image_input_version

if source == "Capture Image":
    st.caption(
        "Tap **Capture Image** below to request camera access. Your browser controls "
        "whether it opens a native camera flow or an in-browser camera preview."
    )
    image_file = st.camera_input(
        "Capture Image",
        key=f"camera_image_{version}",
        help="Camera permission is requested only after you tap this control.",
    )
    preview_caption = "Captured image"
else:
    image_file = st.file_uploader(
        "Upload from Gallery",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"gallery_image_{version}",
        help="Choose an existing image from your device.",
    )
    preview_caption = "Uploaded image"

if image_file is None:
    if source == "Capture Image":
        st.info(
            "If camera access is denied or unavailable, allow Camera permission for this "
            "site in your browser settings and reload. Camera capture requires HTTPS (or "
            "localhost); otherwise, use Upload from Gallery."
        )
else:
    file_size_mb = image_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"Image is too large. Maximum upload size is {MAX_UPLOAD_SIZE_MB} MB.")
    else:
        image = open_image(image_file)
        if image is None:
            st.error("The selected file is not a valid image. Please capture or choose another image.")
        else:
            image_id = hashlib.sha256(image_file.getvalue()).hexdigest()
            st.image(image, caption=preview_caption, use_container_width=True)

            action_columns = st.columns(2)
            classify_pressed = action_columns[0].button("Classify Image", type="primary")
            action_columns[1].button("Remove / Retake", on_click=reset_image_selection)

            if classify_pressed:
                with st.spinner("Classifying image..."):
                    model = load_model()
                    st.session_state.classification_result = classify_image(image, model)
                    st.session_state.classification_image_id = image_id

            if st.session_state.get("classification_image_id") == image_id:
                show_result(st.session_state.classification_result)
