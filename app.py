from pathlib import Path
from io import BytesIO
from base64 import b64decode
from binascii import Error as BinasciiError
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


NATIVE_CAMERA_HTML = """
<div class="native-camera-capture">
    <input id="native-camera-file" type="file" accept="image/*" capture="environment" hidden>
    <button id="native-camera-button" type="button">Capture Image</button>
    <p id="native-camera-message" role="status"></p>
</div>
"""

NATIVE_CAMERA_CSS = """
.native-camera-capture { width: 100%; }
#native-camera-button {
    width: 100%; border: 0; border-radius: 0.5rem; padding: 0.65rem 1rem;
    background: var(--st-primary-color); color: var(--st-primary-contrast-text-color, white);
    font: inherit; font-weight: 600; cursor: pointer;
}
#native-camera-button:focus-visible { outline: 2px solid var(--st-primary-color); outline-offset: 2px; }
#native-camera-message { margin: 0.45rem 0 0; color: var(--st-secondary-text-color); font-size: 0.85rem; }
"""

NATIVE_CAMERA_JS = """
export default function(component) {
    const { parentElement, setStateValue } = component;
    const input = parentElement.querySelector('#native-camera-file');
    const button = parentElement.querySelector('#native-camera-button');
    const message = parentElement.querySelector('#native-camera-message');

    button.onclick = () => {
        // capture="environment" is a browser hint. The device decides which
        // camera controls and lenses, if any, it exposes in its native flow.
        input.value = '';
        message.textContent = 'Opening your device camera…';
        input.click();
    };

    input.oncancel = () => {
        message.textContent = 'Capture cancelled. You can try again or upload from gallery.';
    };

    input.onchange = () => {
        const file = input.files && input.files[0];
        if (!file) {
            message.textContent = 'No image selected. You can try again or upload from gallery.';
            return;
        }
        if (!file.type.startsWith('image/')) {
            message.textContent = 'Please capture an image file.';
            setStateValue('error', 'The selected capture is not an image.');
            return;
        }

        const reader = new FileReader();
        reader.onerror = () => {
            message.textContent = 'The captured image could not be read. Please try again.';
            setStateValue('error', 'The captured image could not be read.');
        };
        reader.onload = () => {
            message.textContent = 'Image captured successfully.';
            setStateValue('capture', {
                data_url: reader.result,
                name: file.name || 'captured-image.jpg',
                size: file.size,
            });
        };
        reader.readAsDataURL(file);
    };
}
"""

native_camera_capture = st.components.v2.component(
    "fruit_native_camera_capture",
    html=NATIVE_CAMERA_HTML,
    css=NATIVE_CAMERA_CSS,
    js=NATIVE_CAMERA_JS,
)


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


def open_image(image_bytes):
    """Validate raw upload/capture bytes and return an independent RGB image."""
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def decode_captured_image(capture):
    """Decode the data URL returned by the native browser capture control."""
    if not isinstance(capture, dict):
        return None

    data_url = capture.get("data_url")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None

    try:
        _, encoded_image = data_url.split(",", 1)
        return b64decode(encoded_image, validate=True)
    except (ValueError, BinasciiError):
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
        "Tap **Capture Image** to use your browser's supported camera capture flow. "
        "Available camera controls and lenses are chosen by your device and browser."
    )
    camera_result = native_camera_capture(
        key=f"native_camera_capture_{version}",
        default={"capture": None, "error": None},
        on_capture_change=lambda: None,
        on_error_change=lambda: None,
    )
    if camera_result.get("error"):
        st.error(camera_result.error)

    image_bytes = decode_captured_image(camera_result.get("capture"))
    if camera_result.get("capture") and image_bytes is None:
        st.error("The captured image could not be processed. Please retake it or upload from gallery.")
    preview_caption = "Captured image"
else:
    image_file = st.file_uploader(
        "Upload from Gallery",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"gallery_image_{version}",
        help="Choose an existing image from your device.",
    )
    image_bytes = image_file.getvalue() if image_file is not None else None
    preview_caption = "Uploaded image"

if image_bytes is None:
    if source == "Capture Image":
        st.info(
            "If camera capture is unavailable or permission is denied, enable Camera permission "
            "for this site in your browser settings and try again. Use HTTPS (or localhost), or "
            "choose Upload from Gallery as a fallback."
        )
else:
    file_size_mb = len(image_bytes) / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"Image is too large. Maximum upload size is {MAX_UPLOAD_SIZE_MB} MB.")
    else:
        image = open_image(image_bytes)
        if image is None:
            st.error("The selected file is not a valid image. Please capture or choose another image.")
        else:
            image_id = hashlib.sha256(image_bytes).hexdigest()
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
