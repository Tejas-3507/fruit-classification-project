import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fruit_classifier.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "apple",
    "banana",
    "mango",
    "non_fruit",
    "orange",
    "strawberry"
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

image_path = input(
    "Enter image path: "
)

image_path = Path(image_path)

if not image_path.exists():
    raise FileNotFoundError(f"Image file not found at {image_path}")

img = Image.open(image_path)

img = img.convert("RGB")

img = img.resize(IMG_SIZE)

img_array = np.array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

prediction = model.predict(img_array)

index = np.argmax(prediction)

confidence = np.max(prediction)*100

label = CLASS_NAMES[index]

print("\nResult")
print("-------------")

if label == "non_fruit":
    print("Type : Non-Fruit")
else:
    print("Type : Fruit")
    print("Fruit Name :", label.title())

print(
    f"Confidence : {confidence:.2f}%"
)
