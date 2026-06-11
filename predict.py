import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "fruit_classifier.keras"

CLASS_NAMES = [
    "apple",
    "banana",
    "mango",
    "non_fruit",
    "orange",
    "strawberry"
]

model = tf.keras.models.load_model(MODEL_PATH)

image_path = input(
    "Enter image path: "
)

img = Image.open(image_path)

img = img.convert("RGB")

img = img.resize((224,224))

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