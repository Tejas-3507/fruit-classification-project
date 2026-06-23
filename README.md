# Fruit Classifier

A TensorFlow and Flask web app that classifies uploaded images as fruit or non-fruit. For fruit images, it predicts one of these classes:

- Apple
- Banana
- Mango
- Orange
- Strawberry

The model also supports a `non_fruit` class.

## Project Structure

```text
FruitClassifier/
|-- app.py                    # Flask web app and prediction API
|-- fruit_classifier.keras    # Trained Keras model used by the server
|-- predict.py                # Command-line prediction script
|-- train.py                  # Training script
|-- requirements.txt          # Python dependencies
|-- Procfile                  # Deployment start command
|-- runtime.txt               # Python runtime hint for hosting platforms
`-- README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Flask server:

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

Health check:

```text
http://localhost:5000/health
```

## Prediction API

Send an image to the `/predict` endpoint:

```bash
curl -X POST -H "Accept: application/json" -F "image=@test/test.jpg" http://localhost:5000/predict
```

Example response:

```json
{
  "confidence": 98.42,
  "fruit_name": "Apple",
  "label": "apple",
  "type": "Fruit"
}
```

For non-fruit images, `fruit_name` will be `null`.

## Train Model

Place training images in this folder structure:

```text
dataset/
|-- apple/
|-- banana/
|-- mango/
|-- non_fruit/
|-- orange/
`-- strawberry/
```

Recommended minimum images:

- 300-500 images per fruit class
- 1000+ images for `non_fruit`

Train:

```bash
python train.py
```

This creates or updates:

```text
fruit_classifier.keras
```

## Command-Line Prediction

```bash
python predict.py
```

Then enter an image path when prompted.

## Deployment

This repo includes a `Procfile`:

```text
web: gunicorn app:app
```

Most Python hosting platforms can use:

```bash
pip install -r requirements.txt
gunicorn app:app
```

Important deployment notes:

- Keep `fruit_classifier.keras` in the repository or upload it with your deployment.
- The app reads the port from the `PORT` environment variable when provided.
- `dataset/`, `test/`, and virtual environment folders are ignored because they are not needed in production.
- If your model file becomes larger than GitHub's file size limit, use Git LFS or external model storage.
