# Fruit Classifier

A TensorFlow and Streamlit web app that classifies uploaded images as fruit or non-fruit. For fruit images, it predicts one of these classes:

- Apple
- Banana
- Mango
- Orange
- Strawberry

The model also supports a `non_fruit` class.

## Project Structure

```text
FruitClassifier/
|-- app.py                    # Streamlit web app
|-- fruit_classifier.keras    # Trained Keras model used by the app
|-- predict.py                # Command-line prediction script
|-- train.py                  # Training script
|-- requirements.txt          # Python dependencies
|-- Procfile                  # Generic server deployment command
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

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal. Streamlit usually starts at:

```text
http://localhost:8501
```

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

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from this GitHub repository.
4. Set the main file path to:

```text
app.py
```

Streamlit Cloud will install dependencies from `requirements.txt`.

### Generic Server

This repo includes a `Procfile`:

```text
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

If your platform does not use `Procfile`, use this start command:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Important deployment notes:

- Keep `fruit_classifier.keras` in the repository or upload it with your deployment.
- `dataset/`, `test/`, and virtual environment folders are ignored because they are not needed in production.
- If your model file becomes larger than GitHub's file size limit, use Git LFS or external model storage.
