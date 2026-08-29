"""
PawScan — Flask interface for the from-scratch Cat vs Dog CNN.

Routes:
    GET  /            -> UI
    POST /predict      -> accepts an image, returns JSON prediction

Expects the trained weights file `cat_dog_cnn_best.pt` (from the Kaggle
notebook) to sit next to this file. Update MODEL_PATH below if you put it
somewhere else.
"""

import io
import os
import time

import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "cat_dog_cnn_best.pt")
IMG_SIZE = 128
CLASSES = ["Cat", "Dog"]   # must match training ImageFolder class order


# ---------------------------------------------------------------------------
# Model definition — must match the architecture used in training exactly
# ---------------------------------------------------------------------------
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_load_error = None
try:
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:  # noqa: BLE001 - surface any load error to the UI
    model_load_error = str(e)

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None, model_error=model_load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model not loaded: {model_load_error}"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    start = time.time()

    tensor = infer_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob_dog = torch.sigmoid(logit).item()   # model outputs P(dog)

    prob_cat = 1 - prob_dog
    elapsed_ms = (time.time() - start) * 1000

    if prob_dog >= 0.5:
        label = "Dog"
        confidence = prob_dog
    else:
        label = "Cat"
        confidence = prob_cat

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "prob_cat": round(prob_cat * 100, 1),
        "prob_dog": round(prob_dog * 100, 1),
        "inference_ms": round(elapsed_ms, 1),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
