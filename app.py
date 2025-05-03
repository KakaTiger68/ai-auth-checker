from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

# Load AI model
model = load_model("authenticity_model.h5")

# Preprocess image (adjust to your model input)
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Change if your model requires different size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def index():
    return "✅ AI Auth Checker API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image file not found"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    try:
        processed = preprocess_image(img_bytes)
        prediction = model.predict(processed)[0][0]
        label = "authentic" if prediction >= 0.5 else "fake"
        confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
        return jsonify({"result": label, "confidence": round(confidence, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
