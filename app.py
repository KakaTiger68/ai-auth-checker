from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Load your trained model
model = load_model("authenticity_model.h5")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("images")
    if len(files) < 3:
        return jsonify({"error": "Upload at least 3 images"}), 400

    predictions = []
    for file in files:
        img_bytes = file.read()
        processed = preprocess(img_bytes)
        prob = float(model.predict(processed)[0][0])  # prob of AUTHENTIC
        predictions.append(prob)

    avg_prob = sum(predictions) / len(predictions)

    if 0.485 <= avg_prob <= 0.52:
        result = "UNABLE TO VERIFY"
    elif avg_prob > 0.52:
        result = "AUTHENTIC"
    else:
        result = "FAKE"

    return jsonify({
        "average_probability": round(avg_prob * 100, 2),
        "result": result
    })

if __name__ == "__main__":
    app.run(debug=True)
