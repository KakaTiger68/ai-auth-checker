from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)  # Cho phép mọi origin gọi API

# Load mô hình
model = load_model("authenticity_model.h5")

# Hàm xử lý ảnh
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # chỉnh theo kích thước input model bạn dùng
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Route kiểm tra server đang chạy
@app.route("/")
def index():
    return "✅ AI Auth Checker API is running!"

# Route dự đoán
@app.route("/predict", methods=["POST"])
@cross_origin()  # ⚠️ Quan trọng để Blogger gọi được
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    
    try:
        img = preprocess_image(img_bytes)
        prediction = model.predict(img)[0][0]
        label = "authentic" if prediction >= 0.5 else "fake"
        confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
        return jsonify({
            "result": label,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Khởi chạy server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
