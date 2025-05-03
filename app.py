from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# ✅ CHỈ ALLOW BLOGSPOT DOMAIN HOẶC CHO TẤT CẢ (tùy lựa chọn)
CORS(app, resources={r"/predict": {"origins": "*"}})
# Nếu bạn muốn chỉ cho phép blogspot: 
# CORS(app, resources={r"/predict": {"origins": "https://checkdohieu.blogspot.com"}})

# ✅ Load mô hình
model = load_model("authenticity_model.h5")

# ✅ API kiểm tra hoạt động
@app.route("/", methods=["GET"])
def home():
    return "✅ AI Auth Checker API is running!"

# ✅ API nhận ảnh và dự đoán
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        # Chuyển ảnh thành mảng số để đưa vào model
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert("RGB")
        img = img.resize((224, 224))  # ⚠️ Resize đúng với model bạn đã huấn luyện
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán
        prediction = model.predict(img_array)[0][0]
        result = "authentic" if prediction >= 0.5 else "fake"

        return jsonify({
            "result": result,
            "confidence": float(prediction if result == "authentic" else 1 - prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Chạy app nếu dùng local test
if __name__ == "__main__":
    app.run(debug=True)
