from flask import Flask, request, jsonify
from flask_cors import CORS
from food_model import predict_food
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit
CORS(app, resources={r"/*": {"origins": ["https://nutrifit-1ac10.web.app", "http://localhost:8080", "http://localhost:5000", "http://127.0.0.1:5000"]}})


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "NutriFit AI Backend is running"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "message": "No image field provided in multipart form-data."
            }), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({
                "success": False,
                "message": "No image file selected."
            }), 400

        # Run model inference and food nutrition estimation
        result = predict_food(image_file)

        return jsonify({
            "success": True,
            "prediction": result
        }), 200

    except Exception as e:
        print("API Prediction Error:", e)
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Server Prediction Error: {str(e)}"
        }), 500


if __name__ == "__main__":
    import os
    import urllib.request
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "food_classifier_food101.keras")
    if not os.path.exists(model_path):
        print("Model missing! Downloading from localtunnel...")
        req = urllib.request.Request("https://nutrifit-ai-models.loca.lt/food_classifier_food101.keras", headers={'Bypass-Tunnel-Reminder': 'true'})
        with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Model downloaded successfully!")

    print("=" * 60)
    print("NutriFit AI Backend Server Starting...")
    print("Server URL: http://0.0.0.0:5000")
    print("Health Check: GET /health")
    print("Root Endpoint: GET /")
    print("Prediction API: POST /predict")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )