
from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import joblib

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sequence_classifier.joblib"
@app.route("/models/hand_landmarker.task", methods=["GET"])
def hand_landmarker_model():
    from flask import send_file

    task_path = BASE_DIR / "models" / "hand_landmarker.task"
    return send_file(task_path, mimetype="application/octet-stream")


model = None
model_error = None

try:
    model = joblib.load(MODEL_PATH)
    print("SignBridge model loaded successfully.")
except Exception as error:
    model_error = str(error)
    print("Could not load model:", model_error)


@app.route("/api/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model could not be loaded",
            "details": model_error
        }), 500

    return jsonify({
        "project": "SignBridge",
        "status": "Python AI service is working!",
        "model_loaded": True,
        "supported_signs": list(model.classes_)
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    data = request.get_json(silent=True) or {}
    frames = data.get("frames")

    if not isinstance(frames, list) or len(frames) != 30:
        return jsonify({
            "error": "Expected exactly 30 frames"
        }), 400

    try:
        import numpy as np

        features = np.asarray(frames, dtype=np.float32)

        if features.size != 1890:
            return jsonify({
                "error": "Each frame must contain 63 landmark values"
            }), 400

        features = features.reshape(1, -1)
        probabilities = model.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        label = str(model.classes_[best_index])

        spoken_text = {
            "HELLO": "Hello!",
            "BYE-BYE": "Bye bye!"
        }.get(label, label.replace("-", " "))

        return jsonify({
            "prediction": label,
            "text": spoken_text,
            "score": round(float(probabilities[best_index]) * 100, 1)
        })

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
