"""
Relievia Machine Learning Backend - Flask API
=============================================
Provides REST API endpoints for pressure-injury risk prediction.
Designed for direct integration with the Relievia HTML/JavaScript digital twin.

Endpoints:
- GET  /health   -> Liveness / readiness probe {"status": "ok"}
- POST /predict  -> Predicts risk probability and risk level from patient and sensor parameters
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS

# Ensure current directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_risk

app = Flask(__name__)

# Enable CORS for all routes so the browser running relievia.html can fetch() directly
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/health", methods=["GET"])
def health_check():
    """Health-check endpoint for server liveness verification."""
    return jsonify({
        "status": "ok",
        "service": "Relievia ML Risk Prediction API",
        "version": "1.0.0",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts patient demographics, posture, duration, and anatomical pressure readings.
    Returns predicted pressure-injury risk probability (0-100) and categorical risk level.

    Expected JSON payload:
    {
        "age": 72,
        "mobility": "immobile",
        "diabetes": 1,
        "posture": "supine",
        "time_in_current_posture": 180,
        "sacral_pressure": 75,
        "heel_pressure": 60,
        "hip_pressure": 55,
        "shoulder_pressure": 40
    }
    """
    if not request.is_json:
        return jsonify({
            "error": "Bad Request",
            "message": "Content-Type must be application/json."
        }), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "error": "Bad Request",
            "message": "Invalid JSON body provided."
        }), 400

    try:
        prediction = predict_risk(payload)
        return jsonify(prediction), 200
    except ValueError as val_err:
        return jsonify({
            "error": "Validation Error",
            "message": str(val_err)
        }), 400
    except Exception as exc:
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Prediction failed: {str(exc)}"
        }), 500


if __name__ == "__main__":
    host = os.environ.get("RELIEVIA_HOST", "0.0.0.0")
    port = int(os.environ.get("RELIEVIA_PORT", 5000))
    print(f"Starting Relievia ML API server on http://{host}:{port}...")
    app.run(host=host, port=port, debug=False)
