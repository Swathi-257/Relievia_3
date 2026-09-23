"""
Relievia Machine Learning Backend - Prediction Module
=====================================================
Provides `predict_risk(features)` to predict pressure-injury risk level and
risk probability from patient vitals and sensor readings.
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Cached model bundle
_MODEL_BUNDLE = None


def load_model():
    """Loads and caches the trained model bundle from disk."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            from .train_model import train_and_evaluate
            print("[INFO] Model not found. Training model now...")
            _MODEL_BUNDLE = train_and_evaluate()
        else:
            _MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return _MODEL_BUNDLE


def normalize_features(raw_features):
    """
    Standardizes and validates input features for model ingestion.
    Supports flexible types (strings, booleans, synonyms like 'bedbound' -> 'immobile').
    """
    if not isinstance(raw_features, dict):
        raise ValueError("Features must be a dictionary.")

    required_keys = [
        "age",
        "mobility",
        "diabetes",
        "posture",
        "time_in_current_posture",
        "sacral_pressure",
        "heel_pressure",
        "hip_pressure",
        "shoulder_pressure",
    ]

    missing = [k for k in required_keys if k not in raw_features]
    if missing:
        raise ValueError(f"Missing required feature keys: {missing}")

    # 1. Mobility normalization
    raw_mob = str(raw_features["mobility"]).strip().lower()
    if raw_mob in ["immobile", "bedbound", "none", "very limited"]:
        mob = "immobile"
    elif raw_mob in ["limited", "slightly limited", "moderate"]:
        mob = "limited"
    elif raw_mob in ["independent", "full", "active", "normal"]:
        mob = "independent"
    else:
        mob = "limited"

    # 2. Diabetes normalization
    raw_diab = raw_features["diabetes"]
    if isinstance(raw_diab, bool):
        diab = 1 if raw_diab else 0
    elif isinstance(raw_diab, (int, float)):
        diab = 1 if int(raw_diab) > 0 else 0
    elif str(raw_diab).strip().lower() in ["true", "yes", "y", "1"]:
        diab = 1
    else:
        diab = 0

    # 3. Posture normalization
    raw_pos = str(raw_features["posture"]).strip().lower()
    if "left" in raw_pos:
        pos = "left"
    elif "right" in raw_pos:
        pos = "right"
    elif "supine" in raw_pos or "flat" in raw_pos or "back" in raw_pos:
        pos = "supine"
    else:
        pos = "supine"

    # 4. Numeric features
    try:
        age = float(raw_features["age"])
        dur = float(raw_features["time_in_current_posture"])
        sac = float(raw_features["sacral_pressure"])
        heel = float(raw_features["heel_pressure"])
        hip = float(raw_features["hip_pressure"])
        shd = float(raw_features["shoulder_pressure"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Numeric features could not be converted to float: {e}")

    normalized = {
        "age": max(0.0, age),
        "mobility": mob,
        "diabetes": diab,
        "posture": pos,
        "time_in_current_posture": max(0.0, dur),
        "sacral_pressure": max(0.0, min(100.0, sac)),
        "heel_pressure": max(0.0, min(100.0, heel)),
        "hip_pressure": max(0.0, min(100.0, hip)),
        "shoulder_pressure": max(0.0, min(100.0, shd)),
    }

    return normalized


def determine_risk_level(probability):
    """Calibrated risk level according to Relievia clinical thresholds."""
    if probability < 35.0:
        return "LOW"
    elif probability < 60.0:
        return "MODERATE"
    elif probability < 80.0:
        return "HIGH"
    else:
        return "CRITICAL"


def predict_risk(features):
    """
    Predicts pressure-injury risk probability and categorical risk level.

    Parameters:
    -----------
    features: dict
        Dictionary containing patient and sensor attributes:
        - age (number)
        - mobility (str: 'immobile', 'limited', 'independent')
        - diabetes (int/bool/str: 1/0, True/False)
        - posture (str: 'supine', 'left', 'right')
        - time_in_current_posture (number of minutes)
        - sacral_pressure (0-100)
        - heel_pressure (0-100)
        - hip_pressure (0-100)
        - shoulder_pressure (0-100)

    Returns:
    --------
    dict:
        {
            "risk_probability": 78.4,
            "risk_level": "HIGH"
        }
    """
    bundle = load_model()
    classifier = bundle.get("classifier")
    regressor = bundle.get("regressor")

    clean_dict = normalize_features(features)
    input_df = pd.DataFrame([clean_dict])

    # Continuous risk estimation (0 - 100)
    if regressor is not None:
        raw_prob = float(regressor.predict(input_df)[0])
    else:
        # Fallback: estimate from classifier probabilities
        probs = classifier.predict_proba(input_df)[0]
        class_centers = {"LOW": 18.0, "MODERATE": 47.5, "HIGH": 70.0, "CRITICAL": 90.0}
        raw_prob = sum(probs[i] * class_centers.get(c, 50.0) for i, c in enumerate(classifier.classes_))

    risk_probability = round(max(0.0, min(100.0, raw_prob)), 1)

    # Risk level classification
    if classifier is not None:
        predicted_level = str(classifier.predict(input_df)[0])
        # Verify alignment with calibrated threshold to avoid border discrepancies
        calibrated_level = determine_risk_level(risk_probability)
        # Use classifier if consistent within bounds, otherwise adhere to calibrated category
        risk_level = calibrated_level if abs(risk_probability - 50) > 25 else predicted_level
    else:
        risk_level = determine_risk_level(risk_probability)

    return {
        "risk_probability": risk_probability,
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    sample_input = {
        "age": 72,
        "mobility": "immobile",
        "diabetes": 1,
        "posture": "supine",
        "time_in_current_posture": 180,
        "sacral_pressure": 75,
        "heel_pressure": 60,
        "hip_pressure": 55,
        "shoulder_pressure": 40,
    }

    print("Running sample prediction with features:")
    for k, v in sample_input.items():
        print(f"  {k}: {v}")

    result = predict_risk(sample_input)
    print("\nResult:")
    print(result)
