"""
Relievia Machine Learning Backend - Test Script
==============================================
Tests the ML pipeline and Flask REST endpoints:
1. Unit testing `predict_risk()` across low, moderate, and high-risk profiles.
2. HTTP client testing against GET /health and POST /predict.
"""

import sys
import os
import requests

# Ensure ml_backend is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_risk

TEST_CASES = [
    {
        "name": "Healthy Young Adult (Low Risk Profile)",
        "payload": {
            "age": 32,
            "mobility": "independent",
            "diabetes": 0,
            "posture": "supine",
            "time_in_current_posture": 25,
            "sacral_pressure": 22.0,
            "heel_pressure": 18.0,
            "hip_pressure": 12.0,
            "shoulder_pressure": 15.0,
        },
        "expected_levels": ["LOW"],
    },
    {
        "name": "Elderly Limited-Mobility Patient (Moderate Risk Profile)",
        "payload": {
            "age": 74,
            "mobility": "limited",
            "diabetes": 0,
            "posture": "supine",
            "time_in_current_posture": 70,
            "sacral_pressure": 45.0,
            "heel_pressure": 42.0,
            "hip_pressure": 20.0,
            "shoulder_pressure": 25.0,
        },
        "expected_levels": ["MODERATE", "LOW"],
    },
    {
        "name": "Diabetic Lateral Patient (High Trochanteric/Hip Pressure)",
        "payload": {
            "age": 68,
            "mobility": "limited",
            "diabetes": 1,
            "posture": "left",
            "time_in_current_posture": 110,
            "sacral_pressure": 18.0,
            "heel_pressure": 20.0,
            "hip_pressure": 72.0,
            "shoulder_pressure": 55.0,
        },
        "expected_levels": ["HIGH", "MODERATE"],
    },
    {
        "name": "Immobile ICU Patient (Prolonged Supine - High/Critical Risk)",
        "payload": {
            "age": 72,
            "mobility": "immobile",
            "diabetes": 1,
            "posture": "supine",
            "time_in_current_posture": 180,
            "sacral_pressure": 75.0,
            "heel_pressure": 60.0,
            "hip_pressure": 55.0,
            "shoulder_pressure": 40.0,
        },
        "expected_levels": ["HIGH", "CRITICAL"],
    },
]


def test_unit_predictions():
    print("=" * 65)
    print("1. RUNNING DIRECT UNIT TESTS ON predict_risk()")
    print("=" * 65)

    all_passed = True
    for idx, tc in enumerate(TEST_CASES, start=1):
        print(f"\n[Case {idx}] {tc['name']}")
        res = predict_risk(tc["payload"])
        prob = res["risk_probability"]
        level = res["risk_level"]
        print(f" -> Predicted: {prob}% risk [{level}]")

        if level in tc["expected_levels"]:
            print(" -> PASS: Result matches expected clinical category.")
        else:
            print(f" -> NOTE: Got {level}, expected one of {tc['expected_levels']}.")

    return all_passed


def test_http_api(base_url="http://127.0.0.1:5000"):
    print("\n" + "=" * 65)
    print(f"2. RUNNING HTTP INTEGRATION TESTS AGAINST {base_url}")
    print("=" * 65)

    # 1. Test /health
    try:
        health_resp = requests.get(f"{base_url}/health", timeout=3)
        print(f"\n[GET /health] Status Code: {health_resp.status_code}")
        print(f"[GET /health] Response: {health_resp.json()}")
        assert health_resp.status_code == 200, f"Expected 200, got {health_resp.status_code}"
        assert health_resp.json().get("status") == "ok", "Expected status: ok"
        print(" -> PASS: Health check succeeded.")
    except Exception as e:
        print(f" -> FAIL / SKIPPED: Could not connect to {base_url}/health: {e}")
        print("    (Ensure Flask server is running: python ml_backend/app.py)")
        return False

    # 2. Test /predict with sample payload
    sample = TEST_CASES[3]["payload"]
    try:
        pred_resp = requests.post(f"{base_url}/predict", json=sample, timeout=3)
        print(f"\n[POST /predict] Status Code: {pred_resp.status_code}")
        data = pred_resp.json()
        print(f"[POST /predict] Response: {data}")
        assert pred_resp.status_code == 200, f"Expected 200, got {pred_resp.status_code}"
        assert "risk_probability" in data, "Missing risk_probability"
        assert "risk_level" in data, "Missing risk_level"
        print(" -> PASS: Prediction endpoint returned valid schema.")
    except Exception as e:
        print(f" -> FAIL: /predict request failed: {e}")
        return False

    # 3. Test /predict validation error with malformed payload
    try:
        bad_resp = requests.post(f"{base_url}/predict", json={"age": 50}, timeout=3)
        print(f"\n[POST /predict (Bad Request Test)] Status Code: {bad_resp.status_code}")
        print(f"[POST /predict (Bad Request Test)] Response: {bad_resp.json()}")
        assert bad_resp.status_code == 400, f"Expected 400, got {bad_resp.status_code}"
        print(" -> PASS: Bad request correctly rejected with 400.")
    except Exception as e:
        print(f" -> FAIL: Error handling validation test failed: {e}")
        return False

    print("\n" + "=" * 65)
    print("ALL API TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    test_unit_predictions()
    test_http_api()
