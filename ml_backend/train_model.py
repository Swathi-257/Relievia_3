"""
Relievia Machine Learning Backend - Model Training Pipeline
===========================================================
Trains a Random Forest model to predict pressure-injury risk level and continuous risk
probability from patient vitals/demographics and simulated bed sensor parameters.

DATASET NOTICE:
This repository was inspected for real clinical pressure-injury datasets. Because no
real clinical dataset exists in this environment, this script automatically creates a
clearly documented, physiologically grounded synthetic prototype dataset
(`prototype_dataset.csv`) strictly for demonstrating the ML training, evaluation, and
serving pipeline. THIS DATASET IS NOT REAL CLINICAL DATA.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Fixed seed for complete reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "prototype_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

FEATURE_COLUMNS = [
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

CATEGORICAL_FEATURES = ["mobility", "posture"]
NUMERIC_FEATURES = [
    "age",
    "diabetes",
    "time_in_current_posture",
    "sacral_pressure",
    "heel_pressure",
    "hip_pressure",
    "shoulder_pressure",
]


def generate_synthetic_prototype_dataset(n_samples=2500, output_csv=DATASET_PATH):
    """
    Generates a physiologically coherent prototype dataset for pressure-injury risk.
    
    Principles reflected:
    - Braden Scale clinical risk factors: immobility, diabetes, advanced age.
    - Posture-dependent tissue loading:
      * Supine posture concentrates tissue load on sacrum and calcaneus (heels).
      * Lateral (left/right) postures transfer load to the greater trochanter (hip) and acromion (shoulder).
    - Duration threshold: Continuous posture time > 90-120 minutes accelerates capillary occlusion risk.
    - Risk categories:
      * LOW (< 35)
      * MODERATE (35 <= score < 60)
      * HIGH (60 <= score < 80)
      * CRITICAL (score >= 80)
    """
    print(f"[DATASET] Generating {n_samples} prototype synthetic patient/sensor records...")

    # Archetype scenarios matching clinical profiles (similar to Relievia simulation scenarios):
    # 1. Healthy / Ambulatory (predominantly LOW risk)
    # 2. Elderly / Frail (predominantly MODERATE risk)
    # 3. Diabetic / Neuropathy (MODERATE to HIGH risk)
    # 4. ICU / Post-op (HIGH risk)
    # 5. Immobile / Acute Tissue Overload (CRITICAL risk)

    n_per_archetype = n_samples // 5
    records = []

    archetypes = [
        {"name": "healthy", "age_range": (20, 50), "mob": "independent", "diab": 0, "time_scale": 40, "base_p": 25, "p_noise": 8},
        {"name": "elderly", "age_range": (65, 88), "mob": "limited",     "diab": 0, "time_scale": 75, "base_p": 45, "p_noise": 10},
        {"name": "diabetic", "age_range": (50, 78), "mob": "limited",    "diab": 1, "time_scale": 90, "base_p": 55, "p_noise": 12},
        {"name": "icu",      "age_range": (55, 85), "mob": "immobile",   "diab": 0, "time_scale": 130, "base_p": 68, "p_noise": 14},
        {"name": "critical", "age_range": (68, 92), "mob": "immobile",   "diab": 1, "time_scale": 170, "base_p": 80, "p_noise": 12},
    ]

    for arch in archetypes:
        for _ in range(n_per_archetype):
            age = int(np.random.randint(arch["age_range"][0], arch["age_range"][1] + 1))
            # Minor mobility noise
            if np.random.rand() < 0.12:
                mob = np.random.choice(["independent", "limited", "immobile"])
            else:
                mob = arch["mob"]

            diab = arch["diab"] if np.random.rand() > 0.10 else (1 - arch["diab"])
            pos = np.random.choice(["supine", "left", "right"], p=[0.50, 0.25, 0.25])
            time_in_posture = int(np.clip(np.random.exponential(scale=arch["time_scale"]) + 10, 5, 360))

            base_p = arch["base_p"]
            noise = arch["p_noise"]

            if pos == "supine":
                sac = np.clip(np.random.normal(base_p * 1.05, noise), 8, 98)
                heel = np.clip(np.random.normal(base_p * 0.95, noise), 8, 95)
                hip = np.clip(np.random.normal(base_p * 0.40, noise * 0.8), 5, 75)
                shd = np.clip(np.random.normal(base_p * 0.45, noise * 0.8), 5, 75)
            elif pos == "left":
                sac = np.clip(np.random.normal(base_p * 0.35, noise * 0.8), 5, 70)
                heel = np.clip(np.random.normal(base_p * 0.35, noise * 0.8), 5, 70)
                hip = np.clip(np.random.normal(base_p * 1.15, noise), 8, 98)
                shd = np.clip(np.random.normal(base_p * 0.90, noise), 8, 95)
            else:  # right
                sac = np.clip(np.random.normal(base_p * 0.35, noise * 0.8), 5, 70)
                heel = np.clip(np.random.normal(base_p * 0.35, noise * 0.8), 5, 70)
                hip = np.clip(np.random.normal(base_p * 1.15, noise), 8, 98)
                shd = np.clip(np.random.normal(base_p * 0.90, noise), 8, 95)

            # Posture-weighted pressure component
            if pos == "supine":
                weighted_p = 0.50 * sac + 0.30 * heel + 0.20 * shd
            else:
                weighted_p = 0.60 * hip + 0.28 * shd + 0.12 * heel

            max_p = max(sac, heel, hip, shd)
            peak_penalty = max(0.0, (max_p - 50) * 0.35) if max_p > 50 else 0.0

            mob_points = 14.0 if mob == "immobile" else (7.0 if mob == "limited" else 0.0)
            diab_points = 7.0 if diab == 1 else 0.0
            age_points = 8.0 if age >= 75 else (4.0 if age >= 60 else 0.0)

            time_penalty = min(18.0, (time_in_posture - 90) * 0.10) if time_in_posture > 90 else 0.0

            raw_score = (
                0.54 * weighted_p
                + peak_penalty
                + mob_points
                + diab_points
                + age_points
                + time_penalty
                + np.random.normal(0, 1.8)
            )

            risk_score = float(np.clip(raw_score, 0.0, 100.0))
            continuous_risk = round(risk_score, 1)

            if continuous_risk < 35.0:
                level = "LOW"
            elif continuous_risk < 60.0:
                level = "MODERATE"
            elif continuous_risk < 80.0:
                level = "HIGH"
            else:
                level = "CRITICAL"

            records.append({
                "age": age,
                "mobility": mob,
                "diabetes": diab,
                "posture": pos,
                "time_in_current_posture": time_in_posture,
                "sacral_pressure": round(sac, 1),
                "heel_pressure": round(heel, 1),
                "hip_pressure": round(hip, 1),
                "shoulder_pressure": round(shd, 1),
                "risk_probability": continuous_risk,
                "risk_level": level,
            })

    df = pd.DataFrame(records)

    df.to_csv(output_csv, index=False)
    print(f"[DATASET] Saved prototype dataset to: {output_csv}")
    print("[DATASET] Target distribution:")
    print(df["risk_level"].value_counts(normalize=True).round(3))
    return df


def load_or_create_dataset():
    """
    Checks if a real or existing dataset is available; otherwise creates the synthetic prototype.
    """
    if os.path.exists(DATASET_PATH):
        print(f"[DATASET] Loading existing dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
    else:
        df = generate_synthetic_prototype_dataset()
    return df


def train_and_evaluate():
    """
    Preprocesses features, trains RandomForestClassifier and Regressor,
    calculates metrics, and saves model artifacts.
    """
    print("=" * 70)
    print("RELIEVIA PRESSURE-INJURY RISK ML BACKEND: MODEL TRAINING")
    print("=" * 70)

    df = load_or_create_dataset()

    X = df[FEATURE_COLUMNS]
    y_class = df["risk_level"]
    y_score = df["risk_probability"]

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    # Split dataset into training and testing sets (80 / 20 stratified)
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.20, random_state=RANDOM_STATE, stratify=y_class
    )

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 1. Primary Model: RandomForestClassifier
    clf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("\n[TRAINING] Fitting RandomForestClassifier on risk_level...")
    clf_pipeline.fit(X_train, y_class_train)

    # 2. Continuous Risk Model: RandomForestRegressor (for granular 0-100 probability)
    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("[TRAINING] Fitting RandomForestRegressor on continuous risk_probability...")
    reg_pipeline.fit(X_train, y_score_train)

    # Evaluate Classifier
    y_class_pred = clf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_class_test, y_class_pred)
    precision_macro = precision_score(y_class_test, y_class_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_class_test, y_class_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_class_test, y_class_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_class_test, y_class_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_class_test, y_class_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_class_test, y_class_pred, average="weighted", zero_division=0)

    # Evaluate Regressor
    y_score_pred = reg_pipeline.predict(X_test)
    mae = mean_absolute_error(y_score_test, y_score_pred)
    r2 = r2_score(y_score_test, y_score_pred)

    print("\n" + "=" * 50)
    print("EVALUATION METRICS ON TEST SET (20% Holdout):")
    print("=" * 50)
    print(f"Accuracy:           {accuracy * 100:.2f}%")
    print(f"Precision (Macro):  {precision_macro * 100:.2f}%")
    print(f"Recall (Macro):     {recall_macro * 100:.2f}%")
    print(f"F1-Score (Macro):   {f1_macro * 100:.2f}%")
    print("-" * 50)
    print(f"Precision (Weight): {precision_weighted * 100:.2f}%")
    print(f"Recall (Weight):    {recall_weighted * 100:.2f}%")
    print(f"F1-Score (Weight):  {f1_weighted * 100:.2f}%")
    print("-" * 50)
    print(f"Continuous Score MAE: {mae:.2f} points (R2: {r2:.4f})")
    print("=" * 50)
    print("\nDetailed Classification Report:")
    print(classification_report(y_class_test, y_class_pred, zero_division=0))

    # Package Artifacts
    model_bundle = {
        "classifier": clf_pipeline,
        "regressor": reg_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "classes": list(clf_pipeline.classes_),
        "random_state": RANDOM_STATE,
        "metrics": {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "mae": float(mae),
            "r2": float(r2),
        },
        "is_synthetic": True,
        "dataset_notice": (
            "Model trained on a clearly documented prototype synthetic dataset. "
            "Created solely for demonstrating the Relievia smart bed ML architecture."
        ),
    }

    joblib.dump(model_bundle, MODEL_PATH)
    print(f"[SUCCESS] Trained model package saved to: {MODEL_PATH}")
    return model_bundle


if __name__ == "__main__":
    train_and_evaluate()
