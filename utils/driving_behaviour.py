from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "driving_behaviour_model.pkl"
LABELS = {0: "Low", 1: "Medium", 2: "High"}


def _train_behaviour_model() -> RandomForestClassifier:
    rng = np.random.default_rng(24)
    n = 2400

    avg_speed = rng.uniform(25, 140, n)
    harsh_braking = rng.integers(0, 50, n)
    harsh_accel = rng.integers(0, 50, n)
    cornering_intensity = rng.uniform(0, 10, n)
    weekly_distance = rng.uniform(40, 1200, n)
    night_freq = rng.uniform(0, 100, n)
    fuel_efficiency = rng.uniform(3, 18, n)

    risk_score = (
        (avg_speed > 105).astype(int) * 2
        + (harsh_braking > 20).astype(int) * 2
        + (harsh_accel > 20).astype(int) * 2
        + (cornering_intensity > 6).astype(int)
        + (night_freq > 45).astype(int)
        + (fuel_efficiency < 6).astype(int)
    )

    y = np.where(risk_score >= 6, 2, np.where(risk_score >= 3, 1, 0))
    X = np.column_stack(
        [
            avg_speed,
            harsh_braking,
            harsh_accel,
            cornering_intensity,
            weekly_distance,
            night_freq,
            fuel_efficiency,
        ]
    )

    model = RandomForestClassifier(n_estimators=260, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


def load_behaviour_model() -> RandomForestClassifier:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return _train_behaviour_model()


def _safety_score(inputs: Dict[str, float]) -> int:
    score = 100
    score -= max(0, (inputs["average_speed"] - 90) * 0.5)
    score -= inputs["harsh_braking_events_per_week"] * 0.9
    score -= inputs["harsh_acceleration_events_per_week"] * 0.7
    score -= inputs["cornering_intensity"] * 2.2
    score -= max(0, inputs["night_driving_frequency"] - 30) * 0.35
    score -= max(0, 7 - inputs["fuel_efficiency"]) * 2
    return int(max(0, min(100, round(score))))


def evaluate_driving_behaviour(inputs: Dict[str, float]) -> Dict[str, str]:
    model = load_behaviour_model()

    X = np.array(
        [[
            inputs["average_speed"],
            inputs["harsh_braking_events_per_week"],
            inputs["harsh_acceleration_events_per_week"],
            inputs["cornering_intensity"],
            inputs["weekly_distance"],
            inputs["night_driving_frequency"],
            inputs["fuel_efficiency"],
        ]]
    )

    risk_idx = int(model.predict(X)[0])
    risk_level = LABELS[risk_idx]

    safety_score = _safety_score(inputs)
    brake_wear_impact = int(min(100, inputs["harsh_braking_events_per_week"] * 2.3 + inputs["cornering_intensity"] * 4))
    fuel_efficiency_impact = int(max(0, min(100, (12 - inputs["fuel_efficiency"]) * 8)))

    main_factors = []
    if inputs["harsh_braking_events_per_week"] > 16:
        main_factors.append("harsh braking")
    if inputs["harsh_acceleration_events_per_week"] > 16:
        main_factors.append("aggressive acceleration")
    if inputs["night_driving_frequency"] > 45:
        main_factors.append("high night driving frequency")
    if inputs["average_speed"] > 100:
        main_factors.append("high cruising speed")

    factors_text = ", ".join(main_factors) if main_factors else "stable driving profile"

    if risk_level == "High":
        recommendation = "Reduce sudden braking and acceleration immediately; adopt smoother driving patterns."
    elif risk_level == "Medium":
        recommendation = "Improve throttle control and keep larger braking distance in traffic."
    else:
        recommendation = "Current driving style is healthy. Maintain smooth acceleration and braking."

    return {
        "driving_safety_score": str(safety_score),
        "risk_level": risk_level,
        "brake_wear_impact": f"{brake_wear_impact}/100",
        "fuel_efficiency_impact": f"{fuel_efficiency_impact}/100",
        "main_factors": factors_text,
        "recommendation": recommendation,
    }
