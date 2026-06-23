from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "ev_battery_model.pkl"


def _train_ev_model() -> RandomForestRegressor:
    rng = np.random.default_rng(42)
    n = 1600

    battery_age = rng.integers(0, 13, n)
    current_capacity = rng.uniform(35, 95, n)
    original_capacity = rng.uniform(60, 110, n)
    charging_freq = rng.integers(1, 12, n)
    fast_charging = rng.integers(0, 10, n)
    weekly_distance = rng.uniform(50, 850, n)
    avg_temp = rng.uniform(-5, 45, n)
    odometer = rng.uniform(2000, 280000, n)

    soh_base = (current_capacity / original_capacity) * 100
    degradation_penalty = (
        battery_age * 1.9
        + fast_charging * 1.8
        + np.maximum(0, avg_temp - 32) * 0.7
        + np.maximum(0, odometer - 120000) / 6000
    )
    soh_target = np.clip(soh_base - degradation_penalty, 35, 100)

    X = np.column_stack(
        [
            battery_age,
            current_capacity,
            original_capacity,
            charging_freq,
            fast_charging,
            weekly_distance,
            avg_temp,
            odometer,
        ]
    )
    y = soh_target

    model = RandomForestRegressor(n_estimators=240, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


def load_ev_model() -> RandomForestRegressor:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return _train_ev_model()


def predict_ev_health(inputs: Dict[str, float]) -> Dict[str, str]:
    model = load_ev_model()

    soh_formula = (inputs["current_battery_capacity"] / max(1e-6, inputs["original_battery_capacity"])) * 100
    X = np.array(
        [[
            inputs["battery_age"],
            inputs["current_battery_capacity"],
            inputs["original_battery_capacity"],
            inputs["charging_frequency_per_week"],
            inputs["fast_charging_frequency_per_week"],
            inputs["average_driving_distance_per_week"],
            inputs["average_temperature_exposure"],
            inputs["current_odometer_reading"],
        ]]
    )

    soh_pred = float(model.predict(X)[0])
    battery_soh = max(25.0, min(100.0, (soh_pred + soh_formula) / 2))

    if battery_soh >= 85:
        risk = "Low"
        replacement_window = "30-48 months"
        recommendation = "Battery profile is healthy. Maintain balanced charging behavior."
    elif battery_soh >= 70:
        risk = "Medium"
        replacement_window = "18-24 months"
        recommendation = "Reduce frequent fast charging to extend battery life."
    else:
        risk = "High"
        replacement_window = "6-12 months"
        recommendation = "Schedule professional battery diagnostics and replacement planning."

    explanation = (
        "Prediction combines capacity loss, charging patterns, thermal exposure, and odometer stress. "
        "Higher fast charging and high temperature exposure accelerate degradation."
    )

    return {
        "battery_soh": f"{battery_soh:.1f}%",
        "degradation_risk": risk,
        "estimated_replacement_window": replacement_window,
        "recommendation": recommendation,
        "explanation": explanation,
    }
