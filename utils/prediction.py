from functools import lru_cache
from typing import Dict, List, Tuple

import joblib
import pandas as pd

MODEL_THRESHOLDS = {
    "BMW 3 Series": {
        "battery_voltage": 12.0,
        "engine_temp": 100,
        "coolant_temp": 95,
        "oil_pressure": 2.0,
        "last_service_km": 20000,
        "brake_pad_thickness": 4.0,
        "brake_sensor": 35,
        "battery_soh": 70,
    },
    "BMW M3": {
        "battery_voltage": 12.1,
        "engine_temp": 108,
        "coolant_temp": 100,
        "oil_pressure": 2.2,
        "last_service_km": 15000,
        "brake_pad_thickness": 4.5,
        "brake_sensor": 35,
        "battery_soh": 72,
    },
    "BMW X5": {
        "battery_voltage": 12.0,
        "engine_temp": 103,
        "coolant_temp": 97,
        "oil_pressure": 2.0,
        "last_service_km": 18000,
        "brake_pad_thickness": 4.2,
        "brake_sensor": 35,
        "battery_soh": 70,
    },
    "BMW 5 Series": {
        "battery_voltage": 12.0,
        "engine_temp": 102,
        "coolant_temp": 96,
        "oil_pressure": 2.0,
        "last_service_km": 18000,
        "brake_pad_thickness": 4.2,
        "brake_sensor": 35,
        "battery_soh": 70,
    },
    "BMW i4": {
        "battery_voltage": 12.3,
        "engine_temp": 98,
        "coolant_temp": 92,
        "oil_pressure": 1.8,
        "last_service_km": 24000,
        "brake_pad_thickness": 4.0,
        "brake_sensor": 35,
        "battery_soh": 75,
    },
    "BMW M5": {
        "battery_voltage": 12.2,
        "engine_temp": 110,
        "coolant_temp": 102,
        "oil_pressure": 2.3,
        "last_service_km": 14000,
        "brake_pad_thickness": 4.8,
        "brake_sensor": 38,
        "battery_soh": 72,
    },
}


@lru_cache(maxsize=1)
def load_model(model_path: str = "fault_prediction_model.pkl"):
    return joblib.load(model_path)


def calculate_health_score(values: Dict, thresholds: Dict) -> int:
    health_score = 100

    if values["battery_voltage"] < thresholds["battery_voltage"]:
        health_score -= 10
    if values["battery_soh"] < thresholds["battery_soh"]:
        health_score -= 12
    if values["engine_temp"] > thresholds["engine_temp"]:
        health_score -= 8
    if values["coolant_temp"] > thresholds["coolant_temp"]:
        health_score -= 8
    if values["oil_pressure"] < thresholds["oil_pressure"]:
        health_score -= 12
    if values["last_service_km"] > thresholds["last_service_km"]:
        health_score -= 8
    if values["brake_pad_thickness"] < thresholds["brake_pad_thickness"]:
        health_score -= 10
    if values["brake_sensor"] < thresholds["brake_sensor"]:
        health_score -= 8
    if values["driving_behavior_score"] < 50:
        health_score -= 5
    if values["maintenance_history_score"] < 50:
        health_score -= 5

    return max(0, health_score)


def calculate_risk_level(health_score: int) -> str:
    if health_score >= 85:
        return "Low"
    if health_score >= 70:
        return "Medium"
    return "High"


def estimate_failure_window(risk_level: str, values: Dict, thresholds: Dict) -> str:
    severity = 0
    severity += max(0.0, thresholds["battery_voltage"] - values["battery_voltage"]) * 10
    severity += max(0.0, values["engine_temp"] - thresholds["engine_temp"]) * 0.7
    severity += max(0.0, thresholds["oil_pressure"] - values["oil_pressure"]) * 8

    if risk_level == "High":
        upper = max(30, int(45 - severity))
        lower = max(7, upper - 16)
        return f"{lower}-{upper} days"
    if risk_level == "Medium":
        upper = max(75, int(95 - severity))
        lower = max(30, upper - 30)
        return f"{lower}-{upper} days"
    return "90-180 days"


def predict_fault(values: Dict, model_name: str) -> Tuple[str, float, List[dict]]:
    model = load_model()
    vehicle_data = pd.DataFrame([values])
    prediction = model.predict(vehicle_data)[0]
    probabilities = model.predict_proba(vehicle_data)[0]

    probability_table = pd.DataFrame(
        {
            "fault_type": model.classes_,
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False)

    probability_table["probability_percent"] = (probability_table["probability"] * 100).round(2)

    top_probability = float(probability_table.iloc[0]["probability_percent"])
    return prediction, top_probability, probability_table.to_dict(orient="records")
