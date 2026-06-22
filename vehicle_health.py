import joblib
import pandas as pd

model = joblib.load("models/fault_prediction_model.pkl")

sample = pd.DataFrame([{
    "mileage": 120000,
    "vehicle_age": 8,
    "engine_temp": 100,
    "oil_temp": 105,
    "battery_voltage": 11.7,
    "tyre_pressure": 33,
    "brake_pad_thickness": 6,
    "vibration_level": 2.8,
    "fuel_efficiency": 7.2,
    "last_service_km": 25000
}])

prediction = model.predict(sample)[0]

health_score = 100

if sample["battery_voltage"][0] < 12:
    health_score -= 15

if sample["engine_temp"][0] > 100:
    health_score -= 10

if sample["last_service_km"][0] > 20000:
    health_score -= 10

print("BMW Vehicle Health Report")
print("-" * 40)

print(f"Health Score: {health_score}/100")
print(f"Predicted Fault: {prediction}")