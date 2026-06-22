import numpy as np
import pandas as pd

np.random.seed(42)

n = 2000

data = []

for i in range(n):
    mileage = np.random.randint(5_000, 220_000)
    vehicle_age = np.random.randint(1, 15)

    # Core sensors
    engine_temp = np.random.normal(92, 8)
    oil_temp = np.random.normal(95, 10)
    battery_voltage = np.random.normal(12.4, 0.5)
    tyre_pressure = np.random.normal(34, 3)
    brake_pad_thickness = np.random.normal(7, 2)
    vibration_level = np.random.normal(2.5, 1)
    fuel_efficiency = np.random.normal(7.5, 1.5)
    last_service_km = np.random.randint(1_000, 30_000)

    # Version 2 sensors
    engine_rpm = np.random.normal(2500, 600)
    coolant_temp = np.random.normal(88, 10)
    oil_pressure = np.random.normal(3.5, 0.8)
    brake_sensor = float(np.clip(np.random.normal(80, 18), 0, 100))
    battery_soh = float(np.clip(np.random.normal(87, 12), 40, 100))
    driving_behavior_score = float(np.clip(np.random.normal(75, 18), 0, 100))
    road_condition_score = float(np.clip(np.random.normal(70, 18), 0, 100))
    weather_impact_score = float(np.clip(np.random.normal(75, 14), 0, 100))
    maintenance_history_score = float(np.clip(np.random.normal(72, 20), 0, 100))

    fault_type = "No Fault"
    risk_level = "Low"

    if (battery_voltage < 11.8 and vehicle_age > 5) or battery_soh < 65:
        fault_type = "Battery Degradation"
        risk_level = "High"

    elif oil_pressure < 2.0:
        fault_type = "Oil Pressure Warning"
        risk_level = "High"

    elif brake_pad_thickness < 3.5 or brake_sensor < 32:
        fault_type = "Brake Wear"
        risk_level = "High"

    elif engine_temp > 105 or coolant_temp > 102 or oil_temp > 115:
        fault_type = "Engine Overheating"
        risk_level = "High"

    elif tyre_pressure < 29 or tyre_pressure > 39:
        fault_type = "Tyre Pressure Issue"
        risk_level = "Medium"

    elif vibration_level > 4.5 and mileage > 100000:
        fault_type = "Suspension/Vibration Issue"
        risk_level = "Medium"

    elif last_service_km > 20000:
        fault_type = "Service Overdue"
        risk_level = "Medium"

    data.append([
        mileage,
        vehicle_age,
        round(engine_temp, 2),
        round(oil_temp, 2),
        round(battery_voltage, 2),
        round(tyre_pressure, 2),
        round(brake_pad_thickness, 2),
        round(vibration_level, 2),
        round(fuel_efficiency, 2),
        last_service_km,
        round(engine_rpm, 0),
        round(coolant_temp, 2),
        round(oil_pressure, 3),
        round(brake_sensor, 1),
        round(battery_soh, 1),
        round(driving_behavior_score, 1),
        round(road_condition_score, 1),
        round(weather_impact_score, 1),
        round(maintenance_history_score, 1),
        fault_type,
        risk_level,
    ])

columns = [
    "mileage",
    "vehicle_age",
    "engine_temp",
    "oil_temp",
    "battery_voltage",
    "tyre_pressure",
    "brake_pad_thickness",
    "vibration_level",
    "fuel_efficiency",
    "last_service_km",
    "engine_rpm",
    "coolant_temp",
    "oil_pressure",
    "brake_sensor",
    "battery_soh",
    "driving_behavior_score",
    "road_condition_score",
    "weather_impact_score",
    "maintenance_history_score",
    "fault_type",
    "risk_level",
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("bmw_vehicle_sensor_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print(df["fault_type"].value_counts())