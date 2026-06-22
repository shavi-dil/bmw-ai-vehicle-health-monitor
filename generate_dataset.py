import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

data = []

for i in range(n):
    mileage = np.random.randint(5_000, 220_000)
    vehicle_age = np.random.randint(1, 15)

    engine_temp = np.random.normal(92, 8)
    oil_temp = np.random.normal(95, 10)
    battery_voltage = np.random.normal(12.4, 0.5)
    tyre_pressure = np.random.normal(34, 3)
    brake_pad_thickness = np.random.normal(7, 2)
    vibration_level = np.random.normal(2.5, 1)
    fuel_efficiency = np.random.normal(7.5, 1.5)
    last_service_km = np.random.randint(1_000, 30_000)

    fault_type = "No Fault"
    risk_level = "Low"

    if battery_voltage < 11.8 and vehicle_age > 5:
        fault_type = "Battery Risk"
        risk_level = "High"

    elif brake_pad_thickness < 3.5:
        fault_type = "Brake Wear"
        risk_level = "High"

    elif engine_temp > 105 or oil_temp > 115:
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
        fault_type,
        risk_level
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
    "fault_type",
    "risk_level"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/bmw_vehicle_sensor_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print(df["fault_type"].value_counts())