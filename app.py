import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st
import sqlite3
import hashlib
import hmac
import secrets
from datetime import date, datetime, timedelta
from PIL import Image

from cv_prototype.image_processing import (
    cv_prototype_available,
    process_uploaded_image,
)
from utils.damage_detection import run_damage_detection
from utils.driving_behaviour import evaluate_driving_behaviour
from utils.ev_battery import predict_ev_health


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

st.set_page_config(
    page_title="BMW AI Predictive Maintenance System",
    page_icon="BMW",
    layout="wide",
)

st.markdown(
    """
    <style>
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: #0b0f19 !important;
                color: #f9fafb !important;
            }

            [data-testid="stSidebar"] {
                background-color: #111827 !important;
            }

            [data-testid="stSidebar"] * {
                color: #f9fafb !important;
            }

            [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] * {
                color: #f9fafb !important;
            }

            [data-testid="stNumberInput"] input {
                background-color: #1f2937 !important;
                color: #ffffff !important;
                border: 1px solid #4b5563 !important;
                -webkit-text-fill-color: #ffffff !important;
            }

            [data-testid="stNumberInput"] button {
                background-color: #1f2937 !important;
                color: #ffffff !important;
                border: 1px solid #4b5563 !important;
                -webkit-text-fill-color: #ffffff !important;
            }

            [data-testid="stSelectbox"] * {
                background-color: #1f2937 !important;
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }

            [data-testid="stButton"] button {
                background-color: #0f4c81 !important;
                color: #ffffff !important;
                border: 1px solid #60a5fa !important;
                font-weight: 700 !important;
                -webkit-text-fill-color: #ffffff !important;
            }

            [data-testid="stButton"] button:hover {
                background-color: #1565a9 !important;
                color: #ffffff !important;
            }

            div[data-testid="stVerticalBlock"] {
                color: #f9fafb !important;
            }

            .stMetric,
            div[data-testid="stMetric"],
            div[data-testid="stMetric"] * {
                color: #f9fafb !important;
            }
        }

        /* Streamlit theme toggle dark-mode support for Cloud apps */
        .stApp[data-theme="dark"],
        html[data-theme="dark"] .stApp {
            background-color: #0b0f19 !important;
            color: #f9fafb !important;
        }

        .stApp[data-theme="dark"] [data-testid="stSidebar"],
        html[data-theme="dark"] [data-testid="stSidebar"] {
            background-color: #111827 !important;
        }

        .stApp[data-theme="dark"] [data-testid="stSidebar"] *,
        html[data-theme="dark"] [data-testid="stSidebar"] * {
            color: #f9fafb !important;
            -webkit-text-fill-color: #f9fafb !important;
        }

        .stApp[data-theme="dark"] [data-testid="stMarkdownContainer"],
        .stApp[data-theme="dark"] [data-testid="stMarkdownContainer"] *,
        html[data-theme="dark"] [data-testid="stMarkdownContainer"],
        html[data-theme="dark"] [data-testid="stMarkdownContainer"] * {
            color: #f9fafb !important;
        }

        .stApp[data-theme="dark"] [data-testid="stNumberInput"] input,
        html[data-theme="dark"] [data-testid="stNumberInput"] input {
            background-color: #1f2937 !important;
            color: #ffffff !important;
            border: 1px solid #4b5563 !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        .stApp[data-theme="dark"] [data-testid="stNumberInput"] button,
        html[data-theme="dark"] [data-testid="stNumberInput"] button {
            background-color: #1f2937 !important;
            color: #ffffff !important;
            border: 1px solid #4b5563 !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        .stApp[data-theme="dark"] [data-testid="stSelectbox"] *,
        html[data-theme="dark"] [data-testid="stSelectbox"] * {
            background-color: #1f2937 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        .stApp[data-theme="dark"] [data-testid="stButton"] button,
        html[data-theme="dark"] [data-testid="stButton"] button {
            background-color: #0f4c81 !important;
            color: #ffffff !important;
            border: 1px solid #60a5fa !important;
            font-weight: 700 !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        .stApp[data-theme="dark"] [data-testid="stButton"] button:hover,
        html[data-theme="dark"] [data-testid="stButton"] button:hover {
            background-color: #1565a9 !important;
            color: #ffffff !important;
        }

        .stApp[data-theme="dark"] .stMetric,
        .stApp[data-theme="dark"] div[data-testid="stMetric"],
        .stApp[data-theme="dark"] div[data-testid="stMetric"] *,
        html[data-theme="dark"] .stMetric,
        html[data-theme="dark"] div[data-testid="stMetric"],
        html[data-theme="dark"] div[data-testid="stMetric"] * {
            color: #f9fafb !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BMW AI Predictive Maintenance System")


@st.cache_resource
def load_model():
    return joblib.load("fault_prediction_model.pkl")


def init_session_state():
    init_local_db()
    defaults = {
        "logged_in": False,
        "user_id": None,
        "vehicle_id": None,
        "full_name": "",
        "user_profile": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_db_connection():
    conn = sqlite3.connect("bmw_health.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_local_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            registration_number TEXT UNIQUE NOT NULL,
            bmw_model TEXT NOT NULL,
            mileage INTEGER DEFAULT 0,
            vehicle_age INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS diagnostic_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            vehicle_id INTEGER NOT NULL,
            mileage INTEGER NOT NULL,
            engine_temp REAL NOT NULL,
            oil_temp REAL NOT NULL,
            battery_voltage REAL NOT NULL,
            tyre_pressure REAL NOT NULL,
            brake_pad_thickness REAL NOT NULL,
            vibration_level REAL NOT NULL,
            fuel_efficiency REAL NOT NULL,
            last_service_km INTEGER NOT NULL,
            predicted_fault TEXT NOT NULL,
            health_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(vehicle_id) REFERENCES vehicles(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            vehicle_id INTEGER NOT NULL,
            reminder_type TEXT NOT NULL,
            reminder_message TEXT NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(vehicle_id) REFERENCES vehicles(id)
        )
        """
    )

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
    return hmac.compare_digest(candidate, digest_hex)


def create_reminders_from_values(user_id, vehicle_id, values):
    reminders = []
    if values.get("last_service_km", 0) > 20000:
        reminders.append(("Service", "Vehicle service is overdue. Book a maintenance session.", str(date.today() + timedelta(days=7))))
    if values.get("battery_voltage", 99) < 12.0:
        reminders.append(("Battery", "Battery voltage is low. Run battery health diagnostics.", str(date.today() + timedelta(days=10))))
    if values.get("brake_pad_thickness", 99) < 4.0:
        reminders.append(("Brake", "Brake pad thickness below safety margin. Schedule inspection.", str(date.today() + timedelta(days=5))))
    if values.get("oil_temp", 0) > 110:
        reminders.append(("Cooling/Oil", "Oil temperature elevated. Inspect oil and cooling system.", str(date.today() + timedelta(days=5))))

    if not reminders:
        return

    conn = get_db_connection()
    cur = conn.cursor()
    for reminder_type, reminder_message, due_date in reminders:
        cur.execute(
            """
            INSERT INTO reminders (user_id, vehicle_id, reminder_type, reminder_message, due_date, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, vehicle_id, reminder_type, reminder_message, due_date, "pending"),
        )
    conn.commit()
    conn.close()


def local_api_request(method, endpoint, payload=None, params=None):
    payload = payload or {}
    params = params or {}
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        if method == "POST" and endpoint == "/register":
            email = payload.get("email", "").strip().lower()
            registration = payload.get("vehicle_registration_number", "").strip().upper()

            cur.execute("SELECT id FROM users WHERE email = ?", (email,))
            existing_email = cur.fetchone()
            cur.execute("SELECT id FROM vehicles WHERE registration_number = ?", (registration,))
            existing_registration = cur.fetchone()

            if existing_email or existing_registration:
                return None, "User already exists with this email or vehicle registration number."

            now_str = datetime.utcnow().isoformat()
            cur.execute(
                "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (payload.get("full_name", ""), email, hash_password(payload.get("password", "")), now_str),
            )
            user_id = cur.lastrowid
            cur.execute(
                """
                INSERT INTO vehicles (user_id, registration_number, bmw_model, mileage, vehicle_age, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, registration, payload.get("bmw_model", "BMW 3 Series"), 0, 0, now_str),
            )
            vehicle_id = cur.lastrowid
            conn.commit()
            return {"message": "Registration successful", "user_id": user_id, "vehicle_id": vehicle_id}, None

        if method == "POST" and endpoint == "/login":
            identifier = payload.get("identifier", "").strip()
            cur.execute(
                """
                SELECT u.id AS user_id, u.full_name, u.password_hash, v.id AS vehicle_id
                FROM users u
                LEFT JOIN vehicles v ON v.user_id = u.id
                WHERE u.email = ? OR v.registration_number = ?
                LIMIT 1
                """,
                (identifier.lower(), identifier.upper()),
            )
            row = cur.fetchone()
            if row is None or not verify_password(payload.get("password", ""), row["password_hash"]):
                return None, "Invalid credentials"
            return {
                "message": "Login successful",
                "user_id": row["user_id"],
                "full_name": row["full_name"],
                "vehicle_id": row["vehicle_id"],
            }, None

        if method == "GET" and endpoint == "/user/profile":
            user_id = int(params.get("user_id", 0))
            cur.execute(
                """
                SELECT u.id, u.full_name, u.email, u.created_at,
                       v.id AS vehicle_id, v.registration_number, v.bmw_model, v.mileage, v.vehicle_age
                FROM users u
                LEFT JOIN vehicles v ON v.user_id = u.id
                WHERE u.id = ?
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None, "User not found"
            return dict(row), None

        if method == "POST" and endpoint == "/vehicles/update":
            user_id = int(params.get("user_id", 0))
            cur.execute("SELECT id FROM vehicles WHERE user_id = ? LIMIT 1", (user_id,))
            row = cur.fetchone()
            if row is None:
                return None, "Vehicle not found"
            cur.execute(
                "UPDATE vehicles SET mileage = ?, vehicle_age = ? WHERE user_id = ?",
                (int(payload.get("mileage", 0)), int(payload.get("vehicle_age", 0)), user_id),
            )
            conn.commit()
            return {"message": "Vehicle profile updated", "vehicle_id": row["id"]}, None

        if method == "POST" and endpoint == "/diagnostics/predict":
            values = payload.copy()
            user_id = int(values.get("user_id", 0))
            vehicle_id = int(values.get("vehicle_id", 0))

            cur.execute("SELECT bmw_model FROM vehicles WHERE id = ? LIMIT 1", (vehicle_id,))
            row = cur.fetchone()
            selected_model = row["bmw_model"] if row else "BMW 3 Series"
            thresholds = MODEL_THRESHOLDS.get(selected_model, MODEL_THRESHOLDS["BMW 3 Series"])

            model = load_model()
            vehicle_data = pd.DataFrame([{
                "mileage": values["mileage"],
                "vehicle_age": values["vehicle_age"],
                "engine_temp": values["engine_temp"],
                "oil_temp": values["oil_temp"],
                "battery_voltage": values["battery_voltage"],
                "tyre_pressure": values["tyre_pressure"],
                "brake_pad_thickness": values["brake_pad_thickness"],
                "vibration_level": values["vibration_level"],
                "fuel_efficiency": values["fuel_efficiency"],
                "last_service_km": values["last_service_km"],
                "engine_rpm": values["engine_rpm"],
                "coolant_temp": values["coolant_temp"],
                "oil_pressure": values["oil_pressure"],
                "brake_sensor": values["brake_sensor"],
                "battery_soh": values["battery_soh"],
                "driving_behavior_score": values["driving_behavior_score"],
                "road_condition_score": values["road_condition_score"],
                "weather_impact_score": values["weather_impact_score"],
                "maintenance_history_score": values["maintenance_history_score"],
            }])
            prediction = model.predict(vehicle_data)[0]
            probability_table_df = format_probability_table(model, vehicle_data)
            health_score = calculate_health_score(values, thresholds)
            risk_level = calculate_risk_level(health_score)
            recommendations = get_recommendations(prediction, values, thresholds)
            explanation = " ".join(get_engineering_explanation(prediction, values, thresholds))

            days_to_failure = calculate_days_to_failure(prediction, values, thresholds)
            if days_to_failure is None:
                failure_window = "90-180 days"
            else:
                failure_window = f"{max(1, days_to_failure - 14)}-{days_to_failure} days"

            now_str = datetime.utcnow().isoformat()
            cur.execute(
                """
                INSERT INTO diagnostic_records (
                    user_id, vehicle_id, mileage, engine_temp, oil_temp, battery_voltage,
                    tyre_pressure, brake_pad_thickness, vibration_level, fuel_efficiency,
                    last_service_km, predicted_fault, health_score, risk_level, recommendation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    vehicle_id,
                    int(values["mileage"]),
                    float(values["engine_temp"]),
                    float(values["oil_temp"]),
                    float(values["battery_voltage"]),
                    float(values["tyre_pressure"]),
                    float(values["brake_pad_thickness"]),
                    float(values["vibration_level"]),
                    float(values["fuel_efficiency"]),
                    int(values["last_service_km"]),
                    prediction,
                    int(health_score),
                    risk_level,
                    "; ".join(recommendations),
                    now_str,
                ),
            )
            record_id = cur.lastrowid
            conn.commit()

            create_reminders_from_values(user_id, vehicle_id, values)

            probability_rows = []
            for _, row_item in probability_table_df.iterrows():
                probability_rows.append(
                    {
                        "fault_type": row_item["Fault Type"],
                        "probability": float(row_item["Probability"]),
                        "probability_percent": round(float(row_item["Probability"]) * 100, 2),
                    }
                )

            return {
                "predicted_fault": prediction,
                "predicted_probability": round(float(probability_rows[0]["probability_percent"]), 2),
                "health_score": health_score,
                "risk_level": risk_level,
                "estimated_failure_window": failure_window,
                "explanation": explanation,
                "recommendations": recommendations,
                "probability_table": probability_rows,
                "record_id": record_id,
            }, None

        if method == "GET" and endpoint.startswith("/diagnostics/history/"):
            user_id = int(endpoint.split("/")[-1])
            cur.execute(
                """
                SELECT id, mileage, engine_temp, oil_temp, battery_voltage, tyre_pressure,
                       brake_pad_thickness, vibration_level, fuel_efficiency, last_service_km,
                       predicted_fault, health_score, risk_level, recommendation, created_at
                FROM diagnostic_records
                WHERE user_id = ?
                ORDER BY datetime(created_at) DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows], None

        if method == "GET" and endpoint.startswith("/reminders/"):
            user_id = int(endpoint.split("/")[-1])
            cur.execute(
                """
                SELECT id, reminder_type, reminder_message, due_date, status
                FROM reminders
                WHERE user_id = ?
                ORDER BY id DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows], None

        if method == "POST" and endpoint == "/reminders/create":
            cur.execute(
                """
                INSERT INTO reminders (user_id, vehicle_id, reminder_type, reminder_message, due_date, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(payload.get("user_id", 0)),
                    int(payload.get("vehicle_id", 0)),
                    payload.get("reminder_type", "General"),
                    payload.get("reminder_message", ""),
                    payload.get("due_date", str(date.today())),
                    payload.get("status", "pending"),
                ),
            )
            reminder_id = cur.lastrowid
            conn.commit()
            return {"id": reminder_id, "message": "Reminder created"}, None

        return None, f"Unsupported local endpoint: {method} {endpoint}"
    except Exception as exc:
        return None, str(exc)
    finally:
        conn.close()


def api_request(method, endpoint, payload=None, params=None):
    # Streamlit Cloud deployment uses direct SQLite handlers in-app.
    # FastAPI remains in the repository for local full-stack demonstration.
    return local_api_request(method, endpoint, payload=payload, params=params)


def load_profile_from_api():
    if not st.session_state.get("logged_in") or not st.session_state.get("user_id"):
        return None
    profile, error = api_request("GET", "/user/profile", params={"user_id": st.session_state["user_id"]})
    if error:
        return None
    st.session_state["user_profile"] = profile
    st.session_state["vehicle_id"] = profile.get("vehicle_id")
    st.session_state["full_name"] = profile.get("full_name", "")
    return profile


def calculate_health_score(values, thresholds):
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


def calculate_risk_level(health_score):
    if health_score >= 85:
        return "Low Risk"

    if health_score >= 70:
        return "Medium Risk"

    return "High Risk"


def get_recommendations(prediction, values, thresholds):
    lookup = {
        "Battery Degradation": [
            "Inspect battery terminals and connections.",
            "Perform battery State of Health (SoH) test.",
            "Schedule battery replacement within 30 days.",
        ],
        "Oil Pressure Warning": [
            "Check engine oil level immediately.",
            "Inspect oil pump and pressure relief valve.",
            "Do not drive until oil pressure is restored.",
        ],
        "Brake Wear": [
            "Inspect brake pads and rotors.",
            "Replace brake pads if below minimum thickness.",
            "Schedule brake system inspection within 7 days.",
        ],
        "Engine Overheating": [
            "Check coolant level and condition.",
            "Inspect radiator and cooling fans.",
            "Check for coolant leaks in hoses and connections.",
        ],
        "Tyre Pressure Issue": [
            "Inspect all tyre pressures.",
            "Inflate or deflate to BMW specification.",
            "Check for punctures or slow leaks.",
        ],
        "Suspension/Vibration Issue": [
            "Inspect shock absorbers and struts.",
            "Check wheel balance and alignment.",
            "Inspect suspension bushings and joints.",
        ],
        "Service Overdue": [
            "Schedule full BMW service immediately.",
            "Replace engine oil and filter.",
            "Inspect all service-interval components.",
        ],
        "No Fault": [
            "All systems are within normal operating range.",
            "Continue with standard maintenance schedule.",
        ],
    }
    return lookup.get(prediction, ["Schedule a full vehicle inspection."])


def get_explainability(values, thresholds):
    explainability = []

    battery_gap = thresholds["battery_voltage"] - values["battery_voltage"]
    if battery_gap > 0:
        explainability.append(("Battery Voltage", -min(35, round(battery_gap * 28))))

    if values["vehicle_age"] > 5:
        explainability.append(("Vehicle Age", min(20, (values["vehicle_age"] - 5) * 4)))

    if values["mileage"] > 80000:
        explainability.append(("Mileage", min(15, round((values["mileage"] - 80000) / 8000))))

    engine_gap = values["engine_temp"] - thresholds["engine_temp"]
    if engine_gap > 0:
        explainability.append(("Engine Temperature", min(18, round(engine_gap * 2))))

    brake_gap = thresholds["brake_pad_thickness"] - values["brake_pad_thickness"]
    if brake_gap > 0:
        explainability.append(("Brake Pad Thickness", -min(25, round(brake_gap * 10))))

    service_gap = values["last_service_km"] - thresholds["last_service_km"]
    if service_gap > 0:
        explainability.append(("Distance Since Last Service", min(18, round(service_gap / 1000))))

    if not explainability:
        explainability.append(("Sensor Profile", 8))
        explainability.append(("Recent Service Interval", -6))
        explainability.append(("Operating Temperature", -4))

    return explainability[:5]


def get_shap_explainability(model, vehicle_data, prediction):
    try:
        import shap
    except ImportError:
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(vehicle_data)
        feature_names = list(vehicle_data.columns)
        class_index = list(model.classes_).index(prediction)

        if isinstance(shap_values, list):
            row_values = shap_values[class_index][0]
        elif getattr(shap_values, "ndim", 0) == 3:
            row_values = shap_values[0, :, class_index]
        else:
            row_values = shap_values[0]

        total_impact = sum(abs(value) for value in row_values) or 1
        contributions = []

        for feature_name, shap_value in zip(feature_names, row_values):
            contribution = round((shap_value / total_impact) * 100)
            contributions.append((feature_name.replace("_", " ").title(), contribution))

        return sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:5]
    except Exception:
        return None


def calculate_days_to_failure(prediction, values, thresholds):
    if prediction == "No Fault":
        return None

    if prediction == "Battery Degradation":
        voltage_deficit = max(0, thresholds["battery_voltage"] - values["battery_voltage"])
        soh_deficit = max(0, thresholds["battery_soh"] - values["battery_soh"])
        return max(7, int(90 - voltage_deficit * 40 - soh_deficit * 1.5))

    if prediction == "Oil Pressure Warning":
        pressure_deficit = max(0, thresholds["oil_pressure"] - values["oil_pressure"])
        return max(3, int(21 - pressure_deficit * 12))

    if prediction == "Brake Wear":
        pad_deficit = max(0, thresholds["brake_pad_thickness"] - values["brake_pad_thickness"])
        sensor_deficit = max(0, thresholds["brake_sensor"] - values["brake_sensor"])
        return max(7, int(60 - pad_deficit * 12 - sensor_deficit * 0.5))

    if prediction == "Engine Overheating":
        temp_excess = max(0, values["engine_temp"] - thresholds["engine_temp"])
        coolant_excess = max(0, values["coolant_temp"] - thresholds["coolant_temp"])
        return max(3, int(30 - temp_excess * 2 - coolant_excess * 1.5))

    if prediction == "Service Overdue":
        service_excess = max(0, values["last_service_km"] - thresholds["last_service_km"])
        return max(7, int(45 - service_excess // 600))

    return 60


def build_gauge(title, value, min_value, max_value, units="", threshold=None):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            number={"suffix": units},
            gauge={
                "axis": {"range": [min_value, max_value]},
                "bar": {"color": "#0066b1"},
                "steps": [
                    {"range": [min_value, (min_value + max_value) / 2], "color": "#e9f2fb"},
                    {"range": [(min_value + max_value) / 2, max_value], "color": "#c5d9ff"},
                ],
            },
        )
    )
    if threshold is not None and min_value < threshold < max_value:
        fig.add_shape(
            type="line",
            x0=0.5,
            x1=0.5,
            y0=0,
            y1=1,
            xref="paper",
            yref="paper",
            line={"color": "#ff5a5a", "width": 3, "dash": "dash"},
        )
    fig.update_layout(margin={"t": 35, "b": 0, "l": 0, "r": 0}, height=260)
    return fig


def get_engineering_explanation(prediction, values, thresholds):
    if prediction == "Battery Degradation":
        explanations = []
        if values["battery_voltage"] < thresholds["battery_voltage"]:
            explanations.append("Battery voltage trend is decreasing below BMW operating threshold.")
        if values["battery_soh"] < thresholds["battery_soh"]:
            explanations.append(f"Battery State of Health ({values['battery_soh']:.0f}%) indicates significant capacity loss.")
        if values["vehicle_age"] > 5:
            explanations.append("Vehicle age exceeds expected battery lifespan.")
        return explanations or ["Battery degradation pattern detected from sensor trends."]

    if prediction == "Oil Pressure Warning":
        explanations = [f"Oil pressure ({values['oil_pressure']:.2f} bar) is below safe operating threshold of {thresholds['oil_pressure']} bar."]
        if values["mileage"] > 100000:
            explanations.append("High mileage increases risk of oil pump wear.")
        return explanations

    if prediction == "Brake Wear":
        explanations = []
        if values["brake_pad_thickness"] < thresholds["brake_pad_thickness"]:
            explanations.append(f"Brake pad thickness ({values['brake_pad_thickness']:.1f}mm) is below the minimum safety threshold.")
        if values["brake_sensor"] < thresholds["brake_sensor"]:
            explanations.append(f"Brake wear sensor ({values['brake_sensor']:.0f}%) indicates advanced wear.")
        return explanations or ["Brake wear detected from sensor data."]

    if prediction == "Engine Overheating":
        explanations = []
        if values["engine_temp"] > thresholds["engine_temp"]:
            explanations.append(f"Engine temperature ({values['engine_temp']}\u00b0C) exceeds safe operating limit.")
        if values["coolant_temp"] > thresholds["coolant_temp"]:
            explanations.append(f"Coolant temperature ({values['coolant_temp']:.1f}\u00b0C) indicates cooling system stress.")
        return explanations or ["Engine thermal anomaly detected."]

    if prediction == "Service Overdue":
        return [
            f"Distance since last service ({values['last_service_km']:,}km) exceeds the {thresholds['last_service_km']:,}km BMW service interval.",
            "Continued operation without service increases mechanical wear risk.",
        ]

    if prediction == "Tyre Pressure Issue":
        return [f"Tyre pressure ({values['tyre_pressure']:.1f} PSI) is outside the safe BMW operating range (29\u201339 PSI)."]

    if prediction == "Suspension/Vibration Issue":
        return [
            f"Vibration level ({values['vibration_level']:.1f}) combined with high mileage ({values['mileage']:,}km) indicates suspension degradation.",
        ]

    return ["All primary sensor readings are within expected operating range."]


def format_probability_table(model, vehicle_data):
    probabilities = model.predict_proba(vehicle_data)[0]
    probability_table = pd.DataFrame(
        {
            "Fault Type": model.classes_,
            "Probability": probabilities,
        }
    )
    probability_table["Probability %"] = (
        probability_table["Probability"] * 100
    ).round(1).astype(str) + "%"

    return probability_table.sort_values("Probability", ascending=False)


def render_auth_page():
    st.markdown("## Account Access")
    st.caption("Use Login/Register for full persistence features, or continue in guest mode for local diagnostics.")

    if st.button("Continue as Guest (Local Diagnostic Mode)"):
        st.session_state["logged_in"] = True
        st.session_state["user_id"] = 0
        st.session_state["vehicle_id"] = 0
        st.session_state["full_name"] = "Guest User"
        st.session_state["user_profile"] = {
            "full_name": "Guest User",
            "bmw_model": "BMW 3 Series",
            "registration_number": "GUEST-LOCAL",
            "mileage": 50000,
            "vehicle_age": 5,
        }
        st.rerun()

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            identifier = st.text_input("Email or Vehicle Registration Number")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            payload = {"identifier": identifier, "password": password}
            data, error = api_request("POST", "/login", payload=payload)
            if error:
                st.error(f"Login failed: {error}")
            else:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = data["user_id"]
                st.session_state["vehicle_id"] = data.get("vehicle_id")
                st.session_state["full_name"] = data.get("full_name", "")
                load_profile_from_api()
                st.success("Login successful")
                st.rerun()

    with tab_register:
        with st.form("register_form"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            registration_number = st.text_input("Vehicle Registration Number")
            bmw_model = st.selectbox("BMW Model", list(MODEL_THRESHOLDS.keys()))
            submitted = st.form_submit_button("Create Account")

        if submitted:
            payload = {
                "full_name": full_name,
                "email": email,
                "password": password,
                "vehicle_registration_number": registration_number,
                "bmw_model": bmw_model,
            }
            data, error = api_request("POST", "/register", payload=payload)
            if error:
                st.error(f"Registration failed: {error}")
            else:
                st.success("Registration successful. Please login.")


def render_dashboard_page():
    profile = st.session_state.get("user_profile") or load_profile_from_api()
    if not profile:
        st.warning("Could not load profile from backend.")
        return

    st.markdown("## Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("User", profile.get("full_name", "-"))
    col2.metric("BMW Model", profile.get("bmw_model", "-"))
    col3.metric("Registration", profile.get("registration_number", "-"))

    history, _ = api_request("GET", f"/diagnostics/history/{st.session_state['user_id']}")
    reminders, _ = api_request("GET", f"/reminders/{st.session_state['user_id']}")

    if history:
        latest = history[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Health Score", f"{latest['health_score']}/100")
        c2.metric("Latest Predicted Fault", latest["predicted_fault"])
        c3.metric("Risk Level", latest["risk_level"])

        if len(history) > 1:
            prev = history[1]
            st.markdown("### Trend Analysis")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Mileage Change", latest["mileage"] - prev["mileage"])
            t2.metric("Battery Voltage Change", round(latest["battery_voltage"] - prev["battery_voltage"], 2))
            t3.metric("Brake Pad Change", round(latest["brake_pad_thickness"] - prev["brake_pad_thickness"], 2))
            t4.metric("Health Score Change", latest["health_score"] - prev["health_score"])

    st.markdown("### Reminder Alerts")
    if reminders:
        for item in reminders[:5]:
            st.warning(f"{item['reminder_type']}: {item['reminder_message']} (due {item['due_date']})")
    else:
        st.info("No reminders available.")


def render_vehicle_profile_page():
    profile = st.session_state.get("user_profile") or load_profile_from_api()
    if not profile:
        st.warning("Could not load profile from backend.")
        return

    st.markdown("## Vehicle Profile")
    with st.form("vehicle_profile_form"):
        mileage = st.number_input("Mileage", min_value=0, value=int(profile.get("mileage") or 0))
        vehicle_age = st.number_input("Vehicle Age", min_value=0, value=int(profile.get("vehicle_age") or 0))
        submitted = st.form_submit_button("Save Vehicle Profile")

    if submitted:
        payload = {"mileage": int(mileage), "vehicle_age": int(vehicle_age)}
        data, error = api_request("POST", "/vehicles/update", payload=payload, params={"user_id": st.session_state["user_id"]})
        if error:
            st.error(f"Could not update profile: {error}")
        else:
            st.success("Vehicle profile updated")
            load_profile_from_api()


def render_history_page():
    st.markdown("## Diagnostic History")
    history, error = api_request("GET", f"/diagnostics/history/{st.session_state['user_id']}")
    if error:
        st.error(error)
        return
    if not history:
        st.info("No diagnostics yet.")
        return
    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True)


def render_reminders_page():
    st.markdown("## Reminders")
    reminders, error = api_request("GET", f"/reminders/{st.session_state['user_id']}")
    if error:
        st.error(error)
        return

    if not reminders:
        st.info("No reminders available.")
    else:
        for item in reminders:
            st.write(f"- [{item['status']}] {item['reminder_type']}: {item['reminder_message']} (due {item['due_date']})")


def render_cv_prototype_page():
    st.markdown("## Computer Vision Prototype")
    st.caption("Prototype only: grayscale and edge-detection based vehicle damage inspection simulation")

    if not cv_prototype_available():
        st.warning("Computer Vision prototype is unavailable in this deployment environment.")
        return

    uploaded = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    image = Image.open(uploaded)
    rgb, gray, edges = process_uploaded_image(image)

    c1, c2, c3 = st.columns(3)
    c1.image(rgb, caption="Original", use_container_width=True)
    c2.image(gray, caption="Grayscale", use_container_width=True)
    c3.image(edges, caption="Edge Detection", use_container_width=True)


def render_ai_damage_detection_page():
    st.markdown("## AI Damage Detection")
    st.caption("YOLO module is a prototype with a general pre-trained model unless a custom damage dataset/model is trained.")

    if not cv_prototype_available():
        st.warning("Computer Vision prototype is unavailable in this deployment environment.")
        return

    st.markdown("### Upload Vehicle Photo")
    uploaded = st.file_uploader("Upload Vehicle Photo", type=["jpg", "jpeg", "png"], key="damage_upload")
    if not uploaded:
        return

    image = Image.open(uploaded)
    original, processed, yolo_output, detections, yolo_used, assessment = run_damage_detection(image)

    col1, col2 = st.columns(2)
    col1.image(original, caption="Uploaded Vehicle Photo", use_container_width=True)
    col2.image(processed, caption="Processed Diagnostic View", use_container_width=True)

    st.markdown("### YOLO Damage Detection")
    st.image(yolo_output, caption="Annotated Image", use_container_width=True)

    if yolo_used and detections:
        st.success(f"YOLO detections found: {len(detections)} object(s)")
        st.markdown("### Confidence Scores")
        for item in detections[:8]:
            st.write(f"- {item.label} ({item.confidence:.2f})")
    elif yolo_used:
        st.info("YOLO loaded, but no clear objects detected in this image.")
    else:
        st.warning("YOLO model unavailable in current environment. Running prototype risk assessment only.")

    st.markdown("### Simulated Damage Assessment")
    risk_col, issue_col = st.columns(2)
    risk_col.metric("Damage Risk Score", f"{assessment.risk_score}/100")
    issue_col.metric("Possible Issue", assessment.possible_issue)
    base_repair_cost = 220 + int(assessment.risk_score * 42)
    if detections:
        base_repair_cost += int(len(detections) * 95)
    st.metric("Estimated Repair Cost", f"${base_repair_cost} - ${int(base_repair_cost * 1.28)}")
    st.write(f"Recommendation: {assessment.recommendation}")


def render_ev_battery_ai_page():
    st.markdown("## EV Battery AI")

    col1, col2 = st.columns(2)
    with col1:
        battery_age = st.number_input("Battery Age (years)", 0, 20, 5)
        current_capacity = st.number_input("Current Battery Capacity (kWh)", 10.0, 200.0, 72.0)
        original_capacity = st.number_input("Original Battery Capacity (kWh)", 10.0, 220.0, 85.0)
        charging_freq = st.number_input("Average Charging Frequency / week", 0, 40, 4)
    with col2:
        fast_charging = st.number_input("Fast Charging Frequency / week", 0, 40, 2)
        weekly_distance = st.number_input("Average Driving Distance / week (km)", 0.0, 2500.0, 380.0)
        avg_temp = st.number_input("Average Temperature Exposure (°C)", -20.0, 60.0, 28.0)
        odometer = st.number_input("Current Odometer Reading (km)", 0.0, 500000.0, 65000.0)

    if st.button("Run EV Battery AI"):
        inputs = {
            "battery_age": battery_age,
            "current_battery_capacity": current_capacity,
            "original_battery_capacity": original_capacity,
            "charging_frequency_per_week": charging_freq,
            "fast_charging_frequency_per_week": fast_charging,
            "average_driving_distance_per_week": weekly_distance,
            "average_temperature_exposure": avg_temp,
            "current_odometer_reading": odometer,
        }
        result = predict_ev_health(inputs)

        r1, r2, r3 = st.columns(3)
        r1.metric("Battery SoH", result["battery_soh"])
        r2.metric("Degradation Risk", result["degradation_risk"])
        r3.metric("Estimated Replacement Window", result["estimated_replacement_window"])

        st.write(f"Recommendation: {result['recommendation']}")
        st.caption(result["explanation"])


def render_driving_behaviour_ai_page():
    st.markdown("## Driving Analytics")

    col1, col2 = st.columns(2)
    with col1:
        average_speed = st.number_input("Average Speed (km/h)", 0.0, 200.0, 72.0)
        harsh_braking = st.number_input("Harsh Braking Events / week", 0, 120, 10)
        harsh_accel = st.number_input("Harsh Acceleration Events / week", 0, 120, 9)
        cornering_intensity = st.slider("Cornering Intensity", 0.0, 10.0, 4.0)
    with col2:
        weekly_distance = st.number_input("Weekly Distance (km)", 0.0, 2500.0, 350.0)
        night_driving = st.slider("Night Driving Frequency (%)", 0.0, 100.0, 28.0)
        fuel_efficiency = st.number_input("Actual Fuel Economy (L/100 km)", 2.0, 25.0, 8.0)

    if st.button("Run Driving Behaviour AI"):
        inputs = {
            "average_speed": average_speed,
            "harsh_braking_events_per_week": harsh_braking,
            "harsh_acceleration_events_per_week": harsh_accel,
            "cornering_intensity": cornering_intensity,
            "weekly_distance": weekly_distance,
            "night_driving_frequency": night_driving,
            "fuel_efficiency": fuel_efficiency,
        }
        result = evaluate_driving_behaviour(inputs)

        s1, s2, s3 = st.columns(3)
        s1.metric("Driving Safety Rating", f"{result['driving_safety_score']}/100")
        s2.metric("Risk Level", result["risk_level"])
        s3.metric("Main Factors", result["main_factors"])

        i1, i2 = st.columns(2)
        i1.metric("Brake Wear Impact", result["brake_wear_impact"])
        i2.metric("Eco Driving Score", result["fuel_efficiency_impact"])

        weekly_report = (
            f"Weekly report: {weekly_distance:.0f} km, "
            f"{harsh_braking} harsh braking events, {harsh_accel} rapid accelerations."
        )
        st.info(weekly_report)

        st.write(f"Recommendation: {result['recommendation']}")


model = load_model()

st.markdown(
    """
    <style>
        :root {
            --bmw-blue: #0066b1;
            --bmw-bg: var(--background-color, #f4f7fb);
            --bmw-text: var(--text-color, #111827);
            --bmw-muted: color-mix(in srgb, var(--bmw-text) 68%, transparent);
            --bmw-card-bg: var(--secondary-background-color, #ffffff);
            --bmw-card-border: color-mix(in srgb, var(--bmw-text) 16%, transparent);
            --bmw-hero-start: #ffffff;
            --bmw-hero-end: #edf4fb;
            --bmw-shadow: rgba(17, 24, 39, 0.08);
            --bmw-card-shadow: rgba(17, 24, 39, 0.05);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bmw-bg: #0e1117;
                --bmw-text: #f3f4f6;
                --bmw-muted: #c9d1d9;
                --bmw-card-bg: #161b22;
                --bmw-card-border: #30363d;
                --bmw-hero-start: #111827;
                --bmw-hero-end: #1f2937;
                --bmw-shadow: rgba(0, 0, 0, 0.28);
                --bmw-card-shadow: rgba(0, 0, 0, 0.2);
            }
        }

        .stApp {
            background: var(--bmw-bg);
            color: var(--bmw-text);
        }

        div[data-testid="stHeader"] {
            background: transparent;
        }

        .bmw-hero {
            border-left: 6px solid var(--bmw-blue);
            padding: 1.2rem 1.4rem;
            background: linear-gradient(90deg, var(--bmw-hero-start) 0%, var(--bmw-hero-end) 100%);
            box-shadow: 0 8px 22px var(--bmw-shadow);
            margin-bottom: 1.3rem;
            border-radius: 8px;
        }

        .bmw-kicker {
            color: var(--bmw-blue);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }

        .bmw-hero h1 {
            margin: 0.15rem 0 0.2rem;
            color: var(--bmw-text);
            font-size: 2rem;
            line-height: 1.15;
        }

        .bmw-version {
            color: var(--bmw-muted);
            font-weight: 600;
        }

        p, li, label, span, div {
            color: inherit;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stText"] {
            color: var(--bmw-text);
        }

        div[data-testid="stMetric"] {
            background: var(--bmw-card-bg);
            border: 1px solid var(--bmw-card-border);
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 14px var(--bmw-card-shadow);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--bmw-muted) !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--bmw-text) !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="input"] textarea {
            color: var(--bmw-text) !important;
        }

        div[data-baseweb="input"] {
            background: var(--bmw-card-bg) !important;
            border-color: var(--bmw-card-border) !important;
        }

        div[data-baseweb="select"] > div {
            color: var(--bmw-text) !important;
            background: var(--bmw-card-bg) !important;
            border-color: var(--bmw-card-border) !important;
        }

        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] * {
            color: var(--bmw-text) !important;
        }

        .section-label {
            color: var(--bmw-text);
            font-size: 1rem;
            font-weight: 800;
            margin: 0.6rem 0 0.5rem;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 0.75rem;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 0;
        }

        div[data-testid="stDataFrame"] {
            overflow-x: auto;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 48px;
            font-weight: 700;
        }

        @media (max-width: 900px) {
            .bmw-hero {
                padding: 1rem;
            }

            .bmw-hero h1 {
                font-size: 1.5rem;
            }

            .bmw-version {
                font-size: 0.9rem;
                line-height: 1.35;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
            }

            div[data-testid="stMetric"] {
                padding: 0.85rem;
            }

            .section-label {
                margin-top: 0.25rem;
            }
        }

        @media (max-width: 480px) {
            .bmw-kicker {
                font-size: 0.72rem;
            }

            .bmw-hero h1 {
                font-size: 1.28rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_hero():
    st.markdown(
        """
        <div class="bmw-hero">
            <div class="bmw-kicker">BMW Engineering Diagnostics</div>
            <h1>BMW AI Predictive Maintenance System</h1>
            <div class="bmw-version">Version 2.0 | Predictive AI · Failure Probability · Estimated Failure Time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_diagnostic_page():
    profile = st.session_state.get("user_profile") or load_profile_from_api() or {}
    selected_model = profile.get("bmw_model") or st.selectbox("BMW Model", list(MODEL_THRESHOLDS.keys()))
    thresholds = MODEL_THRESHOLDS[selected_model]

    st.sidebar.header("Vehicle Configuration")
    st.sidebar.markdown("### Active Engineering Thresholds")
    st.sidebar.write(f"Battery voltage: {thresholds['battery_voltage']} V")
    st.sidebar.write(f"Battery SoH: {thresholds['battery_soh']}%")
    st.sidebar.write(f"Engine temperature: {thresholds['engine_temp']} °C")
    st.sidebar.write(f"Coolant temperature: {thresholds['coolant_temp']} °C")
    st.sidebar.write(f"Oil pressure: {thresholds['oil_pressure']} bar")
    st.sidebar.write(f"Service interval: {thresholds['last_service_km']:,} km")
    st.sidebar.write(f"Brake pad thickness: {thresholds['brake_pad_thickness']} mm")
    st.sidebar.write(f"Brake sensor: {thresholds['brake_sensor']}%")

    input_col, status_col = st.columns([1.25, 0.75])
    with input_col:
        st.markdown('<div class="section-label">Driver Information</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            vehicle_speed = st.slider("Vehicle Speed (km/h)", 0, 260, 82)
            engine_rpm = st.slider("Engine RPM", 500, 6500, 2200)
            fuel_level = st.slider("Fuel Level (%)", 0, 100, 58)
        with d2:
            remaining_range = st.number_input("Remaining Range (km)", 0, 1200, 430)
            drive_mode = st.selectbox("Drive Mode", ["Eco Pro", "Comfort", "Sport", "Sport+"])
            tyre_pressure = st.slider("Tyre Pressure", 20.0, 45.0, 34.0)
        with d3:
            outside_temp = st.slider("Outside Temperature (°C)", -20, 55, 24)
            service_due_km = st.number_input("Service Due in km", 0, 30000, 4500)
            odometer_km = st.number_input("Odometer (km)", 0, 450000, int(profile.get("mileage") or 50000))

        st.markdown('<div class="section-label">Vehicle Health</div>', unsafe_allow_html=True)
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            engine_temp = st.slider("Engine Temperature (°C)", 60, 130, 92)
            oil_temp = st.slider("Oil Temperature (°C)", 60, 145, 98)
        with h2:
            oil_pressure = st.slider("Oil Pressure (bar)", 1.0, 6.0, 3.4)
            coolant_temp = st.slider("Coolant Temperature (°C)", 60, 130, 90)
        with h3:
            battery_voltage = st.slider("Battery Voltage (V)", 10.0, 15.0, 12.4)
            battery_soh = st.slider("Battery State of Health (%)", 40, 100, 87)
        with h4:
            brake_pad_wear = st.slider("Brake Pad Wear (%)", 0, 100, 36)
            transmission_temp = st.slider("Transmission Temperature (°C)", 45, 150, 86)

        actual_fuel_economy = st.number_input("Actual Fuel Economy in L/100 km", 3.0, 20.0, 7.6)

        st.markdown('<div class="section-label">Driving Analytics</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            driving_safety_rating = st.slider("Driving Safety Rating", 0, 100, 81)
            harsh_braking_events = st.number_input("Harsh Braking Events", 0, 200, 8)
        with a2:
            rapid_acceleration_events = st.number_input("Rapid Acceleration Events", 0, 200, 7)
            eco_driving_score = st.slider("Eco Driving Score", 0, 100, 78)
        with a3:
            weekly_driving_distance = st.number_input("Weekly Driving Report (km)", 0.0, 3000.0, 360.0)

        mileage = int(odometer_km)
        vehicle_age = st.number_input("Vehicle Age", 0, 20, int(profile.get("vehicle_age") or 5))

        service_interval_km = int(thresholds["last_service_km"])
        last_service_km = int(max(0, min(50000, service_interval_km - int(service_due_km))))
        fuel_efficiency = float(actual_fuel_economy)
        brake_pad_thickness = float(max(1.0, round(12.0 - (brake_pad_wear * 0.1), 1)))
        brake_sensor = int(max(0, min(100, 100 - brake_pad_wear)))
        vibration_level = round(
            max(
                0.3,
                min(
                    10.0,
                    1.2
                    + (vehicle_speed / 260.0) * 2.2
                    + ((harsh_braking_events + rapid_acceleration_events) / 220.0) * 3.2
                    + (max(0.0, 34.0 - tyre_pressure) * 0.1),
                ),
            ),
            2,
        )

        behavior_base = (
            driving_safety_rating * 0.5
            + eco_driving_score * 0.3
            + max(0, 100 - (harsh_braking_events + rapid_acceleration_events) * 3) * 0.2
        )
        driving_behavior_score = int(max(0, min(100, round(behavior_base))))
        road_condition_score = int(max(35, min(100, round(100 - abs(float(tyre_pressure) - 34.0) * 6 - vibration_level * 4))))
        weather_impact_score = int(max(30, min(100, round(100 - max(0, abs(outside_temp - 22) - 8) * 4))))
        maintenance_history_score = int(
            max(20, min(100, round(100 - (last_service_km / 220) - (brake_pad_wear * 0.25))))
        )

    with status_col:
        st.markdown('<div class="section-label">System Profile</div>', unsafe_allow_html=True)
        st.metric("Selected BMW Model", selected_model)
        st.metric("Diagnostic Mode", "Predictive AI")
        st.metric("Model Status", "Loaded")
        st.metric("Drive Mode", drive_mode)
        st.metric("Fuel Level", f"{fuel_level}%")
        st.metric("Remaining Range", f"{int(remaining_range)} km")
        st.metric("Outside Temperature", f"{outside_temp} °C")
        st.metric("Service Due", f"{int(service_due_km)} km")

    if st.button("Run AI Diagnostic"):
        values = {
            "mileage": mileage,
            "vehicle_age": vehicle_age,
            "engine_temp": engine_temp,
            "oil_temp": oil_temp,
            "battery_voltage": battery_voltage,
            "tyre_pressure": tyre_pressure,
            "brake_pad_thickness": brake_pad_thickness,
            "vibration_level": vibration_level,
            "fuel_efficiency": fuel_efficiency,
            "last_service_km": last_service_km,
            "engine_rpm": engine_rpm,
            "coolant_temp": coolant_temp,
            "oil_pressure": oil_pressure,
            "brake_sensor": brake_sensor,
            "battery_soh": battery_soh,
            "driving_behavior_score": driving_behavior_score,
            "road_condition_score": road_condition_score,
            "weather_impact_score": weather_impact_score,
            "maintenance_history_score": maintenance_history_score,
        }

        payload = {
            **values,
            "user_id": st.session_state.get("user_id", 0),
            "vehicle_id": st.session_state.get("vehicle_id", 0),
        }
        backend_result, backend_error = api_request("POST", "/diagnostics/predict", payload=payload)

        if backend_result:
            prediction = backend_result["predicted_fault"]
            health_score = backend_result["health_score"]
            risk = backend_result["risk_level"]
            failure_prob = backend_result["predicted_probability"]
            estimated_failure_window = backend_result["estimated_failure_window"]
            recommendations = backend_result["recommendations"]
            explanation_lines = [backend_result["explanation"]]
            probability_table = pd.DataFrame(backend_result["probability_table"])
            probability_table["Fault Type"] = probability_table["fault_type"]
            probability_table["Probability %"] = probability_table["probability_percent"].round(1).astype(str) + "%"
            explainability = get_explainability(values, thresholds)
            explanation_source = "Backend rule-based explanation"
        else:
            vehicle_data = pd.DataFrame([values])
            prediction = model.predict(vehicle_data)[0]
            probability_table = format_probability_table(model, vehicle_data)
            health_score = calculate_health_score(values, thresholds)
            risk = calculate_risk_level(health_score)
            recommendations = get_recommendations(prediction, values, thresholds)
            explanation_lines = get_engineering_explanation(prediction, values, thresholds)
            days_to_failure = calculate_days_to_failure(prediction, values, thresholds)
            estimated_failure_window = f"{max(1, days_to_failure-7)}-{days_to_failure} days" if days_to_failure else "90-180 days"
            failure_prob = probability_table.iloc[0]["Probability"] * 100
            explainability = get_shap_explainability(model, vehicle_data, prediction)
            explanation_source = "SHAP feature attribution"
            if explainability is None:
                explainability = get_explainability(values, thresholds)
                explanation_source = "Engineering threshold attribution"
            st.warning("Backend offline. Running local prediction only.")
            if backend_error:
                st.caption(f"Backend message: {backend_error}")

        if health_score < 45 or failure_prob >= 80:
            priority_level = "Critical"
        elif risk.startswith("High") or failure_prob >= 60:
            priority_level = "High"
        elif risk.startswith("Medium") or failure_prob >= 35:
            priority_level = "Medium"
        else:
            priority_level = "Low"

        st.markdown("---")
        st.markdown("### AI Diagnostics")
        card_a, card_b, card_c, card_d = st.columns(4)
        card_a.metric("Vehicle", selected_model)
        card_b.metric("Overall Health Score", f"{health_score}/100")
        card_c.metric("Predicted Fault Category", prediction)
        card_d.metric("Failure Probability", f"{failure_prob:.0f}%")

        info_a, info_b = st.columns(2)
        info_a.metric("Estimated Remaining Useful Life", estimated_failure_window)
        info_b.metric("Priority Level", priority_level)

        action_col, reason_col = st.columns(2)
        with action_col:
            st.subheader("Recommended Actions")
            for item in recommendations:
                st.write(f"✓ {item}")
        with reason_col:
            st.subheader("Engineering Explanation")
            for line in explanation_lines:
                st.write(line)

        st.markdown("---")
        st.subheader("Fault Probability")
        prob_metrics = probability_table.head(4)
        prob_cols = st.columns(len(prob_metrics))
        for idx, (_, row) in enumerate(prob_metrics.iterrows()):
            label = row.get("Fault Type", row.get("fault_type", "Fault"))
            value = row.get("Probability %", f"{row.get('probability_percent', 0):.1f}%")
            prob_cols[idx].metric(label, value)

        display_prob = probability_table[["Fault Type", "Probability %"]] if "Fault Type" in probability_table.columns else probability_table
        st.dataframe(display_prob, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Driving Analytics")
        da1, da2, da3, da4 = st.columns(4)
        da1.metric("Driving Safety Rating", f"{driving_safety_rating}/100")
        da2.metric("Harsh Braking Events", int(harsh_braking_events))
        da3.metric("Rapid Acceleration Events", int(rapid_acceleration_events))
        da4.metric("Eco Driving Score", f"{eco_driving_score}/100")
        st.info(
            "Weekly Driving Report: "
            f"{weekly_driving_distance:.0f} km this week, "
            f"{harsh_braking_events} harsh braking events, "
            f"{rapid_acceleration_events} rapid acceleration events."
        )

        st.markdown("---")
        st.subheader("Computer Vision")
        st.write("Upload Vehicle Photo: Open AI Damage Detection from the sidebar.")
        st.write("YOLO Damage Detection: Available in AI Damage Detection page.")
        st.write("Annotated Image: Available in AI Damage Detection page.")
        st.write("Confidence Scores: Available in AI Damage Detection page.")
        if cv_prototype_available():
            est_repair_cost = 180 + int((100 - health_score) * 45)
            st.metric("Estimated Repair Cost", f"${est_repair_cost} - ${int(est_repair_cost * 1.25)}")
        else:
            st.warning("Computer Vision prototype is unavailable in this deployment environment.")

        st.markdown("---")
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(build_gauge("Vehicle Health", health_score, 0, 100, units="%"), use_container_width=True)
        g2.plotly_chart(build_gauge("Battery SoH", battery_soh, 40, 100, units="%", threshold=thresholds["battery_soh"]), use_container_width=True)
        g3.plotly_chart(build_gauge("Engine Temperature", engine_temp, 60, 130, units="°C", threshold=thresholds["engine_temp"]), use_container_width=True)

        st.markdown("---")
        st.subheader("Explainable AI Attribution")
        st.caption(explanation_source)
        for feature, contribution in explainability:
            sign = "+" if contribution >= 0 else ""
            st.write(f"{feature} ({sign}{contribution}%)")


init_session_state()
render_hero()

if not st.session_state["logged_in"]:
    render_auth_page()
else:
    st.sidebar.header("Navigation")
    st.sidebar.write(f"Logged in as: {st.session_state.get('full_name', 'User')}")
    page = st.sidebar.selectbox(
        "Go to",
        [
            "Dashboard",
            "Vehicle Profile",
            "AI Diagnostic",
            "Diagnostic History",
            "Reminders",
            "AI Damage Detection",
            "EV Battery AI",
            "Driving Analytics",
            "Computer Vision Prototype",
        ],
    )

    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["user_id"] = None
        st.session_state["vehicle_id"] = None
        st.session_state["full_name"] = ""
        st.session_state["user_profile"] = None
        st.rerun()

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Vehicle Profile":
        render_vehicle_profile_page()
    elif page == "AI Diagnostic":
        render_ai_diagnostic_page()
    elif page == "Diagnostic History":
        render_history_page()
    elif page == "Reminders":
        render_reminders_page()
    elif page == "AI Damage Detection":
        render_ai_damage_detection_page()
    elif page == "EV Battery AI":
        render_ev_battery_ai_page()
    elif page == "Driving Analytics":
        render_driving_behaviour_ai_page()
    else:
        render_cv_prototype_page()
