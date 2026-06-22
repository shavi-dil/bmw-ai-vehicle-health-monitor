import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st


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

st.sidebar.header("Vehicle Configuration")
selected_model = st.sidebar.selectbox(
    "BMW Model",
    list(MODEL_THRESHOLDS.keys()),
)
thresholds = MODEL_THRESHOLDS[selected_model]

st.sidebar.markdown("### Active Engineering Thresholds")
st.sidebar.write(f"Battery voltage: {thresholds['battery_voltage']} V")
st.sidebar.write(f"Battery SoH: {thresholds['battery_soh']}%")
st.sidebar.write(f"Engine temperature: {thresholds['engine_temp']} °C")
st.sidebar.write(f"Coolant temperature: {thresholds['coolant_temp']} °C")
st.sidebar.write(f"Oil pressure: {thresholds['oil_pressure']} bar")
st.sidebar.write(f"Service interval: {thresholds['last_service_km']:,} km")
st.sidebar.write(f"Brake pad thickness: {thresholds['brake_pad_thickness']} mm")
st.sidebar.write(f"Brake sensor: {thresholds['brake_sensor']}%")

input_col, status_col = st.columns([1.15, 0.85])

with input_col:
    st.markdown('<div class="section-label">Vehicle Sensor Inputs</div>', unsafe_allow_html=True)

    mileage = st.number_input("Mileage", 0, 300000, 50000)
    vehicle_age = st.number_input("Vehicle Age", 0, 20, 5)

    left, right = st.columns(2)
    with left:
        engine_temp = st.slider("Engine Temperature", 60, 130, 90)
        battery_voltage = st.slider("Battery Voltage", 10.0, 15.0, 12.4)
        brake_pad_thickness = st.slider("Brake Pad Thickness", 1.0, 12.0, 7.0)
        fuel_efficiency = st.slider("Fuel Efficiency", 3.0, 15.0, 7.5)

    with right:
        oil_temp = st.slider("Oil Temperature", 60, 140, 95)
        tyre_pressure = st.slider("Tyre Pressure", 20.0, 45.0, 34.0)
        vibration_level = st.slider("Vibration Level", 0.0, 10.0, 2.5)
        last_service_km = st.number_input("Distance Since Last Service", 0, 50000, 10000)

with st.expander("🔍 Advanced Sensor Inputs — Version 2.0", expanded=False):
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        engine_rpm = st.slider("Engine RPM", 500, 6000, 2500)
        coolant_temp = st.slider("Coolant Temperature (°C)", 60, 130, 88)
        oil_pressure = st.slider("Oil Pressure (bar)", 1.0, 6.0, 3.5)
    with adv2:
        brake_sensor = st.slider("Brake Wear Sensor (%)", 0, 100, 80)
        battery_soh = st.slider("Battery State of Health (%)", 40, 100, 87)
        driving_behavior_score = st.slider("Driving Behaviour Score", 0, 100, 75)
    with adv3:
        road_condition_score = st.slider("Road Condition Score", 0, 100, 70)
        weather_impact_score = st.slider("Weather Impact Score", 0, 100, 75)
        maintenance_history_score = st.slider("Maintenance History Score", 0, 100, 72)

with status_col:
    st.markdown('<div class="section-label">System Profile</div>', unsafe_allow_html=True)
    st.metric("Selected BMW Model", selected_model)
    st.metric("Diagnostic Mode", "Predictive AI")
    st.metric("Model Status", "Loaded")

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

    vehicle_data = pd.DataFrame([values])

    prediction = model.predict(vehicle_data)[0]
    probability_table = format_probability_table(model, vehicle_data)
    health_score = calculate_health_score(values, thresholds)
    risk = calculate_risk_level(health_score)
    recommendations = get_recommendations(prediction, values, thresholds)
    explanation_lines = get_engineering_explanation(prediction, values, thresholds)
    days_to_failure = calculate_days_to_failure(prediction, values, thresholds)
    failure_prob = probability_table.iloc[0]["Probability"] * 100
    explainability = get_shap_explainability(model, vehicle_data, prediction)
    explanation_source = "SHAP feature attribution"

    if explainability is None:
        explainability = get_explainability(values, thresholds)
        explanation_source = "Engineering threshold attribution"

    # ── Version 2 Diagnostic Card ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### AI Diagnostic Report")

    card_a, card_b, card_c, card_d = st.columns(4)
    card_a.metric("Vehicle", selected_model)
    card_b.metric("Health Score", f"{health_score}/100")
    card_c.metric("Predicted Issue", prediction)
    card_d.metric("Failure Probability", f"{failure_prob:.0f}%")

    if days_to_failure is not None:
        est_col, risk_col2 = st.columns(2)
        est_col.metric("Estimated Failure Time", f"{days_to_failure} Days")
        if risk == "High Risk":
            risk_col2.error(f"⚠️ Risk Level: {risk}")
        elif risk == "Medium Risk":
            risk_col2.warning(f"⚠️ Risk Level: {risk}")
        else:
            risk_col2.success(f"✅ Risk Level: {risk}")
    else:
        st.success(f"✅ Risk Level: {risk} — No failure predicted.")

    st.markdown("---")

    # ── Recommended Actions + Engineering Reason ──────────────────────────
    action_col, reason_col = st.columns(2)

    with action_col:
        st.subheader("Recommended Actions")
        for item in recommendations:
            st.write(f"✓ {item}")

    with reason_col:
        st.subheader("Engineering Reason")
        for line in explanation_lines:
            st.write(line)

    st.markdown("---")

    # ── Fault Probability breakdown ───────────────────────────────────────
    st.subheader("Fault Probability")
    prob_metrics = probability_table.head(4)
    prob_cols = st.columns(len(prob_metrics))
    for idx, (_, row) in enumerate(prob_metrics.iterrows()):
        prob_cols[idx].metric(row["Fault Type"], row["Probability %"])

    st.dataframe(
        probability_table[["Fault Type", "Probability %"]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")

    # ── Gauge Charts ──────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            build_gauge("Vehicle Health", health_score, 0, 100, units="%"),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            build_gauge("Battery SoH", battery_soh, 40, 100, units="%", threshold=thresholds["battery_soh"]),
            use_container_width=True,
        )
    with g3:
        st.plotly_chart(
            build_gauge("Engine Temperature", engine_temp, 60, 130, units="°C", threshold=thresholds["engine_temp"]),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Explainable AI Attribution ────────────────────────────────────────
    st.subheader("Explainable AI Attribution")
    st.caption(explanation_source)
    for feature, contribution in explainability:
        sign = "+" if contribution >= 0 else ""
        st.write(f"{feature} ({sign}{contribution}%)")
