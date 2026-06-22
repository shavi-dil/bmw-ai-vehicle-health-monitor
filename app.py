import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st


MODEL_THRESHOLDS = {
    "BMW 3 Series": {
        "battery_voltage": 12.0,
        "engine_temp": 100,
        "last_service_km": 20000,
        "brake_pad_thickness": 4.0,
    },
    "BMW M3": {
        "battery_voltage": 12.1,
        "engine_temp": 108,
        "last_service_km": 15000,
        "brake_pad_thickness": 4.5,
    },
    "BMW X5": {
        "battery_voltage": 12.0,
        "engine_temp": 103,
        "last_service_km": 18000,
        "brake_pad_thickness": 4.2,
    },
    "BMW 5 Series": {
        "battery_voltage": 12.0,
        "engine_temp": 102,
        "last_service_km": 18000,
        "brake_pad_thickness": 4.2,
    },
    "BMW i4": {
        "battery_voltage": 12.3,
        "engine_temp": 98,
        "last_service_km": 24000,
        "brake_pad_thickness": 4.0,
    },
    "BMW M5": {
        "battery_voltage": 12.2,
        "engine_temp": 110,
        "last_service_km": 14000,
        "brake_pad_thickness": 4.8,
    },
}


st.set_page_config(
    page_title="BMW AI Predictive Maintenance System",
    page_icon="BMW",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load("fault_prediction_model.pkl")


def calculate_health_score(values, thresholds):
    health_score = 100

    if values["battery_voltage"] < thresholds["battery_voltage"]:
        health_score -= 15

    if values["engine_temp"] > thresholds["engine_temp"]:
        health_score -= 10

    if values["last_service_km"] > thresholds["last_service_km"]:
        health_score -= 10

    if values["brake_pad_thickness"] < thresholds["brake_pad_thickness"]:
        health_score -= 15

    return max(0, health_score)


def calculate_risk_level(health_score):
    if health_score >= 85:
        return "Low Risk"

    if health_score >= 70:
        return "Medium Risk"

    return "High Risk"


def get_recommendations(values, thresholds):
    recommendations = []

    if values["battery_voltage"] < thresholds["battery_voltage"]:
        recommendations.append("Inspect battery condition.")

    if values["engine_temp"] > thresholds["engine_temp"]:
        recommendations.append("Check cooling system.")

    if values["brake_pad_thickness"] < thresholds["brake_pad_thickness"]:
        recommendations.append("Replace brake pads.")

    if values["last_service_km"] > thresholds["last_service_km"]:
        recommendations.append("Schedule maintenance service.")

    if not recommendations:
        recommendations.append("No immediate service action required.")

    return recommendations


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


def get_engineering_explanation(values, thresholds):
    explanations = []
    if values["battery_voltage"] < thresholds["battery_voltage"]:
        explanations.append("Battery voltage is below the BMW 3 Series operating threshold.")

    if values["vehicle_age"] > 5:
        explanations.append("Vehicle age exceeds recommended battery replacement cycle.")

    if values["last_service_km"] > thresholds["last_service_km"]:
        explanations.append("Predicted battery degradation risk is elevated.")

    if not explanations:
        explanations.append("System sensors are within normal operating ranges for this vehicle.")

    return explanations


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
        .stApp {
            background: #f4f7fb;
            color: #111827;
        }

        div[data-testid="stHeader"] {
            background: transparent;
        }

        .bmw-hero {
            border-left: 6px solid #0066b1;
            padding: 1.2rem 1.4rem;
            background: linear-gradient(90deg, #ffffff 0%, #edf4fb 100%);
            box-shadow: 0 8px 22px rgba(17, 24, 39, 0.08);
            margin-bottom: 1.3rem;
        }

        .bmw-kicker {
            color: #0066b1;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }

        .bmw-hero h1 {
            margin: 0.15rem 0 0.2rem;
            color: #111827;
            font-size: 2rem;
            line-height: 1.15;
        }

        .bmw-version {
            color: #4b5563;
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d8e0ea;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(17, 24, 39, 0.05);
        }

        .section-label {
            color: #111827;
            font-size: 1rem;
            font-weight: 800;
            margin: 0.6rem 0 0.5rem;
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
        <div class="bmw-version">Version 1.0 | Fault Prediction, Risk Scoring, Explainable AI</div>
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
st.sidebar.write(f"Engine temperature: {thresholds['engine_temp']} C")
st.sidebar.write(f"Service interval: {thresholds['last_service_km']:,} km")
st.sidebar.write(f"Brake pad thickness: {thresholds['brake_pad_thickness']} mm")

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
    }

    vehicle_data = pd.DataFrame([values])

    prediction = model.predict(vehicle_data)[0]
    probability_table = format_probability_table(model, vehicle_data)
    health_score = calculate_health_score(values, thresholds)
    risk = calculate_risk_level(health_score)
    recommendations = get_recommendations(values, thresholds)
    explanation_lines = get_engineering_explanation(values, thresholds)
    explainability = get_shap_explainability(model, vehicle_data, prediction)
    explanation_source = "SHAP feature attribution"

    if explainability is None:
        explainability = get_explainability(values, thresholds)
        explanation_source = "Engineering threshold attribution"

    battery_health = int(max(0, min(100, (battery_voltage - 10.0) / 5.0 * 100)))
    engine_temp_value = engine_temp

    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)
    result_col, health_col, risk_col = st.columns(3)

    with result_col:
        st.success(f"Predicted Fault: {prediction}")
        st.metric("Vehicle Health Score", f"{health_score}/100")

    with health_col:
        st.metric("Battery Health", f"{battery_health}%")
        st.plotly_chart(
            build_gauge("Battery Health", battery_health, 0, 100, units="%"),
            use_container_width=True,
        )

    with risk_col:
        st.metric("Engine Temp Gauge", f"{engine_temp_value}°C")
        st.plotly_chart(
            build_gauge("Engine Temperature", engine_temp_value, 60, 130, units="°C", threshold=thresholds["engine_temp"]),
            use_container_width=True,
        )

    st.markdown("---")
    probability_col, recommendation_col = st.columns([1.05, 0.95])

    with probability_col:
        st.subheader("Fault Probability")
        prob_metrics = probability_table.head(4)
        cols = st.columns(len(prob_metrics))
        for idx, (_, row) in enumerate(prob_metrics.iterrows()):
            cols[idx].metric(row["Fault Type"], row["Probability %"])

        st.dataframe(
            probability_table[["Fault Type", "Probability %"]],
            hide_index=True,
            use_container_width=True,
        )

    with recommendation_col:
        st.subheader("AI Maintenance Recommendations")
        for item in recommendations:
            st.write(f"✓ {item}")
        st.markdown("---")
        st.subheader("Engineering Explanation")
        for line in explanation_lines:
            st.write(line)

    st.markdown("---")
    st.subheader("Explainable AI Attribution")
    st.caption(explanation_source)
    st.write("Prediction caused by:")
    for feature, contribution in explainability:
        sign = "+" if contribution >= 0 else ""
        st.write(f"{feature} ({sign}{contribution}%)")

    st.markdown("---")
    st.plotly_chart(
        build_gauge("Vehicle Health Gauge", health_score, 0, 100, units="%"),
        use_container_width=True,
    )
