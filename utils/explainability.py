def build_explanation(prediction: str, values: dict, thresholds: dict) -> str:
    if prediction == "Battery Degradation":
        reasons = []
        if values["battery_voltage"] < thresholds["battery_voltage"]:
            reasons.append("Battery voltage is below the safe operating threshold")
        if values["battery_soh"] < thresholds["battery_soh"]:
            reasons.append("battery state-of-health indicates capacity degradation")
        if values["vehicle_age"] > 5:
            reasons.append("vehicle age increases battery replacement probability")
        return ". ".join(reasons) + "." if reasons else "Battery degradation trend is detected from sensor patterns."

    if prediction == "Engine Overheating":
        return "Engine and coolant temperatures are above expected operating limits, indicating thermal stress in the cooling system."

    if prediction == "Brake Wear":
        return "Brake pad thickness and brake wear sensor values indicate advanced wear beyond recommended service threshold."

    if prediction == "Oil Pressure Warning":
        return "Oil pressure has dropped below safe engine operating range, increasing lubrication failure risk."

    if prediction == "Service Overdue":
        return "Distance since last service exceeds BMW maintenance interval, increasing probability of cumulative wear faults."

    return "No critical anomaly detected from the current sensor profile."
