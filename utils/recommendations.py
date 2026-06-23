def get_recommendations(prediction: str):
    lookup = {
        "Battery Degradation": [
            "Inspect battery terminals and cable resistance.",
            "Perform battery State of Health (SoH) diagnostic.",
            "Schedule battery replacement within 30 days.",
        ],
        "Oil Pressure Warning": [
            "Check engine oil level immediately.",
            "Inspect oil filter and pressure relief valve.",
            "Avoid heavy driving until oil pressure normalizes.",
        ],
        "Brake Wear": [
            "Inspect brake pads and rotor condition.",
            "Replace brake pads if below minimum thickness.",
            "Schedule brake service within 7 days.",
        ],
        "Engine Overheating": [
            "Check coolant level and pressure test cooling loop.",
            "Inspect radiator fan response and thermostat behavior.",
            "Inspect hoses and pump for leaks.",
        ],
        "Tyre Pressure Issue": [
            "Adjust tyre pressure to BMW recommended PSI.",
            "Check for punctures or valve leakage.",
            "Run wheel balancing if vibration persists.",
        ],
        "Suspension/Vibration Issue": [
            "Inspect struts, bushings, and suspension joints.",
            "Check wheel alignment and balancing.",
            "Perform underbody vibration inspection.",
        ],
        "Service Overdue": [
            "Schedule full BMW service appointment.",
            "Replace oil and filters.",
            "Run preventive checks for wear components.",
        ],
        "No Fault": [
            "No immediate action required.",
            "Continue normal preventive maintenance schedule.",
        ],
    }
    return lookup.get(prediction, ["Schedule a full vehicle inspection."])
