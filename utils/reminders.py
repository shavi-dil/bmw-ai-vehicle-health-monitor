from datetime import date, timedelta
from typing import List, Dict


def generate_reminders(values: dict) -> List[Dict]:
    reminders = []

    if values["last_service_km"] > 20000:
        reminders.append(
            {
                "reminder_type": "Service",
                "reminder_message": "Vehicle service is overdue. Book a maintenance session.",
                "due_date": str(date.today() + timedelta(days=7)),
            }
        )

    if values["battery_voltage"] < 12.0:
        reminders.append(
            {
                "reminder_type": "Battery",
                "reminder_message": "Battery voltage is low. Run battery health diagnostics.",
                "due_date": str(date.today() + timedelta(days=10)),
            }
        )

    if values["brake_pad_thickness"] < 4.0:
        reminders.append(
            {
                "reminder_type": "Brake",
                "reminder_message": "Brake pad thickness below safety margin. Schedule inspection.",
                "due_date": str(date.today() + timedelta(days=5)),
            }
        )

    if values["oil_temp"] > 110:
        reminders.append(
            {
                "reminder_type": "Cooling/Oil",
                "reminder_message": "Oil temperature elevated. Inspect oil and cooling system.",
                "due_date": str(date.today() + timedelta(days=5)),
            }
        )

    return reminders
