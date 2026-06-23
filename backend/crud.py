from typing import List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend import models
from backend.auth import hash_password, verify_password


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()


def get_vehicle_by_registration(db: Session, registration: str) -> Optional[models.Vehicle]:
    return db.query(models.Vehicle).filter(models.Vehicle.registration_number == registration).first()


def register_user(
    db: Session,
    full_name: str,
    email: str,
    password: str,
    vehicle_registration_number: str,
    bmw_model: str,
):
    existing_email = get_user_by_email(db, email)
    existing_registration = get_vehicle_by_registration(db, vehicle_registration_number)

    if existing_email or existing_registration:
        return None

    user = models.User(
        full_name=full_name,
        email=email,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.flush()

    vehicle = models.Vehicle(
        user_id=user.id,
        registration_number=vehicle_registration_number,
        bmw_model=bmw_model,
        mileage=0,
        vehicle_age=0,
    )
    db.add(vehicle)
    db.commit()
    db.refresh(user)

    return user


def login_user(db: Session, identifier: str, password: str):
    user = (
        db.query(models.User)
        .join(models.Vehicle, models.Vehicle.user_id == models.User.id)
        .filter(or_(models.User.email == identifier, models.Vehicle.registration_number == identifier))
        .first()
    )

    if user is None:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


def get_user_profile(db: Session, user_id: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return None

    vehicle = db.query(models.Vehicle).filter(models.Vehicle.user_id == user_id).first()
    return user, vehicle


def update_vehicle_profile(db: Session, user_id: int, mileage: int, vehicle_age: int):
    vehicle = db.query(models.Vehicle).filter(models.Vehicle.user_id == user_id).first()
    if not vehicle:
        return None

    vehicle.mileage = mileage
    vehicle.vehicle_age = vehicle_age
    db.commit()
    db.refresh(vehicle)
    return vehicle


def create_diagnostic_record(
    db: Session,
    user_id: int,
    vehicle_id: int,
    values: dict,
    predicted_fault: str,
    health_score: int,
    risk_level: str,
    recommendation: str,
):
    record = models.DiagnosticRecord(
        user_id=user_id,
        vehicle_id=vehicle_id,
        mileage=values["mileage"],
        engine_temp=values["engine_temp"],
        oil_temp=values["oil_temp"],
        battery_voltage=values["battery_voltage"],
        tyre_pressure=values["tyre_pressure"],
        brake_pad_thickness=values["brake_pad_thickness"],
        vibration_level=values["vibration_level"],
        fuel_efficiency=values["fuel_efficiency"],
        last_service_km=values["last_service_km"],
        predicted_fault=predicted_fault,
        health_score=health_score,
        risk_level=risk_level,
        recommendation=recommendation,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_diagnostics(db: Session, user_id: int) -> List[models.DiagnosticRecord]:
    return (
        db.query(models.DiagnosticRecord)
        .filter(models.DiagnosticRecord.user_id == user_id)
        .order_by(models.DiagnosticRecord.created_at.desc())
        .all()
    )


def create_reminder(db: Session, user_id: int, vehicle_id: int, reminder_type: str, reminder_message: str, due_date: str):
    reminder = models.Reminder(
        user_id=user_id,
        vehicle_id=vehicle_id,
        reminder_type=reminder_type,
        reminder_message=reminder_message,
        due_date=due_date,
        status="pending",
    )
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return reminder


def list_reminders(db: Session, user_id: int):
    return (
        db.query(models.Reminder)
        .filter(models.Reminder.user_id == user_id)
        .order_by(models.Reminder.id.desc())
        .all()
    )
