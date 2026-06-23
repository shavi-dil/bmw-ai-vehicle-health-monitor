from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from backend import crud, models, schemas
from backend.database import get_db, init_db
from utils.explainability import build_explanation
from utils.prediction import (
    MODEL_THRESHOLDS,
    calculate_health_score,
    calculate_risk_level,
    estimate_failure_window,
    predict_fault,
)
from utils.recommendations import get_recommendations
from utils.reminders import generate_reminders

app = FastAPI(title="BMW AI Predictive Maintenance API", version="1.0.0")
init_db()


@app.post("/register")
def register_user(payload: schemas.RegisterRequest, db: Session = Depends(get_db)):
    user = crud.register_user(
        db,
        full_name=payload.full_name,
        email=payload.email.strip().lower(),
        password=payload.password,
        vehicle_registration_number=payload.vehicle_registration_number.strip().upper(),
        bmw_model=payload.bmw_model,
    )

    if user is None:
        raise HTTPException(status_code=400, detail="User already exists with this email or vehicle registration number.")

    profile = crud.get_user_profile(db, user.id)
    _, vehicle = profile

    return {
        "message": "Registration successful",
        "user_id": user.id,
        "vehicle_id": vehicle.id,
    }


@app.post("/login")
def login_user(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.login_user(db, payload.identifier.strip(), payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    profile = crud.get_user_profile(db, user.id)
    _, vehicle = profile

    return {
        "message": "Login successful",
        "user_id": user.id,
        "full_name": user.full_name,
        "vehicle_id": vehicle.id if vehicle else None,
    }


@app.get("/user/profile", response_model=schemas.UserProfileResponse)
def get_profile(user_id: int, db: Session = Depends(get_db)):
    profile = crud.get_user_profile(db, user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")

    user, vehicle = profile
    return schemas.UserProfileResponse(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        created_at=user.created_at,
        vehicle_id=vehicle.id if vehicle else None,
        registration_number=vehicle.registration_number if vehicle else None,
        bmw_model=vehicle.bmw_model if vehicle else None,
        mileage=vehicle.mileage if vehicle else None,
        vehicle_age=vehicle.vehicle_age if vehicle else None,
    )


@app.post("/vehicles/update")
def update_vehicle(payload: schemas.VehicleUpdateRequest, user_id: int, db: Session = Depends(get_db)):
    vehicle = crud.update_vehicle_profile(db, user_id, payload.mileage, payload.vehicle_age)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    return {
        "message": "Vehicle profile updated",
        "vehicle_id": vehicle.id,
        "mileage": vehicle.mileage,
        "vehicle_age": vehicle.vehicle_age,
    }


@app.post("/diagnostics/predict", response_model=schemas.DiagnosticPredictResponse)
def diagnostics_predict(payload: schemas.DiagnosticPredictRequest, db: Session = Depends(get_db)):
    values = payload.dict()
    model_name = db.query(models.Vehicle).filter(models.Vehicle.id == payload.vehicle_id).first()
    selected_model = model_name.bmw_model if model_name else "BMW 3 Series"
    thresholds = MODEL_THRESHOLDS.get(selected_model, MODEL_THRESHOLDS["BMW 3 Series"])

    prediction, top_probability, probability_table = predict_fault(values, selected_model)
    health_score = calculate_health_score(values, thresholds)
    risk_level = calculate_risk_level(health_score)
    recommendations = get_recommendations(prediction)
    explanation = build_explanation(prediction, values, thresholds)
    estimated_failure_window = estimate_failure_window(risk_level, values, thresholds)

    record = crud.create_diagnostic_record(
        db,
        user_id=payload.user_id,
        vehicle_id=payload.vehicle_id,
        values=values,
        predicted_fault=prediction,
        health_score=health_score,
        risk_level=risk_level,
        recommendation="; ".join(recommendations),
    )

    for reminder in generate_reminders(values):
        crud.create_reminder(
            db,
            user_id=payload.user_id,
            vehicle_id=payload.vehicle_id,
            reminder_type=reminder["reminder_type"],
            reminder_message=reminder["reminder_message"],
            due_date=reminder["due_date"],
        )

    return schemas.DiagnosticPredictResponse(
        predicted_fault=prediction,
        predicted_probability=top_probability,
        health_score=health_score,
        risk_level=risk_level,
        estimated_failure_window=estimated_failure_window,
        explanation=explanation,
        recommendations=recommendations,
        probability_table=probability_table,
        record_id=record.id,
    )


@app.get("/diagnostics/history/{user_id}", response_model=List[schemas.DiagnosticHistoryItem])
def diagnostics_history(user_id: int, db: Session = Depends(get_db)):
    history = crud.list_diagnostics(db, user_id)
    return history


@app.get("/reminders/{user_id}", response_model=List[schemas.ReminderResponse])
def reminders(user_id: int, db: Session = Depends(get_db)):
    items = crud.list_reminders(db, user_id)
    return [
        schemas.ReminderResponse(
            id=item.id,
            reminder_type=item.reminder_type,
            reminder_message=item.reminder_message,
            due_date=item.due_date,
            status=item.status,
        )
        for item in items
    ]


@app.post("/reminders/create")
def reminders_create(payload: schemas.ReminderCreateRequest, db: Session = Depends(get_db)):
    item = crud.create_reminder(
        db,
        user_id=payload.user_id,
        vehicle_id=payload.vehicle_id,
        reminder_type=payload.reminder_type,
        reminder_message=payload.reminder_message,
        due_date=payload.due_date,
    )

    return {
        "id": item.id,
        "message": "Reminder created",
    }
