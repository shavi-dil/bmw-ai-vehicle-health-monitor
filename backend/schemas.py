from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str = Field(min_length=6)
    vehicle_registration_number: str
    bmw_model: str


class LoginRequest(BaseModel):
    identifier: str
    password: str


class UserProfileResponse(BaseModel):
    id: int
    full_name: str
    email: str
    created_at: datetime
    vehicle_id: Optional[int] = None
    registration_number: Optional[str] = None
    bmw_model: Optional[str] = None
    mileage: Optional[int] = None
    vehicle_age: Optional[int] = None


class VehicleUpdateRequest(BaseModel):
    mileage: int
    vehicle_age: int


class DiagnosticPredictRequest(BaseModel):
    user_id: int
    vehicle_id: int
    mileage: int
    vehicle_age: int
    engine_temp: float
    oil_temp: float
    battery_voltage: float
    tyre_pressure: float
    brake_pad_thickness: float
    vibration_level: float
    fuel_efficiency: float
    last_service_km: int
    engine_rpm: int
    coolant_temp: float
    oil_pressure: float
    brake_sensor: float
    battery_soh: float
    driving_behavior_score: float
    road_condition_score: float
    weather_impact_score: float
    maintenance_history_score: float


class DiagnosticPredictResponse(BaseModel):
    predicted_fault: str
    predicted_probability: float
    health_score: int
    risk_level: str
    estimated_failure_window: str
    explanation: str
    recommendations: List[str]
    probability_table: List[dict]
    record_id: int


class ReminderCreateRequest(BaseModel):
    user_id: int
    vehicle_id: int
    reminder_type: str
    reminder_message: str
    due_date: str


class ReminderResponse(BaseModel):
    id: int
    reminder_type: str
    reminder_message: str
    due_date: str
    status: str


class DiagnosticHistoryItem(BaseModel):
    id: int
    mileage: int
    engine_temp: float
    oil_temp: float
    battery_voltage: float
    tyre_pressure: float
    brake_pad_thickness: float
    vibration_level: float
    fuel_efficiency: float
    last_service_km: int
    predicted_fault: str
    health_score: int
    risk_level: str
    recommendation: str
    created_at: datetime
