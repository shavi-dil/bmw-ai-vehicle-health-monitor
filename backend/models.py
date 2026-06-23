from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(150), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    vehicles = relationship("Vehicle", back_populates="user", cascade="all, delete-orphan")
    diagnostics = relationship("DiagnosticRecord", back_populates="user", cascade="all, delete-orphan")
    reminders = relationship("Reminder", back_populates="user", cascade="all, delete-orphan")


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    registration_number = Column(String(80), unique=True, index=True, nullable=False)
    bmw_model = Column(String(80), nullable=False)
    mileage = Column(Integer, default=0)
    vehicle_age = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="vehicles")
    diagnostics = relationship("DiagnosticRecord", back_populates="vehicle", cascade="all, delete-orphan")
    reminders = relationship("Reminder", back_populates="vehicle", cascade="all, delete-orphan")


class DiagnosticRecord(Base):
    __tablename__ = "diagnostic_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False)

    mileage = Column(Integer, nullable=False)
    engine_temp = Column(Float, nullable=False)
    oil_temp = Column(Float, nullable=False)
    battery_voltage = Column(Float, nullable=False)
    tyre_pressure = Column(Float, nullable=False)
    brake_pad_thickness = Column(Float, nullable=False)
    vibration_level = Column(Float, nullable=False)
    fuel_efficiency = Column(Float, nullable=False)
    last_service_km = Column(Integer, nullable=False)

    predicted_fault = Column(String(120), nullable=False)
    health_score = Column(Integer, nullable=False)
    risk_level = Column(String(50), nullable=False)
    recommendation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="diagnostics")
    vehicle = relationship("Vehicle", back_populates="diagnostics")


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False)
    reminder_type = Column(String(80), nullable=False)
    reminder_message = Column(Text, nullable=False)
    due_date = Column(String(30), nullable=False)
    status = Column(String(30), default="pending", nullable=False)

    user = relationship("User", back_populates="reminders")
    vehicle = relationship("Vehicle", back_populates="reminders")
