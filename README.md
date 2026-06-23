# BMW AI Predictive Maintenance System

An AI-powered predictive maintenance and vehicle health monitoring system for BMW vehicles.

## Features
- BMW vehicle diagnostics
- AI fault prediction with probability outputs
- Predictive maintenance and health scoring
- Engineering recommendations and explainable reasoning
- User registration and login (email or vehicle registration)
- Vehicle profile management
- Persistent diagnostic history and trend analysis
- Reminder generation and reminder management
- FastAPI backend with SQLite database
- Computer Vision prototype (grayscale + edge detection)
- AI Damage Detection page (YOLO prototype + OpenCV damage-risk assessment)
- EV Battery AI page (SoH prediction, degradation risk, replacement window)
- Driving Behaviour AI page (safety score, risk level, behavior insights)

## Technologies
- Python
- Streamlit
- FastAPI
- SQLite / SQLAlchemy
- Scikit-Learn
- Random Forest
- OpenCV
- YOLO (Ultralytics)
- Predictive analytics modules

## AI Techniques Used
- Machine Learning (Random Forest)
- Computer Vision (OpenCV)
- YOLO object detection (prototype mode unless custom damage model is trained)
- Predictive analytics and scoring models
- Explainable AI-style rule-based reasoning

## Project Structure
```
bmw-ai-vehicle-health-monitor/
	app.py
	requirements.txt
	README.md
	fault_prediction_model.pkl
	backend/
		__init__.py
		main.py
		database.py
		models.py
		schemas.py
		auth.py
		crud.py
	data/
		bmw_vehicle_sensor_data.csv
	models/
		fault_prediction_model.pkl
	utils/
		__init__.py
		prediction.py
		recommendations.py
		reminders.py
		explainability.py
	cv_prototype/
		__init__.py
		image_processing.py
```

## Local Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run backend:
```bash
uvicorn backend.main:app --reload
```

3. Run frontend:
```bash
streamlit run app.py
```

Optional backend URL override for Streamlit:
```bash
export BMW_API_URL="http://127.0.0.1:8000"
```

## Notes
- Passwords are hashed using `passlib[bcrypt]`.
- Duplicate registrations are blocked by email and vehicle registration number.
- If backend is unavailable, Streamlit supports a local guest diagnostic mode.

## Author
Shavini Joseph
Bachelor of Computer Science (Artificial Intelligence)
