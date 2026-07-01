# BMW AI Predictive Maintenance System

An AI-powered predictive maintenance and vehicle health monitoring system for BMW vehicles.

## Features
- BMW vehicle diagnostics
- Realistic dashboard structure:
	- Driver Information
	- Vehicle Health
	- AI Diagnostics
	- Computer Vision Prototype
	- Driving Analytics
- AI fault prediction with probability outputs
- Predictive maintenance and health scoring
- Engineering recommendations and explainable reasoning
- User registration and login (email or vehicle registration)
- Vehicle profile management
- Persistent diagnostic history and trend analysis
- Reminder generation and reminder management
- Supabase PostgreSQL backend persistence
- Computer Vision prototype (grayscale + edge detection)
- AI Damage Detection page (YOLO prototype + OpenCV damage-risk assessment)
- EV Battery AI page (SoH prediction, degradation risk, replacement window)
- Driving Analytics page (safety rating, event insights, weekly report)
- Admin data access via Supabase dashboard only (no admin login in Streamlit UI)

## Technologies
- Python
- Streamlit
- Supabase
- PostgreSQL
- psycopg2
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

2. Create a Streamlit secrets file:
```toml
# .streamlit/secrets.toml
SUPABASE_DB_URL = "postgresql://postgres:<password>@<host>:5432/postgres"
SUPABASE_URL = "https://<project-ref>.supabase.co"
SUPABASE_ANON_KEY = "<public-anon-key>"
```

3. Run frontend:
```bash
streamlit run app.py
```

For Streamlit Cloud, add the same keys in App Settings -> Secrets.

## Supabase Tables

Create these tables in Supabase PostgreSQL (the app also creates them automatically if permissions allow):

1. `users`
- `id` (primary key)
- `full_name`
- `email` (unique)
- `password_hash`
- `created_at`

2. `vehicles`
- `id` (primary key)
- `user_id` (foreign key -> users.id)
- `registration_number` (unique)
- `bmw_model`
- `mileage`
- `vehicle_age`
- `created_at`

3. `diagnostic_records`
- `id` (primary key)
- `user_id` (foreign key -> users.id)
- `vehicle_id` (foreign key -> vehicles.id)
- Core sensor fields used by the AI model
- `predicted_fault`
- `health_score`
- `risk_level`
- `predicted_probability`
- `recommendation`
- `sensor_payload` (JSON of diagnostic inputs)
- `created_at`

4. `reminders`
- `id` (primary key)
- `user_id` (foreign key -> users.id)
- `vehicle_id` (foreign key -> vehicles.id)
- `reminder_type`
- `reminder_message`
- `due_date`
- `status`
- `created_at`

## Admin Access Model

- Admin data access is handled through Supabase dashboard only.
- The public Streamlit application exposes only Register/Login and user features.
- No Admin Login or Admin Panel is presented in the Streamlit UI.

## Notes
- Passwords are hashed using PBKDF2-HMAC-SHA256 before storage.
- Duplicate registrations are blocked by email and vehicle registration number.
- If Supabase credentials are missing, the app shows a setup message instead of crashing.

## Author
Shavini Joseph
Bachelor of Computer Science (Artificial Intelligence)
