# Accident Prediction App 🚧

This Streamlit web application predicts the likelihood of a road accident occurring based on location, time, weather conditions, and traffic data.

## 🔍 Features

- Predicts accidents based on:
  - Time of day
  - Weather
  - Road alignment and surface
  - Traffic controls
  - State-based accident likelihood
- Visual map of selected Nigerian state with dynamic highlighting

## 📦 Folder Structure

accident_app/
├── app.py # Main Streamlit app
├── models/
│ ├── accidents_deep_model.keras
│ └── accidents_by_state_model.pkl
├── maps/
│ └── nigeria_states.geojson
├── utils/
│ └── encoders.py # Manual encoding dictionaries
├── requirements.txt
├── .gitignore
└── README.md
