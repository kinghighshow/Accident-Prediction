import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import json
import requests

# Load Keras model
model = tf.keras.models.load_model("accidents_deep_model.keras")

# Load accident possibility by state
state_model = joblib.load("accidents_by_state_model.pkl")

# Load encoders
encoders = {}
encoder_names = ['traffic_control', 'trafficway', 'alignment', 'road_defect', 'surface', 'visibility']
for name in encoder_names:
    encoders[name] = joblib.load(f'encoders/{name}_encoder.pkl')

# List of Nigerian states
states = [col.replace('_State', '') for col in state_model.columns if '_State' in col]

# Load Nigeria GeoJSON
geojson_url = "https://raw.githubusercontent.com/deldersveld/topojson/master/countries/nigeria/nigeria-states.json"
geojson_data = requests.get(geojson_url).json()

# Mapping from user input
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
seasons = ['Dry', 'Rainy']

# Weather risk mapping
weather_ranks = {
    'CLEAR': 1,
    'CLOUDY/OVERCAST': 2,
    'OTHER': 3,
    'RAIN': 4,
    'SEVERE CROSS WIND GATE': 4,
    'SLEET/HAIL': 5,
    'SNOW': 5,
    'FREEZING RAIN/DRIZZLE': 6,
    'FOG/SMOKE/HAZE': 7,
    'BLOWING SNOW': 7,
    'BLOWING SAND, SOIL, DIRT': 6
}

# Streamlit UI
st.title("🚨 Road Accident Risk Predictor")
st.markdown("Enter trip details below:")

state = st.selectbox("Select State", states)
hour = st.slider("Hour of Day", 0, 23)
day_of_week = st.selectbox("Day of Week", days)
month = st.slider("Month", 1, 12)
season = st.selectbox("Season", seasons)

# Use label-based visibility instead of numeric score
visibility_label = st.selectbox("Visibility Level", ['excellent', 'moderate', 'poor', 'severe'])
visibility_score = {'excellent': 1, 'moderate': 2, 'poor': 3, 'severe': 4}[visibility_label]

# Categorical options
weather = st.selectbox("Weather", list(weather_ranks.keys()))
traffic_control = st.selectbox("Traffic Control", encoders['traffic_control'].classes_)
trafficway = st.selectbox("Trafficway", encoders['trafficway'].classes_)
alignment = st.selectbox("Road Alignment", encoders['alignment'].classes_)
road_defect = st.selectbox("Road Defect", encoders['road_defect'].classes_)
surface = st.selectbox("Road Surface", encoders['surface'].classes_)
visibility = st.selectbox("Visibility Type", encoders['visibility'].classes_)

def preprocess_input():
    rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
    night_time = 1 if hour < 6 or hour > 19 else 0
    weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
    poor_visibility = 1 if visibility_score >= 3 else 0

    df = pd.DataFrame([{
        'hour': hour,
        'day_of_week': days.index(day_of_week),
        'month': month,
        'weekend': weekend,
        'rush_hour': rush_hour,
        'night_time': night_time,
        'season': seasons.index(season),
        'poor_visibility': poor_visibility,
        'visibility_score': visibility_score,
        'traffic_control_ordinal': encoders['traffic_control'].transform([traffic_control])[0],
        'weather_ordinal': weather_ranks[weather],
        'trafficway_ordinal': encoders['trafficway'].transform([trafficway])[0],
        'alignment_ordinal': encoders['alignment'].transform([alignment])[0],
        'road_defect_ordinal': encoders['road_defect'].transform([road_defect])[0],
        'surface_ordinal': encoders['surface'].transform([surface])[0],
        'visibility_encoded': encoders['visibility'].transform([visibility])[0],
    }])

    return df

def adjust_prediction(pred, state):
    state_col = f"{state}_State"
    if state_col in state_model.columns:
        state_weight = state_model[state_col].values[0]
        adjustment = state_model['Accident_Possibility'].values[0]
        return pred * adjustment * state_weight
    return pred

def highlight_state_on_map(state):
    df = pd.DataFrame({'State': [state], 'value': [1]})
    fig = px.choropleth(
        df,
        geojson=geojson_data,
        featureidkey="properties.NAME_1",
        locations="State",
        color="value",
        color_continuous_scale="reds",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title=f"Selected State: {state}", margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig)

# Run Prediction
if st.button("Predict"):
    input_df = preprocess_input()
    pred = model.predict(input_df)[0][0]
    adjusted_pred = adjust_prediction(pred, state)

    st.metric("Base Accident Probability", f"{pred:.2%}")
    st.metric("Adjusted (State-based) Risk", f"{adjusted_pred:.2%}")
    highlight_state_on_map(state)
