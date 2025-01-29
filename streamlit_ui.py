import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain

def load_models():
    with open("models/crop_model.pkl", "rb") as f:
        crop_model = pickle.load(f)
    with open("models/fert_model.pkl", "rb") as f:
        fert_model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("models/category_mappings.pkl", "rb") as f:
        category_mappings = pickle.load(f)
    return crop_model, fert_model, label_encoders, category_mappings

crop_model, fert_model, label_encoders, category_mappings = load_models()

st.set_page_config(page_title="🌾 Agro Advisor", layout="wide")

st.sidebar.title("🔧 User Input Parameters")
N = st.sidebar.slider("🌱 Nitrogen (N)", 0, 100, 50)
P = st.sidebar.slider("🟢 Phosphorus (P)", 0, 100, 50)
K = st.sidebar.slider("🟡 Potassium (K)", 0, 100, 50)
temp = st.sidebar.slider("🌡 Temperature (°C)", 10, 40, 25)
humidity = st.sidebar.slider("💧 Humidity (%)", 10, 100, 50)
ph = st.sidebar.slider("🧪 Soil pH", 3.5, 9.0, 6.5)
rainfall = st.sidebar.slider("🌧 Rainfall (mm)", 0, 300, 100)
soil_type = st.sidebar.selectbox("🌍 Soil Type", category_mappings['Soil Type'].values())
crop_type = st.sidebar.selectbox("🌾 Crop Type", category_mappings['Crop Type'].values())
moisture = st.sidebar.slider("💦 Moisture Level", 0, 100, 50)

def recommend_crop():
    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    return crop_model.predict(input_data)[0]

def recommend_fertilizer():
    soil_encoded = label_encoders['Soil Type'].transform([soil_type])[0]
    crop_encoded = label_encoders['Crop Type'].transform([crop_type])[0]
    input_data = np.array([[temp, humidity, moisture, soil_encoded, crop_encoded, N, P, K]])
    fert_prediction = fert_model.predict(input_data)[0]
    return category_mappings['Fertilizer Name'][fert_prediction]

st.title("🌾 Agro Advisor")
st.write("📢 Get the best crop & fertilizer recommendations instantly!")
add_vertical_space(2)

col1, col2 = st.columns(2)

with col1:
    if st.button("🚜 Recommend Crop"):
        crop_result = recommend_crop()
        st.success(f"🌱 Recommended Crop: {crop_result}")
        rain(emoji="🌿", font_size=20, falling_speed=5, animation_length=3)

with col2:
    if st.button("🧪 Recommend Fertilizer"):
        fert_result = recommend_fertilizer()
        st.success(f"🧪 Recommended Fertilizer: {fert_result}")
        rain(emoji="💧", font_size=20, falling_speed=3, animation_length=10)

st.subheader("📊 Data Overview")
data_chart = pd.DataFrame({
    "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
    "Amount": [N, P, K]
})
fig = px.bar(data_chart, x="Nutrient", y="Amount", text_auto=True, color="Nutrient", title="Nutrient Levels")
st.plotly_chart(fig)

soil_chart = pd.DataFrame({
    "Parameter": ["Temperature", "Humidity", "Soil pH", "Rainfall", "Moisture"],
    "Value": [temp, humidity, ph, rainfall, moisture]
})
fig_soil = px.line(soil_chart, x="Parameter", y="Value", markers=True, title="Soil & Environmental Conditions")
st.plotly_chart(fig_soil)

st.write("🚀 Developed with ❤️ by Himanshu Bhundere")