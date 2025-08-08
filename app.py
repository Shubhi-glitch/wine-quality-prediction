import streamlit as st
import joblib
import numpy as np
import time

# Load model and scaler
model = joblib.load("wine_model.joblib")
scaler = joblib.load("wine_scaler.joblib")

# Page config
st.set_page_config(page_title="🍷 Wine Quality Prediction", page_icon="🍇", layout="centered")

# Header
st.title("🍷 Wine Quality Prediction")
st.markdown("""
Welcome to the **Wine Quality Predictor**!  
This tool uses a trained **Random Forest Machine Learning model** to predict wine quality based on its chemical properties.  
Adjust the sliders below and see your prediction come to life!  
""")

st.markdown("---")

# Sidebar for inputs
st.sidebar.header("⚙️ Input Wine Features")
st.sidebar.write("Adjust the values to match your wine sample.")

features = {
    "Fixed Acidity": st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 8.0, 0.1),
    "Volatile Acidity": st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5, 0.01),
    "Citric Acid": st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01),
    "Residual Sugar": st.sidebar.slider("Residual Sugar", 0.5, 15.0, 2.5, 0.1),
    "Chlorides": st.sidebar.slider("Chlorides", 0.01, 0.2, 0.05, 0.001),
    "Free Sulfur Dioxide": st.sidebar.slider("Free Sulfur Dioxide", 1, 72, 15, 1),
    "Total Sulfur Dioxide": st.sidebar.slider("Total Sulfur Dioxide", 6, 289, 46, 1),
    "Density": st.sidebar.slider("Density", 0.9900, 1.0050, 0.9960, 0.0001),
    "pH": st.sidebar.slider("pH", 2.8, 4.0, 3.3, 0.01),
    "Sulphates": st.sidebar.slider("Sulphates", 0.3, 2.0, 0.65, 0.01),
    "Alcohol": st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0, 0.1)
}

# Convert to array & scale
input_data = np.array(list(features.values())).reshape(1, -1)
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict Quality"):
    prediction = model.predict(input_scaled)[0]

    # Quality label & color
    if prediction >= 7:
        quality_label = "Good Quality 🍾"
        color = "#28a745"  # green
    elif prediction >= 5:
        quality_label = "Average Quality 🍷"
        color = "#ffc107"  # yellow
    else:
        quality_label = "Poor Quality 🍇"
        color = "#dc3545"  # red

    # Animation effect
    placeholder = st.empty()
    for i in range(0, prediction + 1):
        placeholder.markdown(f"""
            <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; text-align:center; font-size:20px;">
                <b>Predicted Wine Quality Score:</b> 
                <span style="color:{color}; font-size:28px;">{i}</span><br>
                <span style="font-size:22px; color:{color};">{quality_label if i == prediction else ""}</span>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.1)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Machine Learning")
