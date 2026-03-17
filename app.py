import os
# Force Legacy Keras before any other imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set Page Config
st.set_page_config(page_title="CKD Prediction", layout="wide")

# ----------------------------
# Load Model Components (Cached)
# ----------------------------
@st.cache_resource
def load_assets():
    # compile=False bypasses the 'batch_shape' and 'optional' version errors
    model = load_model("ckd_mlp_model.keras", compile=False)
    scaler = joblib.load("ckd_scaler.pkl")
    encoder = joblib.load("stage_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"⚠️ Model Loading Error: {e}")
    st.info("Check if your requirements.txt has 'tensorflow-cpu==2.15.0'")
    st.stop()

# ----------------------------
# UI Header
# ----------------------------
st.title("🏥 CKD Stage Prediction System")
st.markdown("---")

# ----------------------------
# Input Form
# ----------------------------
with st.container():
    st.subheader("Patient Clinical Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        bp = st.number_input("Blood Pressure", 50, 200, 80)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
        al = st.slider("Albumin (0-5)", 0, 5, 0)
        su = st.slider("Sugar (0-5)", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"], index=0)

    with col2:
        pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"], index=0)
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"], index=1)
        ba = st.selectbox("Bacteria", ["Present", "Not Present"], index=1)
        bgr = st.number_input("Blood Glucose Random", 50, 500, 120)
        bu = st.number_input("Blood Urea", 5, 300, 36)
        sc = st.number_input("Serum Creatinine", 0.1, 15.0, 1.2)

    with col3:
        sod = st.number_input("Sodium", 100, 160, 138)
        pot = st.number_input("Potassium", 2.0, 8.0, 4.4)
        hemo = st.number_input("Hemoglobin", 3.0, 18.0, 15.0)
        pcv = st.number_input("Packed Cell Volume", 10, 60, 44)
        wc = st.number_input("WBC Count", 2000, 20000, 7800)
        rc = st.number_input("RBC Count", 2.0, 8.0, 5.2)

# ----------------------------
# Data Processing & Prediction
# ----------------------------

# Mapping categorical text to numbers (Adjust based on your training encoding)
mapping = {"Normal": 1, "Abnormal": 0, "Present": 1, "Not Present": 0}

if st.button("Generate Diagnostic Prediction", type="primary"):
    # Create input array
    features = np.array([
        age, bp, float(sg), al, su, 
        mapping[rbc], mapping[pc], mapping[pcc], mapping[ba],
        bgr, bu, sc, sod, pot, hemo, pcv, wc, rc
    ]).reshape(1, -1)

    try:
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)
        stage_idx = np.argmax(prediction)
        stage_name = encoder.inverse_transform([stage_idx])[0]

        st.markdown("---")
        st.subheader("Diagnostic Result")
        st.success(f"The predicted condition is: **{stage_name}**")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
