import os
# MANDATORY: This must be the very first thing in the script to handle the Keras error
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------
# Load Model Components (Cached)
# ----------------------------

# We use cache_resource for the model and scaler so they only load ONCE
@st.cache_resource
def load_assets():
    model = load_model("ckd_mlp_model.keras")
    scaler = joblib.load("ckd_scaler.pkl")
    encoder = joblib.load("stage_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------------------
# App Title
# ----------------------------

st.title("CKD Stage Prediction System")
st.write("Predict Chronic Kidney Disease Stage using Deep Learning")

# ----------------------------
# Patient Input Fields
# ----------------------------

st.header("Patient Medical Information")

# Using columns makes the UI much cleaner
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, value=40)
    bp = st.number_input("Blood Pressure", value=80.0)
    sg = st.number_input("Specific Gravity", value=1.02)
    al = st.number_input("Albumin", value=0.0)
    su = st.number_input("Sugar", value=0.0)
    rbc = st.number_input("Red Blood Cells (0/1)", value=0)

with col2:
    pc = st.number_input("Pus Cell (0/1)", value=0)
    pcc = st.number_input("Pus Cell Clumps (0/1)", value=0)
    ba = st.number_input("Bacteria (0/1)", value=0)
    bgr = st.number_input("Blood Glucose Random", value=120.0)
    bu = st.number_input("Blood Urea", value=36.0)
    sc = st.number_input("Serum Creatinine", value=1.2)

with col3:
    sod = st.number_input("Sodium", value=138.0)
    pot = st.number_input("Potassium", value=4.4)
    hemo = st.number_input("Hemoglobin", value=15.0)
    pcv = st.number_input("Packed Cell Volume", value=44.0)
    wc = st.number_input("White Blood Cell Count", value=7800.0)
    rc = st.number_input("Red Blood Cell Count", value=5.2)

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("Predict CKD Stage", type="primary"):
    # Combine inputs into numpy array
    patient_data = np.array([
        age, bp, sg, al, su, rbc, pc, pcc, ba, bgr,
        bu, sc, sod, pot, hemo, pcv, wc, rc
    ]).reshape(1, -1)

    # Scale features
    patient_data_scaled = scaler.transform(patient_data)

    # Prediction
    prediction = model.predict(patient_data_scaled)
    stage_index = np.argmax(prediction)
    stage_label = encoder.inverse_transform([stage_index])[0]

    st.divider()
    st.subheader("Results")
    st.success(f"**Predicted CKD Stage:** {stage_label}")
