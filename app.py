import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ----------------------------
# Load Model Components
# ----------------------------

model = load_model("ckd_mlp_model.keras")
scaler = joblib.load("ckd_scaler.pkl")
encoder = joblib.load("stage_encoder.pkl")

# ----------------------------
# App Title
# ----------------------------

st.title("CKD Stage Prediction System")
st.write("Predict Chronic Kidney Disease Stage using Deep Learning")

# ----------------------------
# Patient Input Fields
# ----------------------------

st.header("Patient Medical Information")

age = st.number_input("Age", 1, 120)
bp = st.number_input("Blood Pressure")
sg = st.number_input("Specific Gravity")
al = st.number_input("Albumin")
su = st.number_input("Sugar")
rbc = st.number_input("Red Blood Cells")
pc = st.number_input("Pus Cell")
pcc = st.number_input("Pus Cell Clumps")
ba = st.number_input("Bacteria")
bgr = st.number_input("Blood Glucose Random")
bu = st.number_input("Blood Urea")
sc = st.number_input("Serum Creatinine")
sod = st.number_input("Sodium")
pot = st.number_input("Potassium")
hemo = st.number_input("Hemoglobin")
pcv = st.number_input("Packed Cell Volume")
wc = st.number_input("White Blood Cell Count")
rc = st.number_input("Red Blood Cell Count")

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("Predict CKD Stage"):

    patient_data = np.array([
        age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,
        bu,sc,sod,pot,hemo,pcv,wc,rc
    ]).reshape(1,-1)

    # Scale features
    patient_data = scaler.transform(patient_data)

    # Prediction
    prediction = model.predict(patient_data)

    stage_index = np.argmax(prediction)

    stage_label = encoder.inverse_transform([stage_index])[0]

    st.success(f"Predicted CKD Stage: {stage_label}")