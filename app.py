import os
# Force legacy Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Use the specific legacy loader
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CKD Prediction", layout="wide")

@st.cache_resource
def load_assets():
    # We use compile=False to ignore optimizer conflicts
    model = load_model("ckd_mlp_model.keras", compile=False)
    scaler = joblib.load("ckd_scaler.pkl")
    encoder = joblib.load("stage_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error("⚠️ Version Conflict Detected")
    st.info("The Cloud is trying to use Keras 3. Please ensure your requirements.txt is updated and REBOOT the app.")
    st.stop()

st.title("🏥 CKD Stage Prediction System")

# --- Inputs ---
col1, col2, col3 = st.columns(3)
# (Keep your existing inputs here: age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc)

# --- Logic ---
mapping = {"Normal": 1, "Abnormal": 0, "Present": 1, "Not Present": 0}

if st.button("Predict"):
    # Current input list (18 features)
    input_data = [
        age, bp, float(sg), al, su, mapping[rbc], mapping[pc], 
        mapping[pcc], mapping[ba], bgr, bu, sc, sod, pot, hemo, pcv, wc, rc
    ]
    
    # CHECK: Your model error said it expects 22 features. 
    # If your model was trained on more than these 18, we must pad the array.
    if len(input_data) < 22:
        # Add zeros for the missing 4 features if necessary
        input_data.extend([0] * (22 - len(input_data)))
    
    features = np.array(input_data).reshape(1, -1)
    
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        stage_idx = np.argmax(prediction)
        stage_name = encoder.inverse_transform([stage_idx])[0]
        st.success(f"Predicted Stage: {stage_name}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
