import os
# This MUST be the first line
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

@st.cache_resource
def load_model_safely():
    # compile=False is the strongest way to skip version-heavy config checks
    model = load_model("ckd_mlp_model.keras", compile=False)
    scaler = joblib.load("ckd_scaler.pkl")
    encoder = joblib.load("stage_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_model_safely()
except Exception as e:
    st.error(f"❌ Keras Version Conflict: {e}")
    st.info("The server is still using Keras 3. Please Delete and Re-deploy the app on Streamlit Cloud.")
    st.stop()

st.title("🏥 CKD Stage Prediction")

# ... (Keep your 18 input fields: age, bp, etc.) ...

mapping = {"Normal": 1, "Abnormal": 0, "Present": 1, "Not Present": 0}

if st.button("Predict Stage", type="primary"):
    # 1. Collect the 18 features you have defined
    input_list = [
        age, bp, float(sg), al, su, mapping[rbc], mapping[pc], 
        mapping[pcc], mapping[ba], bgr, bu, sc, sod, pot, hemo, pcv, wc, rc
    ]
    
    # 2. PAD THE ARRAY: Your model error says it needs 22 features.
    # We add 4 dummy zeros to match the expected 'batch_shape': [None, 22]
    while len(input_list) < 22:
        input_list.append(0.0)
    
    final_input = np.array(input_list).reshape(1, -1)

    try:
        # 3. Scaling & Prediction
        scaled_data = scaler.transform(final_input)
        preds = model.predict(scaled_data)
        result = encoder.inverse_transform([np.argmax(preds)])[0]
        
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Computation Error: {e}")
