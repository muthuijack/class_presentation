import os
# This must be the very first line
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

@st.cache_resource
def load_assets():
    # 1. Manually build the architecture found in your Untitled42.ipynb
    # This avoids reading the 'InputLayer' metadata from the file
    model = Sequential([
        Input(shape=(22,)), 
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax') # 6 stages (0-5) as per your notebook
    ])
    
    # 2. Load the .keras file into a temporary object and extract weights
    # This 'transplants' the math into our fresh, working skeleton
    temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
    model.set_weights(temp_model.get_weights())
    
    scaler = joblib.load("ckd_scaler.pkl")
    encoder = joblib.load("stage_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"❌ Initialization Error: {e}")
    st.stop()

st.title("🏥 CKD Stage Prediction System")

# --- UI Inputs (The 18 variables you have in your app) ---
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
    al = st.slider("Albumin", 0, 5, 0)
    su = st.slider("Sugar", 0, 5, 0)
    rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])

with col2:
    pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
    ba = st.selectbox("Bacteria", ["Present", "Not Present"])
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

# --- Logic & Prediction ---
mapping = {"Normal": 1, "Abnormal": 0, "Present": 1, "Not Present": 0}

if st.button("Predict Stage", type="primary"):
    # 1. Collect your 18 features
    user_inputs = [
        age, bp, float(sg), al, su, mapping[rbc], mapping[pc], 
        mapping[pcc], mapping[ba], bgr, bu, sc, sod, pot, hemo, pcv, wc, rc
    ]
    
    # 2. Add 4 padding zeros to reach the 22 features your model expects
    # In your notebook, the training data used 22 columns
    final_features = user_inputs + [0.0, 0.0, 0.0, 0.0] 
    
    try:
        features_array = np.array(final_features).reshape(1, -1)
        scaled_data = scaler.transform(features_array)
        
        prediction = model.predict(scaled_data)
        stage_idx = np.argmax(prediction)
        
        # This will return "Stage 0", "Stage 1", etc.
        result = encoder.inverse_transform([stage_idx])[0]
        
        st.divider()
        st.success(f"**Predicted Diagnostic Result:** {result}")
    except Exception as e:
        st.error(f"Prediction Calculation Error: {e}")
