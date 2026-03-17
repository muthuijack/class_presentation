import os
# Force Legacy Keras to stop the 'batch_shape' error
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
    # 1. MANUALLY RECONSTRUCT the architecture from your notebook
    # This bypasses the corrupted metadata in the .keras file
    model = Sequential([
        Input(shape=(22,)),  # Your notebook shows 22 features
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax') # Your notebook has 6 stages (0-5)
    ])
    
    # 2. Load only the weights into our manual skeleton
    temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
    model.set_weights(temp_model.get_weights())
    
    scaler = joblib.load("ckd_scaler.pkl")
    encoder = joblib.load("stage_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"❌ Model Loading Error: {e}")
    st.stop()

st.title("🏥 CKD Stage Prediction System")

# --- UI Inputs (18 variables) ---
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

# --- Mapping & Prediction ---
mapping = {"Normal": 1, "Abnormal": 0, "Present": 1, "Not Present": 0}

if st.button("Predict CKD Stage", type="primary"):
    # 1. The 18 features from your UI
    base_features = [
        age, bp, float(sg), al, su, mapping[rbc], mapping[pc], 
        mapping[pcc], mapping[ba], bgr, bu, sc, sod, pot, hemo, pcv, wc, rc
    ]
    
    # 2. FILLING THE GAP: Your model expects 22 features.
    # In your notebook, the training data had 23 columns (1 target + 22 features).
    # We add 4 zeros to satisfy the InputLayer shape.
    final_features = base_features + [0.0, 0.0, 0.0, 0.0] 
    
    features_array = np.array(final_features).reshape(1, -1)
    
    try:
        scaled_data = scaler.transform(features_array)
        prediction = model.predict(scaled_data)
        stage_idx = np.argmax(prediction)
        stage_label = encoder.inverse_transform([stage_idx])[0]
        st.success(f"**Predicted Result:** {stage_label}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
