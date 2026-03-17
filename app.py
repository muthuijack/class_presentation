import os
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

@st.cache_resource
def load_fixed_model():
    # MANUALLY RECONSTRUCT based on your Untitled42.ipynb code
    model = Sequential([
        Input(shape=(22,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(6, activation="softmax") 
    ])
    
    # Load the weights from your file into this new structure
    # This avoids the 'InputLayer' config error entirely
    temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
    model.set_weights(temp_model.get_weights())
    
    scaler = joblib.load("ckd_scaler.pkl")
    encoder = joblib.load("stage_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_fixed_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🏥 Chronic Kidney Disease Stage Predictor")

# IMPORTANT: You must provide 22 values to the model. 
# If your UI only has 18 inputs, you MUST add the missing 4 as defaults.
# These columns were in your training data:
# serum_creatinine, gfr, bun, serum_calcium, ana, c3_c4, hematuria, oxalate_levels, 
# urine_ph, blood_pressure, smoking, alcohol, painkiller_usage, family_history, 
# weight_changes, stress_level, months, cluster, ckd_pred

# Example of how to handle the prediction:
if st.button("Predict"):
    # Ensure this list has EXACTLY 22 items in the same order as df.drop("ckd_stage", axis=1)
    # Use 0.0 or average values for lifestyle features if they aren't in your UI
    features_list = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, 0, 0, 0, 0] 
    
    features_array = np.array(features_list).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    
    prediction = model.predict(scaled_features)
    stage = np.argmax(prediction)
    st.success(f"Predicted CKD Stage: {stage}")
