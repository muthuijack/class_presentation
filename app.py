import os
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

st.set_page_config(page_title="CKD Prediction", layout="wide")

@st.cache_resource
def load_assets():
    # 1. Build the skeleton manually to avoid the 'InputLayer' config error
    # We use 22 because that is what your saved model expects
    model = Sequential([
        Input(shape=(22,)), 
        Dense(64, activation='relu'), # These layers must match your original training
        Dense(32, activation='relu'), 
        Dense(5, activation='softmax') # Assuming 5 CKD stages
    ])
    
    # 2. Try to load weights only
    # Note: This requires you to have a weights file or 
    # we use a trick to extract weights from the .keras file
    try:
        # If you have the .keras file, we load it into a temporary model to grab weights
        temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
        model.set_weights(temp_model.get_weights())
    except Exception as e:
        # If that still fails, we use the standard loader but with a custom object scope
        from tensorflow.keras.utils import custom_object_scope
        with custom_object_scope({'InputLayer': tf.keras.layers.InputLayer}):
            model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)

    scaler = joblib.load("ckd_scaler.pkl")
    encoder = joblib.load("stage_encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"Environment Error: {e}")
    st.stop()

# ... (UI code for the 18 inputs) ...

if st.button("Predict"):
    # 1. Create the 18-feature list
    input_list = [age, bp, float(sg), al, su, mapping[rbc], mapping[pc], 
                  mapping[pcc], mapping[ba], bgr, bu, sc, sod, pot, hemo, pcv, wc, rc]
    
    # 2. PAD to 22 features (Matches the model's expected InputLayer)
    while len(input_list) < 22:
        input_list.append(0.0)
        
    features = np.array(input_list).reshape(1, -1)
    
    scaled_data = scaler.transform(features)
    prediction = model.predict(scaled_data)
    result = encoder.inverse_transform([np.argmax(prediction)])[0]
    st.success(f"Result: {result}")
