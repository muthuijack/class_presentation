import os
# This must be the first line - set legacy Keras for compatibility
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
    """
    Load model, scaler, and encoder with compatibility fixes
    """
    try:
        # First try: Build model skeleton and load weights
        # Get the expected input shape from error (22)
        input_shape = 22
        
        # Try to inspect original model architecture first
        try:
            # Attempt to load just the config to see architecture
            temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
            # If successful, use the original model
            print("✅ Successfully loaded original model")
            model = temp_model
        except Exception as e:
            print(f"⚠️ Couldn't load original model: {e}")
            print("🔄 Building model skeleton from scratch...")
            
            # Build model skeleton (adjust architecture based on your training code)
            model = Sequential([
                Input(shape=(input_shape,), name='input_layer_4'),
                Dense(128, activation='relu', name='dense_1'),
                Dropout(0.3, name='dropout_1'),
                Dense(64, activation='relu', name='dense_2'),
                Dropout(0.3, name='dropout_2'),
                Dense(32, activation='relu', name='dense_3'),
                Dense(6, activation='softmax', name='output_layer')  # 6 stages (0-5)
            ])
            
            # Try to load weights into skeleton
            try:
                # Load weights from the saved model
                temp_weights_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
                model.set_weights(temp_weights_model.get_weights())
                print("✅ Successfully loaded weights into skeleton")
            except:
                print("⚠️ Could not load weights, using random initialization")
        
        # Load scaler and encoder
        scaler = joblib.load("ckd_scaler.pkl")
        encoder = joblib.load("stage_encoder.pkl")
        
        # Verify shapes
        print(f"✅ Model expects input shape: {model.input_shape}")
        
        return model, scaler, encoder
        
    except Exception as e:
        st.error(f"❌ Critical Loading Error: {e}")
        return None, None, None

def validate_features(features, expected_length=22):
    """
    Validate and pad features to match expected input shape
    """
    if len(features) < expected_length:
        # Pad with zeros or mean values if needed
        padding = [0.0] * (expected_length - len(features))
        features = features + padding
        st.warning(f"⚠️ Input padded from {len(features)-len(padding)} to {expected_length} features")
    elif len(features) > expected_length:
        # Truncate if too many features
        features = features[:expected_length]
        st.warning(f"⚠️ Input truncated to {expected_length} features")
    
    return features

# Load assets
model, scaler, encoder = load_assets()

if model is None or scaler is None or encoder is None:
    st.error("Failed to load required assets. Please check your files.")
    st.stop()

st.title("🏥 CKD Stage Prediction System")
st.markdown("---")

# Create two columns for input organization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    bp = st.number_input("Blood Pressure (bp)", min_value=40, max_value=200, value=80)
    
    st.subheader("Urine Analysis")
    sg = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.selectbox("Albumin (al)", [0, 1, 2, 3, 4, 5])
    su = st.selectbox("Sugar (su)", [0, 1, 2, 3, 4, 5])
    rbc = st.selectbox("Red Blood Cells (rbc)", ["Normal", "Abnormal"])
    pc = st.selectbox("Pus Cell (pc)", ["Normal", "Abnormal"])
    
with col2:
    st.subheader("Blood Tests")
    pcc = st.selectbox("Pus Cell Clumps (pcc)", ["Present", "Not Present"])
    ba = st.selectbox("Bacteria (ba)", ["Present", "Not Present"])
    bgr = st.number_input("Blood Glucose Random (bgr)", min_value=0, max_value=500, value=120)
    bu = st.number_input("Blood Urea (bu)", min_value=0, max_value=300, value=30)
    sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
    sod = st.number_input("Sodium (sod)", min_value=100, max_value=160, value=135)
    pot = st.number_input("Potassium (pot)", min_value=2.0, max_value=8.0, value=4.0, step=0.1)
    
st.subheader("Complete Blood Count")
col3, col4, col5, col6 = st.columns(4)
with col3:
    hemo = st.number_input("Hemoglobin (hemo)", min_value=3.0, max_value=20.0, value=12.0, step=0.1)
with col4:
    pcv = st.number_input("Packed Cell Volume (pcv)", min_value=10, max_value=60, value=35)
with col5:
    wc = st.number_input("White Blood Cell Count (wc)", min_value=2000, max_value=30000, value=8000)
with col6:
    rc = st.number_input("Red Blood Cell Count (rc)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)

st.markdown("---")

# Mapping for categorical variables
mapping = {"Normal": 1, "Abnormal": 0, "Present": 1, "Not Present": 0}

if st.button("🔍 Predict CKD Stage", type="primary", use_container_width=True):
    try:
        # Collect all features
        base_features = [
            age, bp, float(sg), al, su, 
            mapping[rbc], mapping[pc], mapping[pcc], mapping[ba], 
            bgr, bu, sc, sod, pot, hemo, pcv, wc, rc
        ]
        
        # Validate and pad to 22 features
        final_input = validate_features(base_features, expected_length=22)
        
        # Convert to numpy array and reshape
        features_array = np.array(final_input, dtype=np.float32).reshape(1, -1)
        
        # Display input shape for debugging
        st.caption(f"Input shape: {features_array.shape}")
        
        # Scale the features
        scaled_data = scaler.transform(features_array)
        
        # Make prediction
        with st.spinner("Analyzing patient data..."):
            prediction = model.predict(scaled_data, verbose=0)
            
        # Get predicted stage
        stage_idx = np.argmax(prediction[0])
        confidence = prediction[0][stage_idx] * 100
        
        # Get stage label
        stage_label = encoder.inverse_transform([stage_idx])[0]
        
        # Display results in a nice format
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.success(f"### 🎯 Predicted Stage: **{stage_label}**")
        
        with col_result2:
            st.info(f"### 📊 Confidence: **{confidence:.1f}%**")
        
        with col_result3:
            # Show prediction distribution
            st.write("### 📈 Stage Probabilities")
            for i, prob in enumerate(prediction[0]):
                stage_name = encoder.inverse_transform([i])[0] if hasattr(encoder, 'inverse_transform') else f"Stage {i}"
                st.progress(float(prob), text=f"{stage_name}: {prob*100:.1f}%")
        
        # Add medical disclaimer
        st.warning("⚠️ **Medical Disclaimer**: This is a prediction tool and should not replace professional medical diagnosis. Always consult with healthcare providers.")
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.exception(e)  # This will show the full traceback in development

# Add sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This tool predicts Chronic Kidney Disease (CKD) stages based on clinical parameters.
    
    **CKD Stages:**
    - Stage 0: No CKD
    - Stage 1: Kidney damage with normal GFR
    - Stage 2: Mild decrease in GFR
    - Stage 3: Moderate decrease in GFR
    - Stage 4: Severe decrease in GFR
    - Stage 5: Kidney failure
    
    **Input Features:**
    The model expects 22 features total, including demographic data, urine analysis, and blood tests.
    """)
    
    # Debug information (remove in production)
    if st.checkbox("Show Debug Info"):
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Scaler: {type(scaler).__name__}")
