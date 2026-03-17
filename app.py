import os
import sys
import subprocess
import warnings

# Environment fixes (KEEP THESE)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Simple NumPy import (NO hacks)
import numpy as np

# Streamlit config MUST be first Streamlit command
import streamlit as st
st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

# Other imports
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

warnings.filterwarnings('ignore')

# Show environment info
st.sidebar.write("📦 Environment:")
st.sidebar.write(f"Python: {sys.version}")
st.sidebar.write(f"NumPy: {np.__version__}")
st.sidebar.write(f"TensorFlow: {tf.__version__}")


@st.cache_resource
def load_assets():
    """Load model with fallback methods"""
    try:
        scaler = joblib.load("ckd_scaler.pkl")
        encoder = joblib.load("stage_encoder.pkl")
        st.sidebar.success("✅ Scaler and encoder loaded")

        model = None

        try:
            # Build model architecture
            model = Sequential([
                Input(shape=(22,), name='input_layer'),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(6, activation='softmax')
            ])

            # Load trained model
            temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
            model.set_weights(temp_model.get_weights())

            st.sidebar.success("✅ Model loaded successfully")

        except Exception as e:
            st.sidebar.warning(f"Model load failed: {e}")
            return None, None, None

        return model, scaler, encoder

    except Exception as e:
        st.error(f"❌ Loading Error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None


# Load assets
with st.spinner("Loading CKD prediction model..."):
    model, scaler, encoder = load_assets()


# Fallback model if loading fails
if model is None:
    st.error("⚠️ Using fallback model")

    model = Sequential([
        Input(shape=(22,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')
    ])


# Simple UI placeholder
st.title("CKD Diagnostic Tool")

st.write("Model is ready for predictions.")
