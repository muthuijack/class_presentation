import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Fix for numpy.core.multiarray issue
import sys
import subprocess

# Ensure numpy is properly loaded before anything else
try:
    import numpy as np
    # Force load of core modules
    np.core.multiarray
    np.core.umath
    np.core.arrayprint
except (ImportError, AttributeError) as e:
    if 'multiarray' in str(e):
        # Reinstall numpy if needed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.24.3"])
        import numpy as np
    else:
        raise e

# STREAMLIT CONFIG - MUST BE FIRST STREAMLIT COMMAND
import streamlit as st
st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

# NOW import everything else
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import warnings
warnings.filterwarnings('ignore')

# Show environment info
st.sidebar.write("📦 Environment:")
st.sidebar.write(f"Python: {sys.version}")
st.sidebar.write(f"NumPy: {np.__version__}")
st.sidebar.write(f"TensorFlow: {tf.__version__}")

@st.cache_resource
def load_assets():
    """Load model with compatibility fixes"""
    try:
        # Load scaler and encoder
        scaler = joblib.load("ckd_scaler.pkl")
        encoder = joblib.load("stage_encoder.pkl")
        st.sidebar.success("✅ Scaler and encoder loaded")
        
        # Try multiple methods to load model
        model = None
        
        # Method 1: Try direct loading first
        try:
            model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
            st.sidebar.success("✅ Model loaded directly")
            return model, scaler, encoder
        except Exception as e:
            st.sidebar.warning(f"Direct loading failed: {str(e)[:100]}")
        
        # Method 2: Build from scratch
        st.sidebar.info("Building model from scratch...")
        model = Sequential([
            Input(shape=(22,), name='input_layer'),
            Dense(128, activation='relu', name='dense_1'),
            Dropout(0.3, name='dropout_1'),
            Dense(64, activation='relu', name='dense_2'),
            Dropout(0.3, name='dropout_2'),
            Dense(32, activation='relu', name='dense_3'),
            Dense(6, activation='softmax', name='output_layer')
        ])
        
        # Try to load weights
        try:
            # Create a temporary model just to get weights
            temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
            model.set_weights(temp_model.get_weights())
            st.sidebar.success("✅ Weights loaded into skeleton")
        except Exception as e:
            st.sidebar.warning(f"Weight loading failed: {e}")
            st.sidebar.info("Using random weights - predictions may be inaccurate")
        
        return model, scaler, encoder
        
    except Exception as e:
        st.error(f"❌ Loading Error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

# Load assets
with st.spinner("Loading CKD prediction model..."):
    model, scaler, encoder = load_assets()

if model is None:
    st.error("⚠️ Failed to load model. Using emergency fallback.")
    
    # Create a simple fallback model
    model = Sequential([
        Input(shape=(22,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')
    ])
    
    with st.expander("🔧 Fix for numpy.core.multiarray issue"):
        st.markdown("")
      
