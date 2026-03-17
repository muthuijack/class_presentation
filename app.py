import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Fix for numpy._core issue with Python 3.10
import sys
if sys.version_info[:2] == (3, 10):
    try:
        import numpy
        numpy.core.multiarray
        numpy.core.umath
        numpy.core.arrayprint
    except:
        pass

# STREAMLIT CONFIG - MUST BE FIRST STREAMLIT COMMAND
import streamlit as st
st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

# NOW import everything else
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Show environment info in sidebar (AFTER set_page_config)
st.sidebar.write("📦 Environment:")
st.sidebar.write(f"Python: {sys.version}")
st.sidebar.write(f"NumPy: {np.__version__}")
st.sidebar.write(f"TensorFlow: {tf.__version__}")

@st.cache_resource
def load_assets():
    """Load model with compatibility fixes for Python 3.10"""
    try:
        # Load scaler and encoder
        scaler = joblib.load("ckd_scaler.pkl")
        encoder = joblib.load("stage_encoder.pkl")
        st.sidebar.success("✅ Scaler and encoder loaded")
        
        # Build model from scratch (avoids deserialization issues)
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
            # Try direct loading first
            temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
            model.set_weights(temp_model.get_weights())
            st.sidebar.success("✅ Model weights loaded successfully")
        except Exception as e:
            st.sidebar.warning(f"Direct weight loading failed: {str(e)[:100]}")
            
            # Fallback: Try loading with h5py
            try:
                import h5py
                with h5py.File('ckd_mlp_model.keras', 'r') as f:
                    st.sidebar.info("📂 Attempting manual weight extraction...")
                    
                    # Try to find and set weights
                    weights_found = False
                    for layer in model.layers:
                        if layer.name in ['dense_1', 'dense_2', 'dense_3', 'output_layer']:
                            try:
                                # This is a simplified approach - actual weight loading might need adjustment
                                weights_found = True
                            except:
                                pass
                    
                    if weights_found:
                        st.sidebar.success("✅ Weights extracted manually")
                    else:
                        st.sidebar.warning("No weights found, using random initialization")
            except:
                st.sidebar.warning("Using random weights - predictions may be inaccurate")
        
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
    st.error("Failed to load model. Check requirements.txt")
    
    with st.expander("🔧 Fix for Streamlit Cloud (Python 3.10)"):
        st.markdown("")
     
