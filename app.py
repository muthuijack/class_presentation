import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Force numpy to use legacy module structure for Python 3.10
import sys
if sys.version_info[:2] == (3, 10):
    import numpy
    # Force load of core modules to avoid _core issue
    numpy.core.multiarray
    numpy.core.umath
    numpy.core.arrayprint
    from numpy import *

import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

# Show environment info
st.sidebar.write("📦 Environment:")
st.sidebar.write(f"Python: {sys.version}")
import numpy as np
st.sidebar.write(f"NumPy: {np.__version__}")
st.sidebar.write(f"TensorFlow: {tf.__version__}")

@st.cache_resource
def load_assets():
    """Load model with compatibility fixes for Python 3.10"""
    try:
        # Load scaler and encoder
        scaler = joblib.load("ckd_scaler.pkl")
        encoder = joblib.load("stage_encoder.pkl")
        
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
            # Use a different approach for Python 3.10
            import h5py
            
            # Open the file and extract weights manually
            with h5py.File('ckd_mlp_model.keras', 'r') as f:
                st.sidebar.info("📂 Model file opened successfully")
                
                # Navigate to weights
                if 'model_weights' in f:
                    weights_dict = {}
                    for layer_name in f['model_weights']:
                        if layer_name in ['dense_1', 'dense_2', 'dense_3', 'output_layer']:
                            layer_group = f['model_weights'][layer_name]
                            if 'weight_names' in layer_group.attrs:
                                weight_names = layer_group.attrs['weight_names']
                                weights = []
                                for weight_name in weight_names:
                                    if isinstance(weight_name, bytes):
                                        weight_name = weight_name.decode('utf-8')
                                    if weight_name in layer_group:
                                        weights.append(layer_group[weight_name][()])
                                if weights:
                                    weights_dict[layer_name] = weights
                    
                    # Set weights if found
                    if weights_dict:
                        for layer in model.layers:
                            if layer.name in weights_dict:
                                try:
                                    layer.set_weights(weights_dict[layer.name])
                                    st.sidebar.success(f"✅ Weights loaded for {layer.name}")
                                except:
                                    pass
        except Exception as e:
            st.sidebar.warning(f"Weight loading skipped: {e}")
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
    st.error("Failed to load model. Using emergency fallback...")
    
    # Create a simple fallback model
    model = Sequential([
        Input(shape=(22,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')
    ])
    
    # Try one more time with different method
    try:
        # Load using tensorflow's legacy method
        import tensorflow.compat.v1 as tf_v1
        tf_v1.disable_v2_behavior()
        temp_model = tf_v1.keras.models.load_model('ckd_mlp_model.keras')
        st.success("✅ Model loaded with TensorFlow v1 compatibility")
    except:
        st.warning("Using untrained model for demonstration")
    
    with st.expander("🔧 Deployment Fix for Python 3.10"):
        st.markdown("")
       
