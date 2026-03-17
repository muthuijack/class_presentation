import os
import sys
import subprocess
import importlib

# Set environment variables first
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Force Python to use the correct NumPy path
if "numpy" in sys.modules:
    del sys.modules["numpy"]

# Import numpy in a specific way to avoid core module issues
try:
    import numpy.core.multiarray
    import numpy.core.umath
    import numpy.core.arrayprint
    import numpy as np
    print(f"✅ NumPy {np.__version__} loaded successfully")
except Exception as e:
    print(f"⚠️ NumPy import issue: {e}")
    # Fallback: try reinstalling
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.22.4"])
    import numpy as np

# STREAMLIT CONFIG - FIRST STREAMLIT COMMAND
import streamlit as st
st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

# Now import other libraries
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import warnings
warnings.filterwarnings('ignore')

# Verify imports
st.sidebar.write("📦 Environment:")
st.sidebar.write(f"Python: {sys.version}")
st.sidebar.write(f"NumPy: {np.__version__}")
st.sidebar.write(f"TensorFlow: {tf.__version__}")

@st.cache_resource
def load_assets():
    """Load model with multiple fallback methods"""
    try:
        # Load scaler and encoder first
        scaler = joblib.load("ckd_scaler.pkl")
        encoder = joblib.load("stage_encoder.pkl")
        st.sidebar.success("✅ Scaler and encoder loaded")
        
        # Try different methods to load the model
        model = None
        
        # Method 1: Build model and load weights separately
        try:
            # Define architecture
            model = Sequential([
                Input(shape=(22,), name='input_layer'),
                Dense(128, activation='relu', name='dense_1'),
                Dropout(0.3, name='dropout_1'),
                Dense(64, activation='relu', name='dense_2'),
                Dropout(0.3, name='dropout_2'),
                Dense(32, activation='relu', name='dense_3'),
                Dense(6, activation='softmax', name='output_layer')
            ])
            
            # Try to load weights using different methods
            try:
                # Method A: Direct load
                temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
                model.set_weights(temp_model.get_weights())
                st.sidebar.success("✅ Model weights loaded")
            except:
                # Method B: Load weights only
                model.load_weights("ckd_mlp_model.keras")
                st.sidebar.success("✅ Weights loaded directly")
                
        except Exception as e:
            st.sidebar.warning(f"Architecture build failed: {e}")
            
            # Method 2: Try loading with custom objects
            try:
                from tensorflow.keras.utils import custom_object_scope
                with custom_object_scope({'InputLayer': tf.keras.layers.InputLayer}):
                    model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
                st.sidebar.success("✅ Model loaded with custom objects")
            except Exception as e2:
                st.sidebar.error(f"All loading methods failed: {e2}")
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

if model is None:
    st.error("⚠️ Could not load the trained model. Using a basic model for demonstration.")
    
    # Create a simple fallback model
    model = Sequential([
        Input(shape=(22,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')
    ])
    
    with st.expander("🔧 Troubleshooting Steps"):
        st.markdown("""
        ### Fix for NumPy core module error:
        
        1. **Create/update `runtime.txt`** in your repository:
