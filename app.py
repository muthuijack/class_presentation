```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Handle numpy import carefully for Streamlit Cloud
try:
    import numpy as np
except ImportError as e:
    if "_core" in str(e):
        # Fallback for numpy 2.0+ compatibility issue
        import numpy.core.multiarray
        import numpy.core.umath
        import numpy.core._multiarray_umath
        import numpy as np
    else:
        raise e

import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CKD Diagnostic Tool", layout="wide")

# Show environment info for debugging
st.sidebar.write("📦 Environment:")
st.sidebar.write(f"Python: {os.sys.version}")
st.sidebar.write(f"NumPy: {np.__version__}")
st.sidebar.write(f"TensorFlow: {tf.__version__}")

@st.cache_resource
def load_assets():
    """Load model with compatibility fixes"""
    try:
        # Load scaler and encoder
        scaler = joblib.load("ckd_scaler.pkl")
        encoder = joblib.load("stage_encoder.pkl")
        
        # Try multiple methods to load model
        model = None
        
        # Method 1: Direct load with compile=False
        try:
            model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
            st.sidebar.success("✅ Model loaded directly")
        except Exception as e:
            st.sidebar.warning(f"Direct load failed: {str(e)[:50]}")
            
            # Method 2: Build from scratch
            try:
                # Create model architecture
                model = Sequential([
                    Input(shape=(22,), name='input_layer'),
                    Dense(128, activation='relu'),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(6, activation='softmax')
                ])
                
                # Try to load weights
                try:
                    temp_model = tf.keras.models.load_model("ckd_mlp_model.keras", compile=False)
                    model.set_weights(temp_model.get_weights())
                    st.sidebar.success("✅ Weights loaded into skeleton")
                except:
                    st.sidebar.warning("Using random weights")
                    
            except Exception as e:
                st.sidebar.error(f"Skeleton build failed: {e}")
                return None, None, None
        
        return model, scaler, encoder
        
    except Exception as e:
        st.error(f"❌ Loading Error: {e}")
        return None, None, None

# Load assets
with st.spinner("Loading CKD prediction model..."):
    model, scaler, encoder = load_assets()

if model is None:
    st.error("Failed to load model. Check requirements.txt")
    
    # Show deployment instructions
    with st.expander("🔧 Fix for Streamlit Cloud"):
        st.markdown("""
        ### Update your GitHub repository:
        
        1. **Create/Update `requirements.txt`**:
        ```txt
        numpy==1.23.5
        tensorflow==2.13.0
        streamlit==1.28.0
        joblib==1.2.0
        scikit-learn==1.3.0
        protobuf==3.20.3
