"""
Load pre-trained models instantly
No training in Streamlit - just loading!
"""

import streamlit as st
import pickle
import os

@st.cache_resource
def load_model(model_name):
    """
    Load a pre-trained model from disk
    Cached so it only loads once per session
    """
    model_path = f"{model_name}.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model '{model_name}' not found!")
        st.info(f"""
        **Please train the models first:**
        
        Run this command:
        ```bash
        python train_all_models.py
        ```
        
        This will train all models once and save them.
        Then Streamlit will just load them instantly!
        """)
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

@st.cache_resource
def load_train_test_splits():
    """Load pre-computed train/test splits"""
    splits_path = 'train_test_splits.pkl'
    
    if not os.path.exists(splits_path):
        st.warning("⚠️ Train/test splits not found. Run: python train_all_models.py")
        return None
    
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    return splits

def check_model_exists(model_name):
    """Check if a model file exists"""
    return os.path.exists(f"{model_name}.pkl")

def list_available_models():
    """List all available pre-trained models"""
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    models = [f.replace('.pkl', '') for f in model_files]
    return models
