"""
Load pre-trained models instantly
No training in Streamlit - just loading!
"""

import streamlit as st
import pickle
import os

MODEL_DIR = 'models'

@st.cache_resource
def load_model(model_name):
    """
    Load a pre-trained model from disk
    Cached so it only loads once per session
    """
    model_files = {
        'linear': 'linear_regression.pkl',
        'polynomial': 'polynomial_regression.pkl',
        'decision_tree': 'decision_tree.pkl',
        'svm': 'svm.pkl',
        'random_forest': 'random_forest.pkl',
        'gradient_boosting': 'gradient_boosting.pkl',
        'kmeans_3': 'kmeans_3.pkl',
        'kmeans_5': 'kmeans_5.pkl',
        'dbscan': 'dbscan.pkl',
        'pca': 'pca.pkl',
        'svd': 'svd.pkl',
        'scaler': 'scaler.pkl'
    }
    
    if model_name not in model_files:
        st.error(f"❌ Unknown model: '{model_name}'")
        return None
    
    filepath = os.path.join(MODEL_DIR, model_files[model_name])
    
    if not os.path.exists(filepath):
        st.error(f"❌ Model not found: '{filepath}'")
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
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model

@st.cache_resource
def load_train_test_splits():
    """Load pre-computed train/test splits"""
    filepath = os.path.join(MODEL_DIR, 'train_test_splits.pkl')
    
    if not os.path.exists(filepath):
        st.warning("⚠️ Train/test splits not found. Run: python train_all_models.py")
        return None
    
    with open(filepath, 'rb') as f:
        splits = pickle.load(f)
    
    return splits

def check_model_exists(model_name):
    """Check if a model file exists"""
    model_files = {
        'linear': 'linear_regression.pkl',
        'polynomial': 'polynomial_regression.pkl',
        'decision_tree': 'decision_tree.pkl',
        'svm': 'svm.pkl',
        'random_forest': 'random_forest.pkl',
        'gradient_boosting': 'gradient_boosting.pkl',
        'kmeans_3': 'kmeans_3.pkl',
        'kmeans_5': 'kmeans_5.pkl',
        'dbscan': 'dbscan.pkl',
        'pca': 'pca.pkl',
        'svd': 'svd.pkl',
        'scaler': 'scaler.pkl'
    }
    
    if model_name in model_files:
        filepath = os.path.join(MODEL_DIR, model_files[model_name])
        return os.path.exists(filepath)
    return False

def list_available_models():
    """List all available pre-trained models"""
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    models = [f.replace('.pkl', '') for f in model_files]
    return models

def load_classification_labels():
    """Load classification labels"""
    filepath = os.path.join(MODEL_DIR, 'classification_labels.pkl')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError("Classification labels not found. Run 'python train_all_models.py' first.")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_model_info():
    """Get information about all available models"""
    models = {
        'Regression': ['linear', 'polynomial'],
        'Classification': ['decision_tree', 'svm', 'random_forest', 'gradient_boosting'],
        'Clustering': ['kmeans_3', 'kmeans_5', 'dbscan'],
        'Dimensionality Reduction': ['pca', 'svd']
    }
    
    info = {}
    for category, model_list in models.items():
        info[category] = {}
        for model in model_list:
            info[category][model] = check_model_exists(model)
    
    return info
