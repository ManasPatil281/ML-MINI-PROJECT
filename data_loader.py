"""
Centralized data loader with session state caching
Supports: 1) Local CSV files (fastest), 2) data.pkl, 3) UCI repository
"""

import streamlit as st
import pandas as pd
import pickle
import os
from config import SESSION_DATA_KEY, DATASET_ID, TRAIN_CSV_PATH, UNIQUE_CSV_PATH

def load_data_once():
    """
    Load data only once per session using st.session_state
    Priority: 1) Local CSV (train.csv), 2) data.pkl, 3) UCI repository
    """
    # Check if data already in session state
    if SESSION_DATA_KEY not in st.session_state:
        try:
            # Priority 1: Load from local CSV files (FASTEST!)
            if os.path.exists(TRAIN_CSV_PATH):
                st.info(f"⚡ Loading from {TRAIN_CSV_PATH}...")
                
                df = pd.read_csv(TRAIN_CSV_PATH)
                
                # Load unique materials if available
                metadata_info = {'name': 'Superconductivity Data (Local CSV)'}
                if os.path.exists(UNIQUE_CSV_PATH):
                    unique_df = pd.read_csv(UNIQUE_CSV_PATH)
                    metadata_info['unique_materials'] = unique_df
                
                # Separate features and target
                if 'critical_temp' not in df.columns:
                    st.error("❌ 'critical_temp' column not found in train.csv!")
                    st.write("Available columns:", df.columns.tolist())
                    st.stop()
                    
                X = df.drop('critical_temp', axis=1)
                y = df[['critical_temp']]
                
                # Preprocess
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())
                
                # Create classification target
                y_class = (y > y.median()).astype(int)
                
                # Store in session state
                st.session_state[SESSION_DATA_KEY] = {
                    'X': X,
                    'y': y,
                    'y_class': y_class,
                    'metadata': metadata_info,
                    'variables': {},
                    'source': f'{TRAIN_CSV_PATH} (local CSV)'
                }
                
                st.success(f"✅ Loaded {len(df):,} samples, {len(X.columns)} features from CSV")
            
            # Priority 2: Load from pickle file (INSTANT!)
            elif os.path.exists('data.pkl'):
                st.info("⚡ Loading from data.pkl...")
                
                with open('data.pkl', 'rb') as f:
                    saved_data = pickle.load(f)
                
                X = saved_data.data.features
                y = saved_data.data.targets
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())
                
                y_class = (y > y.median()).astype(int)
                
                st.session_state[SESSION_DATA_KEY] = {
                    'X': X,
                    'y': y,
                    'y_class': y_class,
                    'metadata': saved_data.metadata,
                    'variables': saved_data.variables,
                    'source': 'data.pkl (cached)'
                }
                
                st.success(f"✅ Loaded {len(X):,} samples from data.pkl")
            
            # Priority 3: Fallback to UCI repository (SLOW)
            else:
                st.warning("⏳ Loading from UCI Repository (first time only)...")
                
           
                
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("""
            **Troubleshooting:**
            1. Ensure `train.csv` is in: d:/ML MINI PROJECT/
            2. Or ensure `data.pkl` exists
            3. Check internet connection for UCI download
            """)
            st.stop()
    
    return st.session_state[SESSION_DATA_KEY]

def get_X():
    """Get features DataFrame"""
    data = load_data_once()
    return data['X']

def get_y():
    """Get target DataFrame"""
    data = load_data_once()
    return data['y']

def get_y_class():
    """Get classification target DataFrame"""
    data = load_data_once()
    return data['y_class']

def get_metadata():
    """Get dataset metadata"""
    data = load_data_once()
    return data.get('metadata', {})

def get_data_source():
    """Get data source information"""
    data = load_data_once()
    return data.get('source', 'Unknown')

def clear_data_cache():
    """Clear data from session state"""
    if SESSION_DATA_KEY in st.session_state:
        del st.session_state[SESSION_DATA_KEY]
        st.success("✅ Data cache cleared!")
