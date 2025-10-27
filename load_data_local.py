"""
Alternative data loader using local CSV file
Use this if UCI repository is slow or unavailable
"""

import streamlit as st
import pandas as pd
import os
from config import SESSION_DATA_KEY

def load_from_csv():
    """
    Load data from local CSV files (train.csv and unique_m.csv)
    """
    # Check if data already in session state
    if SESSION_DATA_KEY not in st.session_state:
        train_file = "train.csv"
        unique_file = "unique_m.csv"
        
        if not os.path.exists(train_file):
            st.error(f"❌ File '{train_file}' not found!")
            st.info("""
            📥 **Please ensure 'train.csv' is in the project directory:**
            - Location: d:/ML MINI PROJECT/train.csv
            """)
            st.stop()
        
        # Load train.csv
        df = pd.read_csv(train_file)
        
        # Load unique_m.csv if available (optional metadata)
        metadata_info = {'name': 'Superconductivity Data (Local)'}
        if os.path.exists(unique_file):
            unique_df = pd.read_csv(unique_file)
            metadata_info['unique_materials'] = unique_df
            st.success(f"✅ Loaded {len(unique_df)} unique materials from {unique_file}")
        
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
        
        # Create classification target (binary: high vs low critical temp)
        y_class = (y > y.median()).astype(int)
        
        # Store in session state
        st.session_state[SESSION_DATA_KEY] = {
            'X': X,
            'y': y,
            'y_class': y_class,
            'metadata': metadata_info,
            'variables': {}
        }
        
        st.success(f"✅ Loaded {len(df)} samples with {len(X.columns)} features from {train_file}")
    
    return st.session_state[SESSION_DATA_KEY]
