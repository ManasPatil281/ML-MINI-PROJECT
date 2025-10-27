import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_loader import get_X, get_y
from model_loader import load_model, load_train_test_splits, check_model_exists

st.set_page_config(page_title="Dimensionality Reduction", page_icon="📉", layout="wide")

st.title("📉 Dimensionality Reduction")

# Check models
if not check_model_exists('pca'):
    st.error("❌ PCA/SVD models not found!")
    st.code("python train_all_models.py", language="bash")
    st.stop()

# Load data
X = get_X()
y = get_y()

# Load pre-computed data
splits = load_train_test_splits()
X_scaled = splits['X_scaled']

st.markdown("""
## Objective: Simplify High-Dimensional Data

**PCA/SVD models are PRE-TRAINED - just loading and visualizing!**
""")

st.subheader("📊 Original Data Dimensions")
col1, col2 = st.columns(2)
col1.metric("Number of Features", X.shape[1])
col2.metric("Number of Samples", X.shape[0])

# PCA
st.markdown("---")
st.subheader("1️⃣ Principal Component Analysis (PCA)")

pca = load_model('pca')

if pca:
    X_pca = pca.transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Components", pca.n_components)
    col2.metric("Variance Explained", f"{cumulative_var[-1]:.2%}")
    col3.metric("Original Features", X.shape[1])
    
    # Variance plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(explained_var)+1)), 
                        y=explained_var, name='Individual'))
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_var)+1)), 
                            y=cumulative_var, mode='lines+markers', 
                            name='Cumulative', yaxis='y2'))
    fig.update_layout(
        title='PCA Explained Variance',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio',
        yaxis2=dict(title='Cumulative Variance', overlaying='y', side='right'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2D visualization
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Critical_Temperature': y.values.flatten()
    })
    
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Critical_Temperature',
                   title='Data Projection on First Two Principal Components',
                   labels={'PC1': f'PC1 ({explained_var[0]:.1%} variance)',
                          'PC2': f'PC2 ({explained_var[1]:.1%} variance)'},
                   color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Component loadings
    st.subheader("Top Contributing Features per Component")
    for i in range(min(3, pca.n_components)):
        loadings = pca.components_[i]
        top_indices = np.argsort(np.abs(loadings))[::-1][:5]
        
        with st.expander(f"PC{i+1} - Top 5 Features"):
            for idx in top_indices:
                st.write(f"- {X.columns[idx]}: {loadings[idx]:.4f}")

# SVD
st.markdown("---")
st.subheader("2️⃣ Singular Value Decomposition (SVD)")

svd = load_model('svd')

if svd:
    X_svd = svd.transform(X_scaled)
    
    explained_var_svd = svd.explained_variance_ratio_
    cumulative_var_svd = np.cumsum(explained_var_svd)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Components", svd.n_components)
    col2.metric("Variance Explained", f"{cumulative_var_svd[-1]:.2%}")
    col3.metric("Original Features", X.shape[1])
    
    # Variance plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(explained_var_svd)+1)), 
                        y=explained_var_svd, name='Individual'))
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_var_svd)+1)), 
                            y=cumulative_var_svd, mode='lines+markers', 
                            name='Cumulative', yaxis='y2'))
    fig.update_layout(title='SVD Explained Variance')
    st.plotly_chart(fig, use_container_width=True)
    
    # 2D visualization
    df_svd = pd.DataFrame({
        'SV1': X_svd[:, 0],
        'SV2': X_svd[:, 1],
        'Critical_Temperature': y.values.flatten()
    })
    
    fig = px.scatter(df_svd, x='SV1', y='SV2', color='Critical_Temperature',
                   title='Data Projection on First Two Singular Vectors',
                   color_continuous_scale='plasma')
    st.plotly_chart(fig, use_container_width=True)

st.success("✅ PCA/SVD models loaded instantly!")
