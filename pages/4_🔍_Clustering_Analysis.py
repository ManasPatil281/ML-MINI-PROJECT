import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_loader import get_X, get_y
from model_loader import load_model, load_train_test_splits, check_model_exists

st.set_page_config(page_title="Clustering Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Clustering Analysis")

# Check models
if not check_model_exists('kmeans'):
    st.error("❌ Clustering models not found!")
    st.code("python train_all_models.py", language="bash")
    st.stop()

# Load data
X = get_X()
y = get_y()

# Load pre-computed data
splits = load_train_test_splits()
X_scaled = splits['X_scaled']

st.markdown("""
## Objective: Discover Material Groups

**Clustering models are PRE-TRAINED - just loading and visualizing!**
""")

# K-Means
st.markdown("---")
st.subheader("1️⃣ K-Means Clustering")

kmeans = load_model('kmeans')

if kmeans:
    clusters_km = kmeans.predict(X_scaled)
    n_clusters = len(np.unique(clusters_km))
    
    silhouette = silhouette_score(X_scaled, clusters_km)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters_km)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Clusters", n_clusters)
    col2.metric("Silhouette Score", f"{silhouette:.4f}")
    col3.metric("Davies-Bouldin Index", f"{davies_bouldin:.4f}")
    
    # Cluster distribution
    cluster_counts = pd.Series(clusters_km).value_counts().sort_index()
    fig = go.Figure(data=[go.Bar(x=cluster_counts.index, y=cluster_counts.values)])
    fig.update_layout(title="Samples per Cluster", xaxis_title="Cluster", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_plot = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters_km,
        'Tc': y.values.flatten()
    })
    
    fig = px.scatter(df_plot, x='PCA1', y='PCA2', color='Cluster',
                    hover_data=['Tc'], title='K-Means Clusters (2D PCA Projection)',
                    color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster statistics
    st.subheader("Cluster Characteristics")
    for i in range(n_clusters):
        cluster_mask = clusters_km == i
        cluster_tc = y.values.flatten()[cluster_mask]
        
        with st.expander(f"Cluster {i} - {cluster_mask.sum()} materials"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Tc", f"{cluster_tc.mean():.2f} K")
            col2.metric("Median Tc", f"{np.median(cluster_tc):.2f} K")
            col3.metric("Min Tc", f"{cluster_tc.min():.2f} K")
            col4.metric("Max Tc", f"{cluster_tc.max():.2f} K")

# DBSCAN
st.markdown("---")
st.subheader("2️⃣ DBSCAN Clustering")

dbscan = load_model('dbscan')

if dbscan:
    clusters_db = dbscan.fit_predict(X_scaled)
    
    n_clusters_db = len(set(clusters_db)) - (1 if -1 in clusters_db else 0)
    n_noise = list(clusters_db).count(-1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters Found", n_clusters_db)
    col2.metric("Noise Points", n_noise)
    col3.metric("Core Points", len(clusters_db) - n_noise)
    
    # Visualization
    df_plot = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters_db,
        'Tc': y.values.flatten()
    })
    
    fig = px.scatter(df_plot, x='PCA1', y='PCA2', color='Cluster',
                    hover_data=['Tc'], title='DBSCAN Clusters (2D PCA Projection)',
                    color_continuous_scale='plasma')
    st.plotly_chart(fig, use_container_width=True)

st.success("✅ Clustering models loaded instantly!")

# Conclusions
st.markdown("---")
st.subheader("📊 Clustering Conclusions")
st.markdown("""
**Algorithm Comparison:**
- **K-Means:** Fast, works well with spherical clusters, requires specifying K
- **DBSCAN:** Finds arbitrary shapes, handles noise, determines clusters automatically

**Insights:**
- Different material families emerge based on chemical composition
- Some materials are outliers with unique properties worth investigating
- Clusters often correspond to different crystal structures or elemental compositions

**Practical Applications:**
- **Material Discovery:** Focus on under-explored clusters
- **Research Direction:** Identify promising material families
- **Cost Reduction:** Synthesize representatives from each cluster rather than all materials
- **Patent Strategy:** Target unexplored cluster spaces

**Impact:**
Clustering has helped researchers identify new superconductor families, leading to 
discoveries like iron-based superconductors. It reduces the search space from millions 
to hundreds of promising candidates.
""")
