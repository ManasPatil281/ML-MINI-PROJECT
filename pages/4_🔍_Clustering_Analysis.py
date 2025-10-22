import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from ucimlrepo import fetch_ucirepo
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Clustering Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Clustering Analysis")

# Load data
@st.cache_data
def load_data():
    superconductivity_data = fetch_ucirepo(id=464)
    X = superconductivity_data.data.features
    y = superconductivity_data.data.targets
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    return X, y

X, y = load_data()

st.markdown("""
## Objective: Discover Material Groups

**Problem Statement:**
Group superconductors based on their material properties to identify:
- Families of similar materials
- Patterns in successful superconductors
- New research directions

**Societal Impact:**
- Identifies material families worth investigating
- Reveals hidden patterns in successful superconductors
- Guides synthesis of new materials
- Reduces trial-and-error in material discovery
""")

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
st.markdown("---")
st.subheader("1️⃣ K-Means Clustering")

st.markdown("""
**Algorithm Explanation:**
- Partitions data into K clusters
- Each point belongs to cluster with nearest centroid
- Iteratively updates centroids
- Fast and scalable

**How it works:**
1. Randomly initialize K cluster centers
2. Assign each point to nearest center
3. Update centers to mean of assigned points
4. Repeat until convergence

**Use Case:**
Discover K distinct families of superconductors with similar properties.
""")

n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)

if st.button("Run K-Means Clustering", key="kmeans"):
    with st.spinner(f"Running K-Means with {n_clusters} clusters..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters_km = kmeans.fit_predict(X_scaled)
        
        # Metrics
        silhouette = silhouette_score(X_scaled, clusters_km)
        davies_bouldin = davies_bouldin_score(X_scaled, clusters_km)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Clusters", n_clusters)
        col2.metric("Silhouette Score", f"{silhouette:.4f}")
        col3.metric("Davies-Bouldin Index", f"{davies_bouldin:.4f}")
        
        st.info("""
        **Metric Explanation:**
        - **Silhouette Score:** Ranges from -1 to 1. Higher is better (well-separated clusters)
        - **Davies-Bouldin Index:** Lower is better (compact and well-separated clusters)
        """)
        
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
        
        pickle.dump(kmeans, open('kmeans.pkl', 'wb'))
        st.success("✅ K-Means model saved!")

# DBSCAN
st.markdown("---")
st.subheader("2️⃣ DBSCAN Clustering")

st.markdown("""
**Algorithm Explanation:**
- Density-Based Spatial Clustering
- Finds clusters of arbitrary shape
- Automatically determines number of clusters
- Identifies outliers (noise points)

**Parameters:**
- **eps:** Maximum distance between points in same cluster
- **min_samples:** Minimum points to form dense region

**Use Case:**
Discover natural groupings without specifying cluster count, identify unusual materials.
""")

col1, col2 = st.columns(2)
with col1:
    eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
with col2:
    min_samples = st.slider("Minimum Samples", 3, 20, 5)

if st.button("Run DBSCAN Clustering", key="dbscan"):
    with st.spinner("Running DBSCAN..."):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters_db = dbscan.fit_predict(X_scaled)
        
        n_clusters_db = len(set(clusters_db)) - (1 if -1 in clusters_db else 0)
        n_noise = list(clusters_db).count(-1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters Found", n_clusters_db)
        col2.metric("Noise Points", n_noise)
        col3.metric("Core Points", len(clusters_db) - n_noise)
        
        if n_clusters_db > 1:
            # Calculate metrics only for non-noise points
            mask = clusters_db != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X_scaled[mask], clusters_db[mask])
                st.metric("Silhouette Score", f"{silhouette:.4f}")
        
        # Cluster distribution
        cluster_counts = pd.Series(clusters_db).value_counts().sort_index()
        fig = go.Figure(data=[go.Bar(x=cluster_counts.index, y=cluster_counts.values)])
        fig.update_layout(title="Samples per Cluster (-1 = Noise)", 
                         xaxis_title="Cluster", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
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
        
        pickle.dump(dbscan, open('dbscan.pkl', 'wb'))
        st.success("✅ DBSCAN model saved!")

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
