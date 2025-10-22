import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dimensionality Reduction", page_icon="📉", layout="wide")

st.title("📉 Dimensionality Reduction")

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
## Objective: Simplify High-Dimensional Data

**Why Reduce Dimensions?**
- Original dataset has 81 features (dimensions)
- Hard to visualize and interpret
- Some features may be redundant
- Reduces computational cost
- Helps avoid curse of dimensionality

**Societal Impact:**
- Makes ML models faster and more efficient
- Helps scientists understand which properties truly matter
- Reduces data storage requirements
- Enables real-time predictions
""")

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("📊 Original Data Dimensions")
col1, col2 = st.columns(2)
col1.metric("Number of Features", X.shape[1])
col2.metric("Number of Samples", X.shape[0])

# PCA
st.markdown("---")
st.subheader("1️⃣ Principal Component Analysis (PCA)")

st.markdown("""
**Algorithm Explanation:**
- Finds directions (principal components) of maximum variance
- Projects data onto these directions
- First component captures most variance, second captures second-most, etc.
- Components are orthogonal (uncorrelated)

**Mathematical Concept:**
PCA performs eigenvalue decomposition on the covariance matrix:
- Eigenvectors = Principal Components (directions)
- Eigenvalues = Amount of variance explained

**Analogy:**
Imagine viewing a 3D object from different angles. PCA finds the "best angles" 
that show the most variation in the object's shape.
""")

n_components_pca = st.slider("Number of PCA Components", 2, 20, 10)

if st.button("Run PCA", key="pca"):
    with st.spinner("Performing PCA..."):
        pca = PCA(n_components=n_components_pca)
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Components", n_components_pca)
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
        
        st.info(f"""
        **Interpretation:**
        - First {n_components_pca} components explain {cumulative_var[-1]:.2%} of total variance
        - This means we can represent {X.shape[1]} features with just {n_components_pca} components
        - Data compression ratio: {X.shape[1]/n_components_pca:.1f}x
        """)
        
        # 2D visualization
        if n_components_pca >= 2:
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
        
        # Component loadings (top features)
        st.subheader("Top Contributing Features per Component")
        for i in range(min(3, n_components_pca)):
            loadings = pca.components_[i]
            top_indices = np.argsort(np.abs(loadings))[::-1][:5]
            
            with st.expander(f"PC{i+1} - Top 5 Features"):
                for idx in top_indices:
                    st.write(f"- {X.columns[idx]}: {loadings[idx]:.4f}")
        
        pickle.dump(pca, open('pca.pkl', 'wb'))
        st.success("✅ PCA model saved!")

# SVD
st.markdown("---")
st.subheader("2️⃣ Singular Value Decomposition (SVD)")

st.markdown("""
**Algorithm Explanation:**
- Factorizes data matrix into three matrices: U, Σ, V^T
- Similar to PCA but works directly on data matrix
- More numerically stable for sparse matrices
- Commonly used in recommendation systems

**Relationship to PCA:**
SVD is the mathematical operation behind PCA. When applied to centered data:
- Right singular vectors = Principal Components
- Singular values = Square root of eigenvalues

**Use Case:**
Alternative to PCA, especially useful for large sparse matrices.
""")

n_components_svd = st.slider("Number of SVD Components", 2, 20, 10)

if st.button("Run SVD", key="svd"):
    with st.spinner("Performing SVD..."):
        svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
        X_svd = svd.fit_transform(X_scaled)
        
        explained_var_svd = svd.explained_variance_ratio_
        cumulative_var_svd = np.cumsum(explained_var_svd)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Components", n_components_svd)
        col2.metric("Variance Explained", f"{cumulative_var_svd[-1]:.2%}")
        col3.metric("Original Features", X.shape[1])
        
        # Variance plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, len(explained_var_svd)+1)), 
                            y=explained_var_svd, name='Individual'))
        fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_var_svd)+1)), 
                                y=cumulative_var_svd, mode='lines+markers', 
                                name='Cumulative', yaxis='y2'))
        fig.update_layout(
            title='SVD Explained Variance',
            xaxis_title='Component',
            yaxis_title='Explained Variance Ratio',
            yaxis2=dict(title='Cumulative Variance', overlaying='y', side='right')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2D visualization
        if n_components_svd >= 2:
            df_svd = pd.DataFrame({
                'SV1': X_svd[:, 0],
                'SV2': X_svd[:, 1],
                'Critical_Temperature': y.values.flatten()
            })
            
            fig = px.scatter(df_svd, x='SV1', y='SV2', color='Critical_Temperature',
                           title='Data Projection on First Two Singular Vectors',
                           color_continuous_scale='plasma')
            st.plotly_chart(fig, use_container_width=True)
        
        pickle.dump(svd, open('svd.pkl', 'wb'))
        st.success("✅ SVD model saved!")

# Comparison
st.markdown("---")
st.subheader("🔄 PCA vs SVD Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **PCA Advantages:**
    - Intuitive interpretation (variance maximization)
    - Produces uncorrelated components
    - Standard in exploratory data analysis
    - Better for small to medium datasets
    """)

with col2:
    st.markdown("""
    **SVD Advantages:**
    - More numerically stable
    - Works well with sparse matrices
    - Used in Netflix Prize challenge
    - Efficient for very large datasets
    """)

# Conclusions
st.markdown("---")
st.subheader("📊 Dimensionality Reduction Conclusions")
st.markdown("""
**Key Findings:**
- 81 features can be effectively reduced to 10-20 components
- First few components capture majority of variance
- Reveals which properties are most important for superconductivity

**Practical Benefits:**
1. **Model Speed:** Reduces training time by 5-10x
2. **Visualization:** Makes high-dimensional data understandable
3. **Storage:** Compresses data significantly
4. **Overfitting:** Reduces risk with fewer features
5. **Interpretation:** Identifies key material properties

**Real-World Impact:**
- Scientists can focus on most important material properties
- Faster screening of new materials
- Better understanding of superconductivity mechanisms
- Enables real-time predictions in lab settings

**Example:**
Instead of measuring all 81 properties, researchers might only need to measure 
10-15 key properties to predict superconductor performance, saving time and resources.
""")
