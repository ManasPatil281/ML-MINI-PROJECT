import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_loader import get_X, get_y, clear_data_cache

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

st.title("📊 Data Exploration & Visualization")

# Sidebar cache management
if st.sidebar.button("🗑️ Clear Data Cache"):
    clear_data_cache()
    st.rerun()

# Load data using session state (instant after first load!)
X = get_X()
y = get_y()

st.success("✅ Dataset ready!")

st.markdown("""
## Understanding the Dataset

**Superconductivity** is crucial for energy efficiency. By analyzing material properties, 
we can discover new superconductors that work at higher temperatures, making them more practical 
and cost-effective for real-world applications.
""")

# Dataset Overview
st.subheader("📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", f"{X.shape[0]:,}")
col2.metric("Features", X.shape[1])
col3.metric("Min Tc (K)", f"{y.min().values[0]:.2f}")
col4.metric("Max Tc (K)", f"{y.max().values[0]:.2f}")

# Basic Statistics
st.subheader("📈 Target Variable Statistics")
col1, col2 = st.columns(2)

with col1:
    st.write("**Critical Temperature Statistics:**")
    st.dataframe(y.describe())

with col2:
    fig = px.histogram(y, x=y.columns[0], nbins=50, 
                       title="Distribution of Critical Temperature",
                       labels={y.columns[0]: "Critical Temperature (K)", "count": "Frequency"})
    st.plotly_chart(fig, use_container_width=True)

# Feature Analysis
st.subheader("🔍 Feature Analysis")

selected_features = st.multiselect(
    "Select features to analyze:",
    X.columns.tolist(),
    default=X.columns[:5].tolist()
)

if selected_features:
    st.write("**Selected Features Statistics:**")
    st.dataframe(X[selected_features].describe())
    
    # Correlation with target - FIXED VERSION
    correlations = []
    y_series = y.iloc[:, 0]  # Convert DataFrame to Series
    
    for feat in selected_features:
        corr = X[feat].corr(y_series)
        correlations.append({"Feature": feat, "Correlation with Tc": corr})
    
    corr_df = pd.DataFrame(correlations).sort_values("Correlation with Tc", ascending=False)
    
    fig = px.bar(corr_df, x="Feature", y="Correlation with Tc",
                 title="Feature Correlation with Critical Temperature")
    st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
st.subheader("🌡️ Feature Correlation Heatmap")
num_features = st.slider("Number of features to show in heatmap:", 5, 20, 10)
top_features = X.columns[:num_features]
corr_matrix = X[top_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
plt.title("Feature Correlation Matrix")
st.pyplot(fig)

# Scatter plots
st.subheader("📍 Feature vs Target Scatter Plots")
feature_to_plot = st.selectbox("Select feature for scatter plot:", X.columns.tolist())

fig = px.scatter(x=X[feature_to_plot], y=y.iloc[:, 0],
                 labels={'x': feature_to_plot, 'y': 'Critical Temperature (K)'},
                 title=f'{feature_to_plot} vs Critical Temperature',
                 opacity=0.5)
st.plotly_chart(fig, use_container_width=True)

# Box plots
st.subheader("📦 Feature Distribution (Box Plots)")
if selected_features:
    fig, ax = plt.subplots(figsize=(12, 6))
    X[selected_features].boxplot(ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.title("Feature Distributions")
    st.pyplot(fig)

# Key Insights
st.markdown("---")
st.subheader("💡 Key Insights")
st.markdown("""
1. **Critical Temperature Range:** The dataset contains superconductors with Tc ranging from near 0K to over 180K
2. **Feature Diversity:** 81 different material properties are used to characterize superconductors
3. **Correlation Analysis:** Some features show strong correlation with critical temperature
4. **Practical Impact:** Higher Tc materials are more valuable as they require less cooling, reducing energy costs
5. **Material Discovery:** Understanding these patterns helps scientists design new superconductors
""")
