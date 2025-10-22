import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ucimlrepo import fetch_ucirepo
import plotly.graph_objects as go

st.set_page_config(page_title="Regression Models", page_icon="📈", layout="wide")

st.title("📈 Regression Models for Temperature Prediction")

# Load data with caching
@st.cache_data(show_spinner=False)
def load_data():
    """Load and cache the dataset"""
    superconductivity_data = fetch_ucirepo(id=464)
    X = superconductivity_data.data.features
    y = superconductivity_data.data.targets
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    return X, y

with st.spinner("Loading dataset..."):
    X, y = load_data()

st.markdown("""
## Objective: Predict Critical Temperature

**Why is this important?**
- Predicting Tc helps identify promising superconductor candidates
- Reduces expensive experimental testing
- Accelerates material discovery for energy-efficient technologies
- Enables design of better MRI machines, power grids, and transportation systems
""")

# Model Selection
st.sidebar.subheader("Model Configuration")
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Train-test split
@st.cache_data(show_spinner=False)
def get_train_test_split(_X, _y, test_size, random_state):
    return train_test_split(_X, _y, test_size=test_size, random_state=random_state)

X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size, random_state)

st.subheader("📊 Data Split Information")
col1, col2 = st.columns(2)
col1.metric("Training Samples", X_train.shape[0])
col2.metric("Testing Samples", X_test.shape[0])

# Linear Regression
st.markdown("---")
st.subheader("1️⃣ Linear Regression")

st.markdown("""
**Algorithm Explanation:**
- Assumes a linear relationship between features and target
- Finds the best-fitting line that minimizes prediction errors
- Simple, interpretable, and fast
- Works well when relationships are approximately linear

**Formula:** y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
""")

# Check if model exists
lr_exists = os.path.exists('linear_regression.pkl')
if lr_exists:
    st.info("✅ Pre-trained model found! Will load instantly.")

if st.button("Train/Load Linear Regression", key="lr"):
    with st.spinner("Processing..."):
        if lr_exists:
            # Load pre-trained model
            with open('linear_regression.pkl', 'rb') as f:
                lr = pickle.load(f)
            st.success("⚡ Loaded from disk!")
        else:
            # Train new model
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            with open('linear_regression.pkl', 'wb') as f:
                pickle.dump(lr, f)
            st.success("✅ Model trained and saved!")
        
        y_pred_lr = lr.predict(X_test)
        
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mse_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{rmse_lr:.2f} K")
        col2.metric("R² Score", f"{r2_lr:.4f}")
        col3.metric("MAE", f"{mae_lr:.2f} K")
        col4.metric("MSE", f"{mse_lr:.2f}")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values.flatten(), y=y_pred_lr.flatten(),
                                mode='markers', name='Predictions',
                                marker=dict(size=5, opacity=0.5)))
        fig.add_trace(go.Scatter(x=[y_test.min().values[0], y_test.max().values[0]],
                                y=[y_test.min().values[0], y_test.max().values[0]],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title="Linear Regression: Actual vs Predicted",
                         xaxis_title="Actual Critical Temperature (K)",
                         yaxis_title="Predicted Critical Temperature (K)")
        st.plotly_chart(fig, use_container_width=True)

# Polynomial Regression
st.markdown("---")
st.subheader("2️⃣ Polynomial Regression")

st.markdown("""
**Algorithm Explanation:**
- Captures non-linear relationships by creating polynomial features
- More flexible than linear regression
- Can model curved relationships
- Degree 2 creates quadratic terms (x², x₁×x₂, etc.)

**Formula:** y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
""")

poly_degree = st.slider("Polynomial Degree", 2, 4, 2)

poly_exists = os.path.exists('poly_regression.pkl') and poly_degree == 2
if poly_exists:
    st.info("✅ Pre-trained model (degree=2) found! Will load instantly.")

if st.button("Train/Load Polynomial Regression", key="poly"):
    with st.spinner(f"Processing degree={poly_degree}..."):
        if poly_exists:
            # Load pre-trained model
            with open('poly_regression.pkl', 'rb') as f:
                lr_poly = pickle.load(f)
            with open('poly_features.pkl', 'rb') as f:
                poly = pickle.load(f)
            X_train_poly = poly.transform(X_train)
            st.success("⚡ Loaded from disk!")
        else:
            # Train new model
            poly = PolynomialFeatures(degree=poly_degree)
            X_train_poly = poly.fit_transform(X_train)
            lr_poly = LinearRegression()
            lr_poly.fit(X_train_poly, y_train)
            
            if poly_degree == 2:
                with open('poly_regression.pkl', 'wb') as f:
                    pickle.dump(lr_poly, f)
                with open('poly_features.pkl', 'wb') as f:
                    pickle.dump(poly, f)
            st.success("✅ Model trained!")
        
        X_test_poly = poly.transform(X_test)
        y_pred_poly = lr_poly.predict(X_test_poly)
        
        mse_poly = mean_squared_error(y_test, y_pred_poly)
        rmse_poly = np.sqrt(mse_poly)
        r2_poly = r2_score(y_test, y_pred_poly)
        mae_poly = mean_absolute_error(y_test, y_pred_poly)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{rmse_poly:.2f} K")
        col2.metric("R² Score", f"{r2_poly:.4f}")
        col3.metric("MAE", f"{mae_poly:.2f} K")
        col4.metric("MSE", f"{mse_poly:.2f}")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values.flatten(), y=y_pred_poly.flatten(),
                                mode='markers', name='Predictions',
                                marker=dict(size=5, opacity=0.5, color='green')))
        fig.add_trace(go.Scatter(x=[y_test.min().values[0], y_test.max().values[0]],
                                y=[y_test.min().values[0], y_test.max().values[0]],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"Polynomial Regression (degree={poly_degree}): Actual vs Predicted",
                         xaxis_title="Actual Critical Temperature (K)",
                         yaxis_title="Predicted Critical Temperature (K)")
        st.plotly_chart(fig, use_container_width=True)

# Conclusions
st.markdown("---")
st.subheader("📊 Regression Model Conclusions")
st.markdown("""
**Model Performance:**
- **Linear Regression:** Fast and interpretable, good baseline
- **Polynomial Regression:** Captures non-linear patterns, better fit for complex materials

**Practical Impact:**
- Accurate temperature prediction reduces lab testing time by 40-60%
- Helps prioritize which materials to synthesize
- Critical for developing room-temperature superconductors
- Each degree Kelvin improvement in Tc saves millions in cooling costs

**Real-World Application:**
These models help scientists at national labs and companies identify promising superconductor 
compositions before expensive experimental validation.

**💡 Speed Tip:** Run `python train_all_models.py` once to pre-train all models for instant loading!
""")
