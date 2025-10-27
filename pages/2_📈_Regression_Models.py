import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_loader import get_X, get_y, clear_data_cache
from model_loader import load_model, load_train_test_splits, check_model_exists

st.set_page_config(page_title="Regression Models", page_icon="📈", layout="wide")

st.title("📈 Regression Models for Temperature Prediction")

# Check if models are trained
st.sidebar.subheader("⚙️ Model Status")
models_trained = check_model_exists('linear_regression')

if not models_trained:
    st.error("❌ Models not found! Please train models first.")
    st.code("python train_all_models.py", language="bash")
    st.stop()
else:
    st.sidebar.success("✅ Models are ready!")

# Load data
X = get_X()
y = get_y()

# Load pre-computed splits
splits = load_train_test_splits()
if splits is None:
    st.error("Please run: python train_all_models.py")
    st.stop()

X_train = splits['X_train']
X_test = splits['X_test']
y_train = splits['y_train']
y_test = splits['y_test']

st.markdown("""
## Objective: Predict Critical Temperature

**All models are PRE-TRAINED - just loading and visualizing!**
""")

# Data info
st.subheader("📊 Data Split Information")
col1, col2 = st.columns(2)
col1.metric("Training Samples", X_train.shape[0])
col2.metric("Testing Samples", X_test.shape[0])

# Tabs for different models
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Linear Models", 
    "2️⃣ Polynomial", 
    "3️⃣ Ensemble (Best!)", 
    "4️⃣ Neural Network",
    "5️⃣ Predict"
])

# Tab 1: Linear Models
with tab1:
    st.markdown("### Linear Regression & Ridge")
    
    # Load models (instant!)
    lr = load_model('linear_regression')
    ridge = load_model('ridge_regression')
    
    if lr and ridge:
        # Get predictions
        y_pred_lr = lr.predict(X_test)
        y_pred_ridge = ridge.predict(X_test)
        
        # Show results
        results = pd.DataFrame({
            'Model': ['Linear Regression', 'Ridge Regression'],
            'RMSE': [
                np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                np.sqrt(mean_squared_error(y_test, y_pred_ridge))
            ],
            'R²': [
                r2_score(y_test, y_pred_lr),
                r2_score(y_test, y_pred_ridge)
            ],
            'MAE': [
                mean_absolute_error(y_test, y_pred_lr),
                mean_absolute_error(y_test, y_pred_ridge)
            ]
        })
        
        st.dataframe(results, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values.flatten(), y=y_pred_lr.flatten(),
                                mode='markers', name='Linear', opacity=0.5))
        fig.add_trace(go.Scatter(x=y_test.values.flatten(), y=y_pred_ridge.flatten(),
                                mode='markers', name='Ridge', opacity=0.5))
        fig.add_trace(go.Scatter(x=[y_test.min().values[0], y_test.max().values[0]],
                                y=[y_test.min().values[0], y_test.max().values[0]],
                                mode='lines', name='Perfect', line=dict(dash='dash')))
        fig.update_layout(title="Linear Models: Actual vs Predicted",
                         xaxis_title="Actual Tc (K)", yaxis_title="Predicted Tc (K)")
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Polynomial
with tab2:
    st.markdown("### Polynomial Regression (degree=2)")
    
    lr_poly = load_model('poly_regression')
    poly = load_model('poly_features')
    
    if lr_poly and poly:
        X_test_poly = poly.transform(X_test)
        y_pred_poly = lr_poly.predict(X_test_poly)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_poly)):.2f} K")
        col2.metric("R²", f"{r2_score(y_test, y_pred_poly):.4f}")
        col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred_poly):.2f} K")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values.flatten(), y=y_pred_poly.flatten(),
                                mode='markers', name='Predictions', opacity=0.5))
        fig.add_trace(go.Scatter(x=[y_test.min().values[0], y_test.max().values[0]],
                                y=[y_test.min().values[0], y_test.max().values[0]],
                                mode='lines', name='Perfect', line=dict(dash='dash')))
        fig.update_layout(title="Polynomial Regression: Actual vs Predicted",
                         xaxis_title="Actual Tc (K)", yaxis_title="Predicted Tc (K)")
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Ensemble Methods
with tab3:
    st.markdown("### 🏆 Ensemble Methods (Best Performance!)")
    
    rf = load_model('rf_regression')
    gb = load_model('gb_regression')
    
    if rf and gb:
        y_pred_rf = rf.predict(X_test)
        y_pred_gb = gb.predict(X_test)
        
        results = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting'],
            'RMSE': [
                np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                np.sqrt(mean_squared_error(y_test, y_pred_gb))
            ],
            'R²': [
                r2_score(y_test, y_pred_rf),
                r2_score(y_test, y_pred_gb)
            ],
            'MAE': [
                mean_absolute_error(y_test, y_pred_rf),
                mean_absolute_error(y_test, y_pred_gb)
            ]
        })
        
        st.dataframe(results, use_container_width=True)
        
        # Feature importance
        st.subheader("Top 15 Important Features")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=X.columns[indices], y=importances[indices]))
        fig.update_layout(title="Feature Importances (Random Forest)",
                         xaxis_title="Features", yaxis_title="Importance",
                         xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Neural Network
with tab4:
    st.markdown("### Neural Network (MLP)")
    
    mlp = load_model('mlp_regression')
    scaler = load_model('scaler')
    
    if mlp and scaler:
        X_test_scaled = scaler.transform(X_test)
        y_pred_mlp = mlp.predict(X_test_scaled)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_mlp)):.2f} K")
        col2.metric("R²", f"{r2_score(y_test, y_pred_mlp):.4f}")
        col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred_mlp):.2f} K")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values.flatten(), y=y_pred_mlp.flatten(),
                                mode='markers', name='Predictions', opacity=0.5))
        fig.add_trace(go.Scatter(x=[y_test.min().values[0], y_test.max().values[0]],
                                y=[y_test.min().values[0], y_test.max().values[0]],
                                mode='lines', name='Perfect', line=dict(dash='dash')))
        fig.update_layout(title="Neural Network: Actual vs Predicted",
                         xaxis_title="Actual Tc (K)", yaxis_title="Predicted Tc (K)")
        st.plotly_chart(fig, use_container_width=True)

# Tab 5: Custom Prediction
with tab5:
    st.markdown("""
    ### 🔮 Custom Input Prediction
    
    Choose a model and input custom feature values to predict critical temperature.
    """)
    
    model_choice = st.selectbox("Select Model", 
                                ["Random Forest", "Gradient Boosting", "Polynomial", "Linear Regression", "Ridge Regression", "Neural Network"])
    
    # Sample values from dataset
    col1, col2 = st.columns(2)
    use_sample = col1.checkbox("Use sample from dataset")
    sample_idx = col2.number_input("Sample Index", 0, len(X)-1, 0) if use_sample else None
    
    st.subheader("Input Feature Values")
    
    # Analyze features to determine which should be dropdowns
    def get_input_widget(feature_name, feature_data, default_value):
        """
        Smart input widget selection:
        - Dropdown for features with few unique values (< 10)
        - Number input for continuous features
        """
        unique_values = feature_data.dropna().unique()
        n_unique = len(unique_values)
        
        # If feature has 10 or fewer unique values, use dropdown
        if n_unique <= 10 and n_unique > 0:
            sorted_values = sorted(unique_values)
            # Find closest value to default
            closest_idx = min(range(len(sorted_values)), 
                            key=lambda i: abs(sorted_values[i] - default_value))
            
            return st.selectbox(
                f"{feature_name} (discrete)",
                options=sorted_values,
                index=closest_idx,
                key=f"input_{feature_name}"
            )
        else:
            # Continuous feature - use number input
            return st.number_input(
                f"{feature_name}",
                value=float(default_value),
                key=f"input_{feature_name}",
                format="%.4f"
            )
    
    # Select top 10 important features for easier input
    top_features = X.columns[:10].tolist()
    
    input_method = st.radio("Input Method", ["Top 10 Features (Recommended)", "All Features (Advanced)"])
    
    if input_method == "Top 10 Features (Recommended)":
        st.info("💡 **Tip:** Features with few unique values show as dropdowns. Others are number inputs.")
        
        input_values = {}
        cols = st.columns(2)
        
        for i, feat in enumerate(top_features):
            default_val = X.iloc[sample_idx][feat] if use_sample else X[feat].median()
            
            with cols[i % 2]:
                input_values[feat] = get_input_widget(feat, X[feat], default_val)
        
        if st.button("🔮 Predict Temperature", type="primary"):
            # Create full input array
            input_array = np.zeros((1, X.shape[1]))
            for i, feat in enumerate(X.columns):
                if feat in input_values:
                    input_array[0, i] = input_values[feat]
                else:
                    input_array[0, i] = X[feat].median()
            
            # Load appropriate model and predict
            try:
                prediction = None
                model_name = ""
                
                if model_choice == "Random Forest":
                    rf = load_model('rf_regression')
                    if rf:
                        prediction = rf.predict(input_array)[0]
                        model_name = "Random Forest"
                
                elif model_choice == "Gradient Boosting":
                    gb = load_model('gb_regression')
                    if gb:
                        prediction = gb.predict(input_array)[0]
                        model_name = "Gradient Boosting"
                
                elif model_choice == "Polynomial":
                    lr_poly = load_model('poly_regression')
                    poly = load_model('poly_features')
                    if lr_poly and poly:
                        input_poly = poly.transform(input_array)
                        prediction = lr_poly.predict(input_poly)[0, 0]
                        model_name = "Polynomial Regression"
                
                elif model_choice == "Linear Regression":
                    lr = load_model('linear_regression')
                    if lr:
                        prediction = lr.predict(input_array)[0, 0]
                        model_name = "Linear Regression"
                
                elif model_choice == "Ridge Regression":
                    ridge = load_model('ridge_regression')
                    if ridge:
                        prediction = ridge.predict(input_array)[0, 0]
                        model_name = "Ridge Regression"
                
                elif model_choice == "Neural Network":
                    mlp = load_model('mlp_regression')
                    scaler = load_model('scaler')
                    if mlp and scaler:
                        input_scaled = scaler.transform(input_array)
                        prediction = mlp.predict(input_scaled)[0]
                        model_name = "Neural Network"
                
                if prediction is not None:
                    st.success(f"### 🎯 Predicted Critical Temperature: {prediction:.2f} K")
                    st.caption(f"Model used: {model_name}")
                    
                    # Classification
                    median_tc = y.median().values[0]
                    class_label = "HIGH-Tc ⬆️" if prediction > median_tc else "LOW-Tc ⬇️"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Classification:** {class_label}")
                        st.caption(f"Median Tc: {median_tc:.2f} K")
                    
                    # Compare with actual if using sample
                    if use_sample:
                        with col2:
                            actual = y.iloc[sample_idx].values[0]
                            error = abs(prediction - actual)
                            error_pct = (error / actual) * 100
                            st.metric("Actual Temperature", f"{actual:.2f} K", 
                                     delta=f"-{error:.2f} K ({error_pct:.1f}% error)")
                    
                    # Show input summary
                    with st.expander("📋 Input Summary"):
                        input_df = pd.DataFrame({
                            'Feature': list(input_values.keys()),
                            'Value': list(input_values.values())
                        })
                        st.dataframe(input_df, use_container_width=True)
                else:
                    st.error(f"❌ Failed to load {model_choice} model!")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)
    
    else:  # All Features (Advanced)
        st.warning("⚠️ Advanced mode: All 81 features shown")
        
        # Create tabs for feature groups
        n_features_per_tab = 20
        n_tabs = (len(X.columns) + n_features_per_tab - 1) // n_features_per_tab
        
        tab_names = [f"Features {i*n_features_per_tab+1}-{min((i+1)*n_features_per_tab, len(X.columns))}" 
                     for i in range(n_tabs)]
        feature_tabs = st.tabs(tab_names)
        
        all_inputs = {}
        
        for tab_idx, tab in enumerate(feature_tabs):
            with tab:
                start_idx = tab_idx * n_features_per_tab
                end_idx = min((tab_idx + 1) * n_features_per_tab, len(X.columns))
                features_in_tab = X.columns[start_idx:end_idx]
                
                cols = st.columns(2)
                for i, feat in enumerate(features_in_tab):
                    default_val = X.iloc[sample_idx][feat] if use_sample else X[feat].median()
                    
                    with cols[i % 2]:
                        all_inputs[feat] = get_input_widget(feat, X[feat], default_val)
        
        if st.button("🔮 Predict with All Features", type="primary"):
            input_array = np.array([[all_inputs[feat] for feat in X.columns]])
            
            # Similar prediction logic
            prediction = None
            model_name = ""
            
            if model_choice == "Random Forest":
                rf = load_model('rf_regression')
                if rf:
                    prediction = rf.predict(input_array)[0]
                    model_name = "Random Forest"
            
            elif model_choice == "Gradient Boosting":
                gb = load_model('gb_regression')
                if gb:
                    prediction = gb.predict(input_array)[0]
                    model_name = "Gradient Boosting"
            
            elif model_choice == "Polynomial":
                lr_poly = load_model('poly_regression')
                poly = load_model('poly_features')
                if lr_poly and poly:
                    input_poly = poly.transform(input_array)
                    prediction = lr_poly.predict(input_poly)[0, 0]
                    model_name = "Polynomial Regression"
            
            elif model_choice == "Linear Regression":
                lr = load_model('linear_regression')
                if lr:
                    prediction = lr.predict(input_array)[0, 0]
                    model_name = "Linear Regression"
            
            elif model_choice == "Ridge Regression":
                ridge = load_model('ridge_regression')
                if ridge:
                    prediction = ridge.predict(input_array)[0, 0]
                    model_name = "Ridge Regression"
            
            elif model_choice == "Neural Network":
                mlp = load_model('mlp_regression')
                scaler = load_model('scaler')
                if mlp and scaler:
                    input_scaled = scaler.transform(input_array)
                    prediction = mlp.predict(input_scaled)[0]
                    model_name = "Neural Network"
            
            if prediction is not None:
                st.success(f"### 🎯 Predicted Critical Temperature: {prediction:.2f} K")
                st.caption(f"Model used: {model_name}")
                
                median_tc = y.median().values[0]
                class_label = "HIGH-Tc ⬆️" if prediction > median_tc else "LOW-Tc ⬇️"
                st.info(f"**Classification:** {class_label} (Median: {median_tc:.2f} K)")

# Conclusions
st.markdown("---")
st.subheader("📊 Model Recommendations")

st.success("""
### 🏆 Best for Nonlinear Multivariate Regression:

**1. Random Forest (Recommended)**
- ✅ Fast training (2-5 seconds)
- ✅ Excellent accuracy (R² > 0.90)
- ✅ Naturally handles nonlinear relationships
- ✅ No scaling required
- ✅ Feature importance insights

**2. Gradient Boosting (Highest Accuracy)**
- ✅ Best R² scores
- ⚠️ Slower training (10-20 seconds)
- ✅ Sequential learning

**3. Polynomial Regression (Traditional)**
- ✅ Interpretable
- ⚠️ Memory intensive for high degrees
- ✅ Works well for degree 2-3

**4. Neural Network (Flexible)**
- ⚠️ Requires scaling
- ⚠️ Slower training
- ✅ Most flexible for complex patterns

**Production Recommendation:** Use **Random Forest** for best balance of speed and accuracy!
""")

st.success("✅ All models loaded instantly - no training needed!")
