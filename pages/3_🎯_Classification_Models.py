import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_loader import get_X, get_y_class
from model_loader import load_model, load_train_test_splits, check_model_exists

st.set_page_config(page_title="Classification Models", page_icon="🎯", layout="wide")

st.title("🎯 Classification Models")

# Check if models are trained
st.sidebar.subheader("⚙️ Model Status")
if not check_model_exists('decision_tree'):
    st.error("❌ Models not found! Please train models first.")
    st.code("python train_all_models.py", language="bash")
    st.stop()
else:
    st.sidebar.success("✅ All models ready!")

# Load data
X = get_X()
y_class = get_y_class()

# Load pre-computed splits
splits = load_train_test_splits()
X_train_c = splits['X_train_c']
X_test_c = splits['X_test_c']
y_train_c = splits['y_train_c']
y_test_c = splits['y_test_c']

st.markdown("""
## Objective: Classify Superconductor Types

**All models are PRE-TRAINED - just loading and visualizing results!**

Classification: HIGH-Tc (above median) vs LOW-Tc (below median)
""")

st.subheader("📊 Class Distribution")
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(y_class))
col2.metric("HIGH-Tc (Class 1)", int(y_class.sum().values[0]))
col3.metric("LOW-Tc (Class 0)", len(y_class) - int(y_class.sum().values[0]))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Decision Tree",
    "2️⃣ SVM", 
    "3️⃣ Ensemble",
    "4️⃣ Predict"
])

# Tab 1: Decision Tree
with tab1:
    st.markdown("### Decision Tree Classifier")
    
    dt = load_model('decision_tree')
    
    if dt:
        y_pred_dt = dt.predict(X_test_c)
        acc_dt = accuracy_score(y_test_c, y_pred_dt)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{acc_dt:.4f}")
            st.text("Classification Report:")
            report = classification_report(y_test_c, y_pred_dt, target_names=['LOW-Tc', 'HIGH-Tc'])
            st.text(report)
        
        with col2:
            cm = confusion_matrix(y_test_c, y_pred_dt)
            fig = ff.create_annotated_heatmap(cm, x=['LOW-Tc', 'HIGH-Tc'], y=['LOW-Tc', 'HIGH-Tc'],
                                             colorscale='Blues')
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tree visualization
        from sklearn.tree import plot_tree
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(dt, filled=True, feature_names=X.columns, class_names=['LOW-Tc', 'HIGH-Tc'],
                 max_depth=3, ax=ax, fontsize=10)
        plt.title("Decision Tree Structure (Max Depth 3 shown)")
        st.pyplot(fig)

# Tab 2: SVM
with tab2:
    st.markdown("### Support Vector Machine")
    
    svm = load_model('svm')
    
    if svm:
        y_pred_svm = svm.predict(X_test_c)
        acc_svm = accuracy_score(y_test_c, y_pred_svm)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{acc_svm:.4f}")
            st.text("Classification Report:")
            report = classification_report(y_test_c, y_pred_svm, target_names=['LOW-Tc', 'HIGH-Tc'])
            st.text(report)
        
        with col2:
            cm = confusion_matrix(y_test_c, y_pred_svm)
            fig = ff.create_annotated_heatmap(cm, x=['LOW-Tc', 'HIGH-Tc'], y=['LOW-Tc', 'HIGH-Tc'],
                                             colorscale='Viridis')
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
        
        # SVM decision boundary on 2D PCA
        st.subheader("Decision Boundary Visualization")
        
        sample_size = min(5000, len(X_train_c))
        sample_indices = np.random.choice(len(X_train_c), sample_size, replace=False)
        X_train_sample = X_train_c.iloc[sample_indices]
        y_train_sample = y_train_c.iloc[sample_indices]
        
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_sample)
        
        from sklearn.svm import SVC
        svm_pca = SVC(random_state=42, kernel='rbf')
        svm_pca.fit(X_train_pca, y_train_sample.values.ravel())
        
        x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
        y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
        
        x_grid = np.linspace(x_min, x_max, 200)
        y_grid = np.linspace(y_min, y_max, 200)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=2)
        
        y_train_array = y_train_sample.values.ravel()
        scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_array, 
                           cmap='RdYlBu', edgecolors='black', s=30, alpha=0.8, linewidth=0.5)
        ax.set_title('SVM Decision Boundary (2D PCA Projection)', fontsize=14, fontweight='bold')
        ax.set_xlabel('First Principal Component', fontsize=12)
        ax.set_ylabel('Second Principal Component', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class (0=LOW-Tc, 1=HIGH-Tc)', fontsize=10)
        
        st.pyplot(fig)

# Tab 3: Ensemble
with tab3:
    st.markdown("### 🏆 Ensemble Learning")
    
    rf = load_model('random_forest')
    gb = load_model('gradient_boosting')
    
    if rf and gb:
        y_pred_rf = rf.predict(X_test_c)
        y_pred_gb = gb.predict(X_test_c)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Random Forest")
            acc_rf = accuracy_score(y_test_c, y_pred_rf)
            st.metric("Accuracy", f"{acc_rf:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test_c, y_pred_rf, target_names=['LOW-Tc', 'HIGH-Tc']))
        
        with col2:
            st.subheader("Gradient Boosting")
            acc_gb = accuracy_score(y_test_c, y_pred_gb)
            st.metric("Accuracy", f"{acc_gb:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test_c, y_pred_gb, target_names=['LOW-Tc', 'HIGH-Tc']))
        
        # Feature Importance
        st.subheader("Top 15 Feature Importances")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=X.columns[indices], y=importances[indices]))
        fig.update_layout(title="Feature Importances (Random Forest)",
                         xaxis_title="Features", yaxis_title="Importance",
                         xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Custom Prediction
with tab4:
    st.markdown("### 🔮 Custom Input Classification")
    
    selected_features = st.multiselect(
        "Select features to input:",
        X.columns.tolist(),
        default=X.columns[:5].tolist()
    )
    
    input_values = {}
    for feat in selected_features:
        input_values[feat] = st.number_input(f"Value for {feat}", value=0.0, key=f"class_{feat}")
    
    if st.button("Classify Material"):
        if selected_features:
            input_array = np.zeros((1, X.shape[1]))
            for i, feat in enumerate(X.columns):
                if feat in input_values:
                    input_array[0, i] = input_values[feat]
                else:
                    input_array[0, i] = X[feat].mean()
            
            gb = load_model('gradient_boosting')
            if gb:
                prediction = gb.predict(input_array)[0]
                class_name = "HIGH-Tc" if prediction == 1 else "LOW-Tc"
                st.success(f"Predicted Class: {class_name} (Class {prediction})")

st.success("✅ All models loaded instantly - no training needed!")
