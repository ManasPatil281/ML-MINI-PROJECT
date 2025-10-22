import streamlit as st
import time

# Page configuration
st.set_page_config(
    page_title="Superconductivity ML Project",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("🔬 Superconductivity Prediction - Mini Project")
st.markdown("---")

# Loading indicator
with st.spinner("Loading project overview..."):
    time.sleep(2)  # Simulate loading time

st.markdown("""
## Project Overview

### Domain: **Energy & Materials Science**

Superconductivity is a phenomenon where certain materials can conduct electricity without resistance at very low temperatures. 
This project uses machine learning to predict the critical temperature of superconductors, which has significant societal impact:

### 🌍 Real-World Applications & Societal Impact

1. **Energy Efficiency**
   - Zero-resistance power transmission reduces energy loss
   - Can save billions in electricity costs globally
   - Reduces carbon footprint in power distribution

2. **Medical Technology**
   - Essential for MRI machines
   - Improved medical imaging capabilities
   - Better healthcare accessibility

3. **Transportation**
   - Maglev trains using superconducting magnets
   - Faster, cleaner public transportation
   - Reduced traffic congestion

4. **Renewable Energy**
   - Better energy storage systems
   - More efficient wind turbines
   - Enhanced solar power systems

### 📊 Dataset Information

- **Source:** UCI Machine Learning Repository
- **Samples:** 21,263 superconductors
- **Features:** 81 material properties
- **Target:** Critical Temperature (Tc) in Kelvin

### 🎯 Project Objectives

This project demonstrates various machine learning techniques:
- Data Visualization & Exploration
- Regression Models (Linear, Polynomial)
- Classification Models (Decision Trees, SVM, Ensemble)
- Clustering Algorithms (K-Means, DBSCAN)
- Dimensionality Reduction (PCA, SVD)
- MLOps Best Practices

### 📱 Navigation

Use the sidebar to navigate through different sections:
- **Home:** Project overview (current page)
- **Data Exploration:** Dataset analysis and visualizations
- **Regression Models:** Predict critical temperature
- **Classification Models:** Classify superconductor types
- **Clustering Analysis:** Group similar materials
- **Dimensionality Reduction:** Feature analysis
- **Model Comparison:** Performance metrics
- **MLOps Tools:** Tool comparison and workflow
- **Conclusion:** Key findings and recommendations
""")

st.markdown("---")
st.info("👈 Select a page from the sidebar to begin exploring!")

# Performance tips
st.markdown("---")
st.subheader("⚡ Performance Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **First Time Setup:**
    ```bash
    # Pre-train all models (run once)
    python train_all_models.py
    ```
    This takes 2-5 minutes but makes the app load instantly afterward!
    """)

with col2:
    st.markdown("""
    **Why pages load slowly:**
    - First visit downloads 21K samples from UCI
    - After that, data is cached automatically
    - Pre-trained models load in milliseconds
    - Subsequent visits are much faster ⚡
    """)

st.success("💡 **Pro Tip:** The first page visit might be slow, but navigation between pages is instant thanks to caching!")
