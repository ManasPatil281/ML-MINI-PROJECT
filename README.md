# 🔬 Superconductivity Prediction - ML Mini Project

A comprehensive machine learning project demonstrating end-to-end data science workflow for predicting superconductor critical temperatures and classifying materials to accelerate discovery of energy-efficient technologies.

---

## 🎯 Project Overview

### What is Superconductivity?

Superconductivity is a quantum mechanical phenomenon where certain materials can conduct electricity with **zero electrical resistance** when cooled below a critical temperature (Tc). This breakthrough property enables:

- **Zero energy loss** in power transmission
- **Powerful electromagnets** for MRI machines and particle accelerators
- **Quantum computing** applications
- **Maglev transportation** systems

### Project Goals

This project applies **9 different machine learning algorithms** to:

1. **Predict** the critical temperature (Tc) of superconducting materials
2. **Classify** materials as HIGH-Tc or LOW-Tc superconductors
3. **Discover** hidden patterns and material families through clustering
4. **Reduce** dimensionality to identify the most important material properties
5. **Compare** model performance to select optimal algorithms

### Why This Matters

Traditional superconductor discovery involves:
- **Years** of laboratory experiments
- **$500K-$1M** in R&D costs per material
- **Trial-and-error** synthesis and testing

**Machine Learning accelerates this by:**
- Predicting properties in **seconds** instead of years
- Reducing experimental costs by **60-70%**
- Identifying promising candidates from millions of possibilities
- Guiding researchers to the most promising materials

---

## 🌍 Societal Impact

### Domain: Energy & Materials Science

**Real-World Applications:**

#### 1. **Energy Efficiency** ⚡
- **Zero-resistance power grids**: Save 5-10% of global electricity (200 TWh/year)
- **Reduced carbon emissions**: Equivalent to removing 30 million cars
- **Lower electricity costs** for consumers

#### 2. **Medical Technology** 🏥
- **MRI machines**: More powerful and efficient imaging
- **SQUID sensors**: Ultra-sensitive brain scanning
- **Better diagnostics** leading to improved patient outcomes

#### 3. **Transportation** 🚄
- **Maglev trains**: 300+ mph speeds with minimal energy
- **Electric aviation**: Lightweight motors with superconducting magnets
- **Sustainable transport** infrastructure

#### 4. **Renewable Energy** 🌱
- **Superconducting magnetic energy storage (SMES)**: Grid stabilization
- **Efficient generators**: Wind turbines with 99% efficiency
- **Fusion reactors**: Tokamak confinement magnets

#### 5. **Quantum Computing** 💻
- **Superconducting qubits**: Foundation of quantum processors
- **Cryogenic electronics**: Ultra-fast, low-power circuits
- **Next-generation computing** infrastructure

---

## 📊 Dataset Description

### Source
**UCI Machine Learning Repository** - Superconductivity Data  
**Citation:** Hamidieh, K. (2018). Superconductivty Data. UCI Machine Learning Repository.

### Dataset Statistics

| Property | Value |
|----------|-------|
| **Total Samples** | 21,263 superconducting materials |
| **Features** | 81 material properties |
| **Target Variable** | Critical Temperature (Tc) in Kelvin |
| **Tc Range** | 0.55 K to 185 K |
| **Median Tc** | ~55 K |
| **Missing Values** | None (clean dataset) |

### Feature Categories

The 81 features are derived from the chemical formula and include:

#### **Atomic Properties** (mean, weighted mean, geometric mean, entropy)
- Atomic mass
- Atomic radius  
- Density
- Electron affinity
- Ionization energy
- Electronegativity

#### **Thermal Properties**
- Fusion heat
- Thermal conductivity
- Boiling point
- Specific heat

#### **Chemical Properties**
- Number of valence electrons
- Atomic number (Z)
- Elemental composition

#### **Statistical Aggregations**
- Mean, weighted mean, geometric mean, entropy, range, standard deviation for each property

### Example Materials in Dataset
- **YBa₂Cu₃O₇** (Yttrium Barium Copper Oxide): Tc = 92 K
- **MgB₂** (Magnesium Diboride): Tc = 39 K
- **Nb₃Sn** (Niobium-Tin): Tc = 18 K
- **Fe-based superconductors**: Tc = 20-55 K

---

## 🤖 Machine Learning Algorithms (Detailed)

### 1. 📊 Data Exploration & Visualization

**Objective:** Understand data distribution, correlations, and patterns before modeling.

**Techniques Applied:**
- **Descriptive Statistics**: Mean, median, std deviation, quartiles
- **Distribution Analysis**: Histograms, box plots, density plots
- **Correlation Analysis**: Heatmaps showing feature relationships
- **Feature Engineering**: Identifying multicollinearity

**Key Insights Generated:**
- Critical temperature distribution is **right-skewed** (most materials < 50K)
- **High correlation** between atomic mass and density (r > 0.7)
- **81 features have significant multicollinearity** → justifies PCA
- Outliers exist at Tc > 130K (high-temperature superconductors)

**Visualizations Provided:**
- Tc distribution histogram
- Correlation heatmap (81×81 matrix)
- Scatter plots for top features vs Tc
- Box plots for feature distributions

---

### 2. 📈 Regression Models

#### 2.1 Linear Regression

**Algorithm:** Ordinary Least Squares (OLS)

**Mathematical Formulation:**
```
Tc = b0 + b1 * feature1 + b2 * feature2 + ... + bn * featuren
```

**Model Training:**
- **Train-test split**: 80% train, 20% test
- **Fitting**: OLS estimates coefficients by minimizing squared errors

**Model Evaluation:**
- **R² Score**: 0.73 (73% of variance explained)
- **Adjusted R²**: Penalizes for excessive features, more reliable for model comparison

**Residual Analysis:**
- Residuals plotted against fitted values
- No patterns → linear model is appropriate

**Visualizations:**
- Scatter plot of predicted vs actual Tc values
- Residuals histogram (should be normal)

---

#### 2.2 Polynomial Regression

**Algorithm:** Extends linear regression by adding polynomial terms

**Mathematical Formulation:**
```
Tc = b0 + b1 * feature1 + b2 * feature1^2 + ... + bn * featuren^k
```

**Model Training:**
- **Feature Engineering**: Generate polynomial features (degree 2 and 3)
- **Train-test split**: 80% train, 20% test
- **Fitting**: OLS on polynomial features

**Model Evaluation:**
- **R² Score**: 0.92 (92% of variance explained)
- **Adjusted R²**: Higher than linear regression, justifies added complexity

**Residual Analysis:**
- Residuals plotted against fitted values
- No patterns → polynomial model is appropriate

**Visualizations:**
- Scatter plot of predicted vs actual Tc values
- Residuals histogram (should be normal)
- Polynomial feature importance (higher degree features may be less important)

---

### 3. 📉 Model Comparison

**Objective:** Compare performance of different algorithms to select the best one for each task.

**Regression Models Compared:**
- Linear Regression
- Polynomial Regression (degree 2 and 3)

**Classification Models Compared:**
- Decision Trees
- Support Vector Machines (SVM)
- Random Forest
- Gradient Boosting

**Metrics Used:**
- **Regression**: R² Score, Adjusted R², RMSE (Root Mean Squared Error)
- **Classification**: Accuracy, Precision, Recall, F1 Score, ROC-AUC

**Results Summary:**
- **Best Regression Model**: Polynomial Regression (degree 3) - R² = 0.92
- **Best Classification Model**: Gradient Boosting - 98.1% accuracy

**Visualizations:**
- Bar charts of model performance metrics
- ROC curves for classification models
- Feature importance for top models

---

## 📈 Results & Insights

| Task | Best Model | Performance | Use Case |
|------|-----------|-------------|----------|
| Temperature Prediction | Polynomial Regression | R² = 0.92 | Accurate predictions |
| Material Classification | Gradient Boosting | 98.1% accuracy | HIGH/LOW-Tc classification |
| Material Grouping | K-Means | Silhouette = 0.342 | Family discovery |
| Feature Reduction | PCA | 85-95% variance | Faster models |

---

## 💡 Key Findings

1. **Polynomial features significantly improve prediction accuracy** (R² from 0.73 to 0.92)
2. **Ensemble methods outperform single models** (GB: 98.1% vs DT: 92.1%)
3. **81 features can be reduced to 10-15** while retaining 85-95% of information
4. **3-5 distinct material families** emerge from clustering analysis
5. **Chemical composition features** are most important for prediction

---

## ⚡ Usage Guide

### Running the Streamlit App

1. **Start the app**:
    ```bash
    streamlit run app.py
    ```
2. **Open in browser**: Go to `http://localhost:8501`
3. **Select task**: Choose between regression, classification, clustering, or dimensionality reduction
4. **Upload data**: Provide a CSV file with material features
5. **View results**: Explore predictions, visualizations, and download options

### Example Workflow

1. **Predict Tc of new materials**:
    - Select "Regression" task
    - Upload CSV with material features
    - Download results with predicted Tc values

2. **Classify materials into HIGH-Tc/LOW-Tc**:
    - Select "Classification" task
    - Upload CSV with material features
    - View and download classification report

3. **Discover material families**:
    - Select "Clustering" task
    - Upload CSV with material features
    - View clustering results and download data

4. **Reduce dimensionality of feature space**:
    - Select "Dimensionality Reduction" task
    - Upload CSV with material features
    - Download transformed features

---

## 🔮 Future Enhancements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Add real-time data integration
- [ ] Deploy on cloud (AWS/Azure)
- [ ] Implement active learning loop
- [ ] Add quantum mechanical features
- [ ] Create API for lab integration

---

## 📝 License

This project is created for educational purposes as part of a mini project requirement.

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn community for excellent ML tools
- Streamlit team for the amazing framework
- Research community working on superconductivity

---

**Made with ❤️ for advancing materials science and sustainable energy solutions**
