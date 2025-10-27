"""
Pre-train all models to speed up Streamlit app loading
Run this script ONCE: python train_all_models.py
Then Streamlit will just LOAD the models (instant!)
"""

# Make sure all imports are at the top
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    RandomForestRegressor, 
    GradientBoostingRegressor
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neural_network import MLPRegressor

print("=" * 60)
print("TRAINING ALL MODELS - RUN THIS ONCE!")
print("=" * 60)

# Load data from train.csv
print("\n1. Loading data from train.csv...")
if not os.path.exists('train.csv'):
    print("❌ ERROR: train.csv not found!")
    print("Please ensure train.csv is in the project directory")
    exit(1)

df = pd.read_csv('train.csv')
print(f"✅ Loaded {len(df):,} samples")

# Separate features and target
if 'critical_temp' not in df.columns:
    print("❌ ERROR: 'critical_temp' column not found!")
    print("Available columns:", df.columns.tolist())
    exit(1)

X = df.drop('critical_temp', axis=1)
y = df[['critical_temp']]
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Create classification target
y_class = (y > y.median()).astype(int)

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]:,}")

# Split data
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Scale data for clustering and neural networks
print("\n3. Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.fit_transform(X_train)

# Dictionary to store all models
models = {}

# REGRESSION MODELS
print("\n" + "=" * 60)
print("TRAINING REGRESSION MODELS")
print("=" * 60)

print("\n4. Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
models['linear_regression'] = lr
print("✅ Linear Regression trained")

print("\n5. Training Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
models['ridge_regression'] = ridge
print("✅ Ridge Regression trained")

print("\n6. Training Polynomial Regression (degree=2)...")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y_train)
models['poly_regression'] = lr_poly
models['poly_features'] = poly
print("✅ Polynomial Regression trained")

print("\n7. Training Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train.values.ravel())
models['rf_regression'] = rf_reg
print("✅ Random Forest Regressor trained")

print("\n8. Training Gradient Boosting Regressor...")
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42)
gb_reg.fit(X_train, y_train.values.ravel())
models['gb_regression'] = gb_reg
print("✅ Gradient Boosting Regressor trained")

print("\n9. Training Neural Network (MLP) Regressor...")
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
mlp.fit(X_train_scaled, y_train.values.ravel())
models['mlp_regression'] = mlp
models['scaler'] = scaler
print("✅ Neural Network trained")

# CLASSIFICATION MODELS
print("\n" + "=" * 60)
print("TRAINING CLASSIFICATION MODELS")
print("=" * 60)

print("\n10. Training Decision Tree Classifier...")
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_c, y_train_c)
models['decision_tree'] = dt
print("✅ Decision Tree trained")

print("\n11. Training SVM Classifier (this may take time)...")
svm = SVC(random_state=42, kernel='rbf')
svm.fit(X_train_c, y_train_c.values.ravel())
models['svm'] = svm
print("✅ SVM trained")

print("\n12. Training Random Forest Classifier...")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_c, y_train_c.values.ravel())
models['random_forest'] = rf
print("✅ Random Forest Classifier trained")

print("\n13. Training Gradient Boosting Classifier...")
gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb.fit(X_train_c, y_train_c.values.ravel())
models['gradient_boosting'] = gb
print("✅ Gradient Boosting Classifier trained")

# CLUSTERING MODELS
print("\n" + "=" * 60)
print("TRAINING CLUSTERING MODELS")
print("=" * 60)

print("\n14. Training K-Means Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
models['kmeans'] = kmeans
print("✅ K-Means trained")

print("\n15. Training DBSCAN Clustering...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
models['dbscan'] = dbscan
print("✅ DBSCAN trained")

# DIMENSIONALITY REDUCTION
print("\n" + "=" * 60)
print("TRAINING DIMENSIONALITY REDUCTION MODELS")
print("=" * 60)

print("\n16. Training PCA...")
pca = PCA(n_components=10)
pca.fit(X_scaled)
models['pca'] = pca
print("✅ PCA trained")

print("\n17. Training SVD...")
svd = TruncatedSVD(n_components=10, random_state=42)
svd.fit(X_scaled)
models['svd'] = svd
print("✅ SVD trained")

# SAVE ALL MODELS
print("\n" + "=" * 60)
print("SAVING ALL MODELS")
print("=" * 60)

for name, model in models.items():
    filename = f'{name}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved {filename}")

# Save train/test splits for consistent evaluation
print("\n18. Saving train/test splits...")
splits = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'X_train_c': X_train_c,
    'X_test_c': X_test_c,
    'y_train_c': y_train_c,
    'y_test_c': y_test_c,
    'X_scaled': X_scaled
}
with open('train_test_splits.pkl', 'wb') as f:
    pickle.dump(splits, f)
print("✅ Saved train_test_splits.pkl")

print("\n" + "=" * 60)
print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 60)
print("\nYou can now run: streamlit run app.py")
print("All models will LOAD INSTANTLY (no retraining needed)")
print("=" * 60)
