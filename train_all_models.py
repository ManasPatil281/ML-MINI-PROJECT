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
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING ALL MODELS FOR SUPERCONDUCTIVITY PREDICTION")
print("=" * 60)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('train.csv')
print(f"   ✓ Loaded {len(df)} samples with {len(df.columns)} features")

# Prepare features and target
X = df.drop('critical_temp', axis=1)
y = df['critical_temp']

# Train-test split
print("\n2. Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   ✓ Training set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")

# Save splits
os.makedirs('models', exist_ok=True)
with open('models/train_test_splits.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, f)
print("   ✓ Saved train-test splits")

# Feature Scaling
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Scaler trained and saved")

# =========================================================================
# REGRESSION MODELS
# =========================================================================
print("\n" + "=" * 60)
print("TRAINING REGRESSION MODELS")
print("=" * 60)

# Linear Regression
print("\n4. Training Linear Regression...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
with open('models/linear_regression.pkl', 'wb') as f:
    pickle.dump(linear_model, f)
train_score = linear_model.score(X_train, y_train)
test_score = linear_model.score(X_test, y_test)
print(f"   ✓ R² Score (Train): {train_score:.3f}")
print(f"   ✓ R² Score (Test): {test_score:.3f}")

# Ridge Regression
print("\n5. Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
with open('models/ridge_regression.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)
train_score = ridge_model.score(X_train, y_train)
test_score = ridge_model.score(X_test, y_test)
print(f"   ✓ R² Score (Train): {train_score:.3f}")
print(f"   ✓ R² Score (Test): {test_score:.3f}")

# Polynomial Regression
print("\n6. Training Polynomial Regression (degree=2)...")
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train, y_train)
with open('models/polynomial_regression.pkl', 'wb') as f:
    pickle.dump(poly_model, f)
train_score = poly_model.score(X_train, y_train)
test_score = poly_model.score(X_test, y_test)
print(f"   ✓ R² Score (Train): {train_score:.3f}")
print(f"   ✓ R² Score (Test): {test_score:.3f}")

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

# Create binary classification target (high temp vs low temp)
y_class_train = (y_train > y_train.median()).astype(int)
y_class_test = (y_test > y_test.median()).astype(int)

# Decision Tree
print("\n10. Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_scaled, y_class_train)
with open('models/decision_tree.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
train_acc = dt_model.score(X_train_scaled, y_class_train)
test_acc = dt_model.score(X_test_scaled, y_class_test)
print(f"   ✓ Accuracy (Train): {train_acc:.3f}")
print(f"   ✓ Accuracy (Test): {test_acc:.3f}")

# SVM
print("\n11. Training SVM Classifier...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_class_train)
with open('models/svm.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
train_acc = svm_model.score(X_train_scaled, y_class_train)
test_acc = svm_model.score(X_test_scaled, y_class_test)
print(f"   ✓ Accuracy (Train): {train_acc:.3f}")
print(f"   ✓ Accuracy (Test): {test_acc:.3f}")

# Random Forest
print("\n12. Training Random Forest Classifier...")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_scaled, y_class_train)
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
train_acc = rf_model.score(X_train_scaled, y_class_train)
test_acc = rf_model.score(X_test_scaled, y_class_test)
print(f"   ✓ Accuracy (Train): {train_acc:.3f}")
print(f"   ✓ Accuracy (Test): {test_acc:.3f}")

# Gradient Boosting
print("\n13. Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_class_train)
with open('models/gradient_boosting.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
train_acc = gb_model.score(X_train_scaled, y_class_train)
test_acc = gb_model.score(X_test_scaled, y_class_test)
print(f"   ✓ Accuracy (Train): {train_acc:.3f}")
print(f"   ✓ Accuracy (Test): {test_acc:.3f}")

# =========================================================================
# CLUSTERING MODELS
# =========================================================================
print("\n" + "=" * 60)
print("TRAINING CLUSTERING MODELS")
print("=" * 60)

# K-Means (k=3)
print("\n14. Training K-Means Clustering (k=3)...")
kmeans_3 = KMeans(n_clusters=3, random_state=42)
kmeans_3.fit(X_train_scaled)
with open('models/kmeans_3.pkl', 'wb') as f:
    pickle.dump(kmeans_3, f)
print(f"   ✓ Trained with {kmeans_3.n_clusters} clusters")

# K-Means (k=5)
print("\n15. Training K-Means Clustering (k=5)...")
kmeans_5 = KMeans(n_clusters=5, random_state=42)
kmeans_5.fit(X_train_scaled)
with open('models/kmeans_5.pkl', 'wb') as f:
    pickle.dump(kmeans_5, f)
print(f"   ✓ Trained with {kmeans_5.n_clusters} clusters")

# DBSCAN
print("\n16. Training DBSCAN Clustering...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_train_scaled)
with open('models/dbscan.pkl', 'wb') as f:
    pickle.dump(dbscan, f)
n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
n_noise = list(dbscan.labels_).count(-1)
print(f"   ✓ Found {n_clusters} clusters")
print(f"   ✓ Noise points: {n_noise}")

# =========================================================================
# DIMENSIONALITY REDUCTION
# =========================================================================
print("\n" + "=" * 60)
print("TRAINING DIMENSIONALITY REDUCTION")
print("=" * 60)

# PCA
print("\n17. Training PCA (10 components)...")
pca = PCA(n_components=10, random_state=42)
pca.fit(X_train_scaled)
with open('models/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
variance_ratio = pca.explained_variance_ratio_.sum()
print(f"   ✓ Variance retained: {variance_ratio:.3%}")

# SVD
print("\n18. Training SVD (10 components)...")
svd = TruncatedSVD(n_components=10, random_state=42)
svd.fit(X_train_scaled)
with open('models/svd.pkl', 'wb') as f:
    pickle.dump(svd, f)
variance_ratio = svd.explained_variance_ratio_.sum()
print(f"   ✓ Variance retained: {variance_ratio:.3%}")

# Save classification labels
with open('models/classification_labels.pkl', 'wb') as f:
    pickle.dump({
        'y_class_train': y_class_train,
        'y_class_test': y_class_test
    }, f)

print("\n" + "=" * 60)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("=" * 60)
print(f"\nModels saved in: {os.path.abspath('models')}")
print("\nYou can now run: streamlit run app.py")
