"""
Pre-train all models to speed up Streamlit app loading
Run this script once: python train_all_models.py
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from ucimlrepo import fetch_ucirepo
import numpy as np

print("Loading data...")
superconductivity_data = fetch_ucirepo(id=464)
X = superconductivity_data.data.features
y = superconductivity_data.data.targets
X = X.fillna(X.mean())
y = y.fillna(y.mean())
y_class = (y > y.median()).astype(int)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train and save models
models = {}

print("\n1. Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
models['linear_regression'] = lr

print("2. Training Polynomial Regression...")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y_train)
models['poly_regression'] = lr_poly
models['poly_features'] = poly

print("3. Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_c, y_train_c)
models['decision_tree'] = dt

print("4. Training SVM (this may take a while)...")
svm = SVC(random_state=42, kernel='rbf')
svm.fit(X_train_c, y_train_c)
models['svm'] = svm

print("5. Training Random Forest...")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_c, y_train_c)
models['random_forest'] = rf

print("6. Training Gradient Boosting...")
gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb.fit(X_train_c, y_train_c)
models['gradient_boosting'] = gb

print("7. Training K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
models['kmeans'] = kmeans

print("8. Training DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
models['dbscan'] = dbscan

print("9. Training PCA...")
pca = PCA(n_components=10)
pca.fit(X_scaled)
models['pca'] = pca

print("10. Training SVD...")
svd = TruncatedSVD(n_components=10, random_state=42)
svd.fit(X_scaled)
models['svd'] = svd

print("11. Saving scaler...")
models['scaler'] = scaler

# Save all models
print("\nSaving models...")
for name, model in models.items():
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved {name}.pkl")

print("\n✅ All models trained and saved successfully!")
print("You can now run the Streamlit app with fast loading.")
