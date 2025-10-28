import os
import sys

print("Checking for trained models...")

model_dir = 'models'
required_models = [
    'linear_regression.pkl',
    'polynomial_regression.pkl',
    'decision_tree.pkl',
    'svm.pkl',
    'random_forest.pkl',
    'gradient_boosting.pkl',
    'kmeans_3.pkl',
    'kmeans_5.pkl',
    'dbscan.pkl',
    'pca.pkl',
    'svd.pkl',
    'scaler.pkl',
    'train_test_splits.pkl',
    'classification_labels.pkl'
]

if not os.path.exists(model_dir):
    print(f"\n❌ Models directory '{model_dir}' not found!")
    print("\n🔧 Run this command to train all models:")
    print("   python train_all_models.py")
    sys.exit(1)

missing_models = []
for model_file in required_models:
    filepath = os.path.join(model_dir, model_file)
    if not os.path.exists(filepath):
        missing_models.append(model_file)
        print(f"❌ Missing: {model_file}")
    else:
        print(f"✅ Found: {model_file}")

if missing_models:
    print(f"\n❌ {len(missing_models)} model(s) missing!")
    print("\n🔧 Run this command to train all models:")
    print("   python train_all_models.py")
    sys.exit(1)
else:
    print(f"\n✅ All {len(required_models)} models are ready!")
    print("\n🚀 You can now run:")
    print("   streamlit run app.py")
    sys.exit(0)
