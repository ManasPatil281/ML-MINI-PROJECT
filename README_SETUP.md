# Setup Instructions

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train All Models (REQUIRED - First Time Only)
```bash
python train_all_models.py
```

This will:
- Load the dataset (train.csv)
- Train all 8 ML models
- Save models to `models/` folder
- Takes ~2-3 minutes

### Step 3: Check Models (Optional)
```bash
python check_models.py
```

### Step 4: Run Streamlit App
```bash
streamlit run app.py
```

## Troubleshooting

### Error: "Models not found"
**Solution:** Run `python train_all_models.py`

### Error: "train.csv not found"
**Solution:** Ensure `train.csv` is in the project root directory

### Models to Train
1. Linear Regression
2. Polynomial Regression (degree=2)
3. Decision Tree Classifier
4. SVM Classifier
5. Random Forest Classifier
6. Gradient Boosting Classifier
7. K-Means Clustering (k=3, k=5)
8. DBSCAN Clustering
9. PCA (10 components)
10. SVD (10 components)

## File Structure
```
ML MINI PROJECT/
├── app.py                          # Main app
├── train_all_models.py            # Train all models (RUN THIS FIRST!)
├── check_models.py                # Check if models exist
├── train.csv                      # Dataset
├── models/                        # Trained models (created by training)
│   ├── linear_regression.pkl
│   ├── polynomial_regression.pkl
│   └── ...
└── pages/                         # Streamlit pages
    ├── 1_📊_Data_Exploration.py
    ├── 2_📈_Regression_Models.py
    └── ...
```
