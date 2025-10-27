"""
Configuration settings for the ML application
"""

# Cache settings
CACHE_TTL = 3600  # 1 hour

# Model settings
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 10
USE_PARALLEL = True  # Use n_jobs=-1 for parallel processing

# Data sampling for visualization
MAX_PLOT_SAMPLES = 5000  # Limit scatter plot points

# Streamlit settings
WIDE_MODE = True
INITIAL_SIDEBAR_STATE = "expanded"

# Session state keys
SESSION_DATA_KEY = "superconductivity_data"
SESSION_MODELS_KEY = "trained_models"

# Dataset info
DATASET_ID = 464

# Data source settings
USE_LOCAL_CSV = True  # Set to True to use local CSV files
TRAIN_CSV_PATH = "train.csv"
UNIQUE_CSV_PATH = "unique_m.csv"

# Page settings
PAGE_TITLE = "Superconductivity ML Project"
PAGE_ICON = "🔬"

# Model training settings
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42
