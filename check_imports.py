"""
Check if all required packages are installed
"""

import sys

print("Python version:", sys.version)
print("\nChecking imports...")

try:
    import pickle
    print("✅ pickle")
except ImportError as e:
    print("❌ pickle:", e)

try:
    import pandas as pd
    print("✅ pandas")
except ImportError as e:
    print("❌ pandas:", e)

try:
    import numpy as np
    print("✅ numpy")
except ImportError as e:
    print("❌ numpy:", e)

try:
    import sklearn
    print("✅ sklearn")
except ImportError as e:
    print("❌ sklearn:", e)

print("\nIf any imports failed, run:")
print("pip install pandas numpy scikit-learn")
