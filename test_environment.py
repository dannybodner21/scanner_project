#!/usr/bin/env python3
"""Test script to diagnose numpy/pandas issues"""

import sys
import os
print(f"Python path: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} imported successfully")
    print(f"NumPy file: {np.__file__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__} imported successfully")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import django
    print(f"✅ Django {django.VERSION} imported successfully")
except ImportError as e:
    print(f"❌ Django import failed: {e}")