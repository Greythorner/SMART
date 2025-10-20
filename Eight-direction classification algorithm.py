#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Reproducibility
---------------
- Python ≥ 3.9
- Packages: numpy, pandas, scipy, scikit-learn, openpyxl (for Excel I/O)
- Determinism is encouraged by explicit random_state=57 where applicable.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

# Silence non-critical warnings for cleaner logs (optional)
warnings.filterwarnings("ignore")

# ===============================
# 0) Configuration
# ===============================
DATA_PATH = "C:/Users/wuqiushuo/Desktop/Eight-direction classification algorithm data.xlsx"  # <-- set to your file path
N_SAMPLES = 5000  # number of time points per trace

# Savitzky–Golay smoothing params (window_size must be odd)
SG_WINDOW = 51
SG_POLY = 3

# Label configuration (one-hot columns)
N_CLASSES = 8

# Train/validation split & random seeds
TEST_SIZE = 0.20
RANDOM_STATE = 57


# ===============================
# 1) Data loading & smoothing
# ===============================
# Load the dataset (expects an .xlsx with the columns described above)
data = pd.read_excel(DATA_PATH, engine="openpyxl")

# Assemble raw time-series matrix: shape (num_samples, N_SAMPLES)
resistance_columns = [f"Resistance{i}" for i in range(1, N_SAMPLES + 1)]
resistance_values = data[resistance_columns].values

# Apply Savitzky–Golay filter row-wise for denoising
# (vectorized row-wise application for clarity)
smoothed_values = np.apply_along_axis(
    lambda row: savgol_filter(row, window_length=SG_WINDOW, polyorder=SG_POLY),
    axis=1,
    arr=resistance_values
)


# ===============================
# 2) Feature engineering
# ===============================
def extract_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Compute lightweight time- and frequency-domain descriptors.

    Parameters
    ----------
    signal : np.ndarray
        1D array of shape (N_SAMPLES,)

    Returns
    -------
    Dict[str, float]
        Dictionary of scalar features per sample.
    """
    features = {}

    # ---- Time-domain features ----
    features["mean"] = float(np.mean(signal))
    features["std"] = float(np.std(signal))
    features["min"] = float(np.min(signal))
    features["max"] = float(np.max(signal))
    features["median"] = float(np.median(signal))
    features["skew"] = float(skew(signal))
    features["kurtosis"] = float(kurtosis(signal))
    features["ptp"] = float(np.ptp(signal))  # peak-to-peak amplitude

    # ---- Frequency-domain features (magnitude spectrum summary) ----
    fft_vals = np.abs(np.fft.fft(signal))
    features["fft_mean"] = float(np.mean(fft_vals))
    features["fft_std"] = float(np.std(fft_vals))

    return features


# Extract features for each sample and aggregate into a DataFrame
feature_rows = [extract_features(row) for row in smoothed_values]
features_df = pd.DataFrame(feature_rows)

# Standardize features (z-score); retain numpy array for downstream ML
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_df.values)


# ===============================
# 3) Labels & stratified split
# ===============================
# One-hot labels → integer class ids (0..N_CLASSES-1)
class_label_cols = [f"class_label{i}" for i in range(1, N_CLASSES + 1)]
y_onehot = data[class_label_cols].values
y = np.argmax(y_onehot, axis=1)

# Stratified split preserves class proportions
X_train, X_val, y_train, y_val = train_test_split(
    features_normalized,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ===============================
# 4) Models & ensemble
# ===============================
# Base learners
clf_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)
clf_gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)

# Soft-voting ensemble leverages predicted probabilities
ensemble_clf = VotingClassifier(
    estimators=[("rf", clf_rf), ("gb", clf_gb)],
    voting="soft"
)

# Train ensemble
ensemble_clf.fit(X_train, y_train)


# ===============================
# 5) Evaluation
# ===============================
y_pred = ensemble_clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("Validation Accuracy: {:.4f}".format(acc))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

