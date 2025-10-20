#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Dependencies
------------
Python >= 3.9
numpy, pandas, scipy, scikit-learn, openpyxl
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =============================================================================
# CONFIGURATION (centralized, publication-friendly)
# =============================================================================

@dataclass(frozen=True)
class Config:
    # I/O
    input_excel: str = r'C:/Users/wuqiushuo/Desktop/Position predition algorithm Data.xlsx'
    id_col: str = 'id'

    # Schema
    resistance_prefix: str = 'Resistance'
    class_label_prefix: str = 'class_label'
    num_classes: int = 49  # "class_label1"..."class_label49"

    # Peak detection (kept as original logic)
    peak_height: float = 1.1
    peak_distance: int = 50

    # Train/test split (stratified 80/20)
    test_size: float = 0.2
    random_state: int = 512

    # Model selection spaces (as in the original plan)
    svm_param_grid: dict | None = None
    rf_param_grid: dict | None = None
    lr_param_grid: dict | None = None

    # Soft-voting weights (svm, rf, lr)
    voting_weights: tuple[int, int, int] = (1, 2, 1)


def _default_grids() -> tuple[dict, dict, dict]:
    """Default grids aligned with the original specification."""
    svm_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    rf_grid  = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}
    lr_grid  = {'C': [0.1, 1, 10]}
    return svm_grid, rf_grid, lr_grid


CFG = Config()
if CFG.svm_param_grid is None or CFG.rf_param_grid is None or CFG.lr_param_grid is None:
    _svm, _rf, _lr = _default_grids()
    CFG = Config(
        svm_param_grid=_svm,
        rf_param_grid=_rf,
        lr_param_grid=_lr
    )


# =============================================================================
# DATA LOADING & SCHEMA PARSING
# =============================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the Excel dataset and assign a deterministic sample ID.
    """
    df = pd.read_excel(path, engine='openpyxl')
    df[CFG.id_col] = np.arange(1, len(df) + 1, dtype=int)
    return df


def parse_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identify resistance columns and one-hot label columns.

    Returns
    -------
    resistance_cols : list[str]
    class_cols      : list[str]
    """
    resistance_cols = [c for c in df.columns if CFG.resistance_prefix in c]
    if not resistance_cols:
        raise ValueError("No resistance columns detected (expected names like 'Resistance1', ...).")

    class_cols = [f"{CFG.class_label_prefix}{i}" for i in range(1, CFG.num_classes + 1)]
    missing = [c for c in class_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing class label columns: {missing}")

    return resistance_cols, class_cols


def melt_to_long(df: pd.DataFrame, resistance_cols: list[str]) -> pd.DataFrame:
    """
    Convert wide-format resistance columns to a long series per sample
    for peak detection and feature extraction.

    Output columns: [id, time, value]
    - 'time' is the numeric index extracted from 'ResistanceX'.
    """
    long_df = pd.melt(
        df[[CFG.id_col] + resistance_cols],
        id_vars=[CFG.id_col],
        value_vars=resistance_cols,
        var_name='time',
        value_name='value'
    )
    long_df['time'] = long_df['time'].str.extract(r'(\d+)').astype(int)
    return long_df


# =============================================================================
# FEATURE ENGINEERING (peak-based; preserved logic)
# =============================================================================

def extract_custom_features(single_series: pd.DataFrame,
                            height_threshold: float,
                            distance_threshold: int) -> dict:
    """
    Derive peak-based features per sample (unchanged in spirit).

    Features
    --------
    - num_peaks
    - peak_time_diff_k           : differences between consecutive peak times
    - peak_time_ratio_k          : diffs normalized by total time span
    - peak_width_k               : proxy via time gaps between successive peaks
    - peak_height_k              : height above the series global minimum
    """
    features = {CFG.id_col: int(single_series[CFG.id_col].iloc[0])}

    values = single_series['value'].to_numpy()
    times = single_series['time'].to_numpy()

    peaks, _ = find_peaks(values, height=height_threshold, distance=distance_threshold)
    features['num_peaks'] = int(len(peaks))

    if len(peaks) > 1:
        peak_times = times[peaks]

        # Time differences between consecutive peaks
        diffs = np.diff(peak_times)
        for i, d in enumerate(diffs):
            features[f'peak_time_diff_{i}'] = float(d)

        # Ratios normalized by total span (guard against zero-span)
        total_span = int(times.max() - times.min())
        total_span = total_span if total_span > 0 else 1
        ratios = diffs / float(total_span)
        for i, r in enumerate(ratios):
            features[f'peak_time_ratio_{i}'] = float(r)

        # Widths (proxy: gaps between successive peak positions)
        widths = peak_times - np.roll(peak_times, 1)
        for i, w in enumerate(widths[1:]):
            features[f'peak_width_{i}'] = float(w)

        # Heights above global minimum
        global_min = float(values.min())
        heights = values[peaks] - global_min
        for i, h in enumerate(heights):
            features[f'peak_height_{i}'] = float(h)

    return features


# =============================================================================
# MODELING (scikit-learn)
# =============================================================================

def prepare_xy(features_df: pd.DataFrame, data_raw: pd.DataFrame, class_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Align features and labels; convert one-hot labels to class indices.
    """
    X_df = features_df.copy().fillna(0.0).astype(np.float32)
    y = np.argmax(data_raw[class_cols].to_numpy(), axis=1).astype(int)
    return X_df, y


def fit_base_models(X_train: np.ndarray, y_train: np.ndarray, random_state: int):
    """
    Grid-search SVM / RF / LR with the original parameter spaces; return best estimators.
    """
    svm = SVC(kernel='rbf', probability=True, random_state=random_state)
    rf  = RandomForestClassifier(random_state=random_state)
    lr  = LogisticRegression(random_state=random_state + 10, max_iter=1000)

    svm_grid = GridSearchCV(svm, CFG.svm_param_grid, cv=5, n_jobs=-1)
    rf_grid  = GridSearchCV(rf,  CFG.rf_param_grid,  cv=5, n_jobs=-1)
    lr_grid  = GridSearchCV(lr,  CFG.lr_param_grid,  cv=5, n_jobs=-1)

    svm_grid.fit(X_train, y_train)
    rf_grid.fit(X_train, y_train)
    lr_grid.fit(X_train, y_train)

    return svm_grid.best_estimator_, rf_grid.best_estimator_, lr_grid.best_estimator_


def build_voter(svm_model, rf_model, lr_model) -> VotingClassifier:
    """
    Construct the soft-voting ensemble with fixed weights (svm, rf, lr).
    """
    return VotingClassifier(
        estimators=[('svm', svm_model), ('rf', rf_model), ('lr', lr_model)],
        voting='soft',
        weights=list(CFG.voting_weights)
    )


# =============================================================================
# MAIN (single printed metric)
# =============================================================================

def main() -> None:
    # Fix NumPy PRNG for reproducibility of any stochastic ops
    np.random.seed(CFG.random_state)

    # Load and validate schema
    data = load_dataset(CFG.input_excel)
    resistance_cols, class_cols = parse_schema(data)

    # Long-format melting for peak extraction
    long_df = melt_to_long(data, resistance_cols)

    # Per-sample feature extraction
    features = []
    for gid, g in long_df.groupby(CFG.id_col, sort=True):
        feats = extract_custom_features(
            single_series=g,
            height_threshold=CFG.peak_height,
            distance_threshold=CFG.peak_distance
        )
        features.append(feats)

    features_df = pd.json_normalize(features).fillna(0.0).astype(np.float32)

    # Prepare X/y and standardize
    X_df, y = prepare_xy(features_df, data, class_cols)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.test_size,
        shuffle=True,
        stratify=y,
        random_state=CFG.random_state
    )

    # Model selection and soft-voting
    svm_best, rf_best, lr_best = fit_base_models(X_train, y_train, random_state=462)
    voter = build_voter(svm_best, rf_best, lr_best)

    # Train and evaluate (single metric)
    voter.fit(X_train, y_train)
    y_pred = voter.predict(X_test)
    weighted_voting_accuracy = accuracy_score(y_test, y_pred)

    # ---- The only printed output (as requested) ----
    print(f"Weighted Voting Accuracy: {weighted_voting_accuracy:.6f}")


if __name__ == "__main__":
    main()


# In[ ]:




