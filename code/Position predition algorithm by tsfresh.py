#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Dependencies
------------
Python >= 3.9
numpy, pandas, tsfresh, scikit-learn, openpyxl, scipy
"""

from __future__ import annotations

import os
# Allow MKL/OpenMP duplication in some environments (e.g., Jupyter with TSFRESH/numba)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

from dataclasses import dataclass
from tsfresh import extract_features, select_features
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    # I/O
    input_excel: str = r'C:/Users/wuqiushuo/Desktop/Position predition algorithm Data.xlsx'
    id_col: str = 'id'

    # Schema
    resistance_prefix: str = 'Resistance'
    class_label_prefix: str = 'class_label'
    num_classes: int = 49  # class_label1..class_label49

    # TSFRESH settings (using defaults via convenience API)
    tsfresh_column_id: str = 'id'
    tsfresh_column_sort: str = 'time'
    tsfresh_column_value: str = 'value'

    # Train/test split
    test_size: float = 0.2          # stratified 80/20
    random_state: int = 42          # keep user’s original seed for this script

    # Model selection grids (faithful to the original)
    svm_grid: dict | None = None
    rf_grid: dict | None = None
    lr_grid: dict | None = None

    # Soft-voting weights (svm, rf, lr)
    voting_weights: tuple[int, int, int] = (1, 2, 1)

    # Feature curation
    top_k_features: int = 200       # retain up to 200 selected features


def _default_grids():
    svm = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    rf  = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}
    lr  = {'C': [0.1, 1, 10]}
    return svm, rf, lr


CFG = Config()
if CFG.svm_grid is None or CFG.rf_grid is None or CFG.lr_grid is None:
    _svm, _rf, _lr = _default_grids()
    CFG = Config(svm_grid=_svm, rf_grid=_rf, lr_grid=_lr)


# =============================================================================
# DATA LOADING & LONG-FORM TRANSFORM
# =============================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load Excel and assign a deterministic per-row sample id.
    """
    df = pd.read_excel(path, engine='openpyxl')
    df[CFG.id_col] = np.arange(1, len(df) + 1, dtype=int)
    return df


def parse_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identify resistance columns and one-hot class label columns.
    """
    resistance_cols = [c for c in df.columns if CFG.resistance_prefix in c]
    if not resistance_cols:
        raise ValueError("No resistance columns found (expected 'Resistance*').")

    class_cols = [f"{CFG.class_label_prefix}{i}" for i in range(1, CFG.num_classes + 1)]
    missing = [c for c in class_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    return resistance_cols, class_cols


def melt_to_long(df: pd.DataFrame, resistance_cols: list[str]) -> pd.DataFrame:
    """
    Convert wide resistance columns into long series for TSFRESH.
    Output columns: [id, time, value]
    """
    long_df = pd.melt(
        df[[CFG.id_col] + resistance_cols],
        id_vars=[CFG.id_col],
        value_vars=resistance_cols,
        var_name='time',
        value_name='value'
    )
    # Extract numeric index from names like "Resistance123" → 123
    long_df['time'] = long_df['time'].str.extract(r'(\d+)').astype(int)
    return long_df


# =============================================================================
# FEATURE EXTRACTION / SELECTION
# =============================================================================

def extract_tsfresh_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    TSFRESH feature extraction over the long-format series.
    """
    feats = extract_features(
        long_df,
        column_id=CFG.tsfresh_column_id,
        column_sort=CFG.tsfresh_column_sort,
        column_value=CFG.tsfresh_column_value
    )
    # Keep numeric columns; drop all-NaN columns pre-imputation to avoid mean-strategy errors
    feats = feats.select_dtypes(include=[np.number])
    feats = feats.loc[:, feats.notna().any(axis=0)]
    return feats


def impute_missing(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-impute remaining missing values using scikit-learn SimpleImputer.
    """
    imputer = SimpleImputer(strategy='mean')
    imputed = imputer.fit_transform(feats)
    return pd.DataFrame(imputed, index=feats.index, columns=feats.columns)


def select_supervised_features(feats_imputed: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Supervised TSFRESH feature selection. Returns a column-subset DataFrame.
    """
    # TSFRESH expects y aligned by index; ensure Series with same index
    y_series = pd.Series(y, index=feats_imputed.index)
    selected = select_features(feats_imputed, y_series)
    # Retain up to top_k_features (by column order as in original practice)
    if selected.shape[1] > CFG.top_k_features:
        selected = selected.iloc[:, :CFG.top_k_features]
    return selected


# =============================================================================
# MODELING
# =============================================================================

def fit_best_base_models(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    """
    Grid-search best SVM/RF/LR estimators using predefined parameter grids.
    """
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    rf  = RandomForestClassifier(random_state=seed)
    lr  = LogisticRegression(random_state=seed, max_iter=1000)

    svm_grid = GridSearchCV(svm, CFG.svm_grid, cv=5, n_jobs=-1)
    rf_grid  = GridSearchCV(rf,  CFG.rf_grid,  cv=5, n_jobs=-1)
    lr_grid  = GridSearchCV(lr,  CFG.lr_grid,  cv=5, n_jobs=-1)

    svm_grid.fit(X_train, y_train)
    rf_grid.fit(X_train, y_train)
    lr_grid.fit(X_train, y_train)

    return svm_grid.best_estimator__, rf_grid.best_estimator__, lr_grid.best_estimator__


def build_voter(svm_model, rf_model, lr_model) -> VotingClassifier:
    """
    Assemble the soft-voting ensemble with fixed weights (svm, rf, lr).
    """
    return VotingClassifier(
        estimators=[('svm', svm_model), ('rf', rf_model), ('lr', lr_model)],
        voting='soft',
        weights=list(CFG.voting_weights)
    )


# =============================================================================
# MAIN (prints ONLY Weighted Voting Accuracy)
# =============================================================================

def main() -> None:
    # Deterministic NumPy seed (complements model seeds)
    np.random.seed(CFG.random_state)

    # Load & schema
    data = load_dataset(CFG.input_excel)
    resistance_cols, class_cols = parse_schema(data)

    # Long-format and TSFRESH features
    long_df = melt_to_long(data, resistance_cols)
    extracted_features = extract_tsfresh_features(long_df)

    # Labels (one-hot → class index); keep alignment by index
    class_mat = data[class_cols].to_numpy()
    y = np.argmax(class_mat, axis=1).astype(int)
    y_series_indexed = pd.Series(y, index=extracted_features.index)

    # Impute → supervised selection → top-K curation
    feats_imputed = impute_missing(extracted_features)
    selected_feats = select_supervised_features(feats_imputed, y_series_indexed)

    # Standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(selected_feats)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.test_size,
        shuffle=True,
        stratify=y,
        random_state=CFG.random_state
    )

    # Base model selection and soft voting
    svm_best, rf_best, lr_best = fit_best_base_models(X_train, y_train, seed=52)
    voter = build_voter(svm_best, rf_best, lr_best)

    # Train & evaluate (single metric)
    voter.fit(X_train, y_train)
    y_pred = voter.predict(X_test)
    weighted_voting_accuracy = accuracy_score(y_test, y_pred)

    # ---- The only printed output ----
    print(f"Weighted Voting Accuracy: {weighted_voting_accuracy:.6f}")


if __name__ == "__main__":
    main()

