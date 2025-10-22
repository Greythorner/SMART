#!/usr/bin/env python
# coding: utf-8
"""

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, List

# Avoid MKL/OpenMP duplicate warnings in some environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

from tsfresh import extract_features, select_features
# from tsfresh.feature_extraction import MinimalFCParameters  # Optional: to drastically reduce feature set & memory

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    # I/O
    input_excel: str = r"C:/Users/wuqiushuo/Desktop/Position predition algorithm Data.xlsx"
    id_col: str = "id"

    # Schema
    resistance_prefix: str = "Resistance"
    class_label_prefix: str = "class_label"
    num_classes: int = 49  # class_label1..class_label49

    # tsfresh long-form column names
    tsfresh_column_id: str = "id"
    tsfresh_column_sort: str = "time"
    tsfresh_column_value: str = "value"

    # Train/test split
    test_size: float = 0.2        # stratified 80/20 split
    random_state: int = 52        # top-level RNG seed

    # Model selection grids
    svm_grid: dict | None = None
    rf_grid: dict | None = None
    lr_grid: dict | None = None

    # Soft-voting weights (svm, rf, lr)
    voting_weights: Tuple[int, int, int] = (1, 2, 1)

    # Feature curation
    top_k_features: int = 200     # retain at most this many selected features


def _default_grids() -> Tuple[dict, dict, dict]:
    """Default hyperparameter grids for SVM, RandomForest, and LogisticRegression."""
    svm = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
    rf = {"n_estimators": [50, 100, 200], "max_depth": [10, 20, 30]}
    lr = {"C": [0.1, 1, 10]}
    return svm, rf, lr


CFG = Config()
if CFG.svm_grid is None or CFG.rf_grid is None or CFG.lr_grid is None:
    _svm, _rf, _lr = _default_grids()
    CFG = Config(svm_grid=_svm, rf_grid=_rf, lr_grid=_lr)


# =============================================================================
# Data loading & long-form conversion
# =============================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the Excel file and attach a deterministic per-row `id`.

    Parameters
    ----------
    path : str
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with an added `id` column.
    """
    df = pd.read_excel(path, engine="openpyxl")
    df[CFG.id_col] = np.arange(1, len(df) + 1, dtype=int)
    return df


def parse_schema(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify resistance and one-hot label columns.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-form input DataFrame.

    Returns
    -------
    (resistance_cols, class_cols)
        Lists of the detected columns; raises if any are missing.
    """
    resistance_cols = [c for c in df.columns if CFG.resistance_prefix in c]
    if not resistance_cols:
        raise ValueError(
            "No resistance columns found (expected names like 'Resistance1', 'Resistance2', ...)."
        )

    class_cols = [f"{CFG.class_label_prefix}{i}" for i in range(1, CFG.num_classes + 1)]
    missing = [c for c in class_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    return resistance_cols, class_cols


def melt_to_long(df: pd.DataFrame, resistance_cols: List[str]) -> pd.DataFrame:
    """
    Convert wide resistance columns into long form for tsfresh.

    Output columns: [id, time, value], where `time` is parsed from the
    resistance column suffix.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-form DataFrame with resistance columns.
    resistance_cols : List[str]
        Names of resistance columns to melt.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame sorted by [id, time].
    """
    long_df = pd.melt(
        df[[CFG.id_col] + resistance_cols],
        id_vars=[CFG.id_col],
        value_vars=resistance_cols,
        var_name="time",
        value_name="value",
    )
    # Extract numeric time index from names like "Resistance123" → 123
    long_df["time"] = long_df["time"].str.extract(r"(\d+)").astype(int)
    long_df = long_df.sort_values(
        [CFG.tsfresh_column_id, CFG.tsfresh_column_sort], kind="mergesort"
    )
    long_df.reset_index(drop=True, inplace=True)
    return long_df


# =============================================================================
# Feature extraction / selection
# =============================================================================

def extract_tsfresh_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unsupervised tsfresh feature extraction on the long-form series.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-form data with columns [id, time, value].

    Returns
    -------
    pd.DataFrame
        Feature matrix (index aligned to sample id). Non-numeric and all-NaN
        columns are dropped; ±inf values are replaced with NaN.
    """
    # Use a positive integer for n_jobs; -1 is not supported by tsfresh
    n_jobs_ts = max(1, (os.cpu_count() or 1))

    feats = extract_features(
        long_df,
        column_id=CFG.tsfresh_column_id,
        column_sort=CFG.tsfresh_column_sort,
        column_value=CFG.tsfresh_column_value,
        n_jobs=n_jobs_ts,
        disable_progressbar=True,
        # default_fc_parameters=MinimalFCParameters(),  # <- enable to reduce compute/memory if needed
    )

    feats = feats.select_dtypes(include=[np.number])
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.loc[:, feats.notna().any(axis=0)]
    return feats


def impute_fit_transform(feats_train: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    """
    Fit a mean imputer on the training features and transform the training set.

    Parameters
    ----------
    feats_train : pd.DataFrame
        Training feature matrix.

    Returns
    -------
    (X_train_imputed, imputer)
    """
    imputer = SimpleImputer(strategy="mean")
    Xtr = pd.DataFrame(
        imputer.fit_transform(feats_train),
        index=feats_train.index,
        columns=feats_train.columns,
    )
    return Xtr, imputer


def impute_transform(feats: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:
    """
    Apply a fitted imputer to a new feature matrix (e.g., test set).

    Parameters
    ----------
    feats : pd.DataFrame
        Feature matrix to transform.
    imputer : SimpleImputer
        Pre-fitted imputer.

    Returns
    -------
    pd.DataFrame
        Transformed feature matrix.
    """
    return pd.DataFrame(imputer.transform(feats), index=feats.index, columns=feats.columns)


def select_supervised_on_train(
    Xtr_imputed: pd.DataFrame, y_train: np.ndarray
) -> pd.DataFrame:
    """
    Supervised tsfresh feature selection on the **training** set only.

    Parameters
    ----------
    Xtr_imputed : pd.DataFrame
        Imputed training feature matrix.
    y_train : np.ndarray
        Training labels (class indices).

    Returns
    -------
    pd.DataFrame
        Column-reduced training matrix after supervised selection (capped
        to `CFG.top_k_features` for compactness).
    """
    y_series = pd.Series(y_train, index=Xtr_imputed.index)
    selected = select_features(Xtr_imputed, y_series)
    if selected.shape[1] > CFG.top_k_features:
        selected = selected.iloc[:, : CFG.top_k_features]
    return selected


# =============================================================================
# Modeling
# =============================================================================

def fit_best_base_models(
    X_train: np.ndarray, y_train: np.ndarray, seed: int
) -> tuple[SVC, RandomForestClassifier, LogisticRegression]:
    """
    Grid search best SVM, RandomForest, and LogisticRegression models on train only.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix (standardized).
    y_train : np.ndarray
        Class labels.
    seed : int
        Random seed for model reproducibility.

    Returns
    -------
    (svm_best, rf_best, lr_best)
    """
    svm = SVC(kernel="rbf", probability=True, random_state=seed)
    rf = RandomForestClassifier(random_state=seed)
    lr = LogisticRegression(random_state=seed, max_iter=1000)

    svm_grid = GridSearchCV(svm, CFG.svm_grid, cv=5, n_jobs=-1)
    rf_grid = GridSearchCV(rf, CFG.rf_grid, cv=5, n_jobs=-1)
    lr_grid = GridSearchCV(lr, CFG.lr_grid, cv=5, n_jobs=-1)

    svm_grid.fit(X_train, y_train)
    rf_grid.fit(X_train, y_train)
    lr_grid.fit(X_train, y_train)

    return svm_grid.best_estimator_, rf_grid.best_estimator_, lr_grid.best_estimator_


def build_voter(
    svm_model: SVC, rf_model: RandomForestClassifier, lr_model: LogisticRegression
) -> VotingClassifier:
    """
    Compose a soft-voting ensemble from the three tuned base models.

    Returns
    -------
    VotingClassifier
    """
    return VotingClassifier(
        estimators=[("svm", svm_model), ("rf", rf_model), ("lr", lr_model)],
        voting="soft",
        weights=list(CFG.voting_weights),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # Global RNG seed (estimators have their own random_state as well)
    np.random.seed(CFG.random_state)

    # 1) Load & schema
    data = load_dataset(CFG.input_excel)
    resistance_cols, class_cols = parse_schema(data)

    # 2) Long-form conversion (X) & tsfresh feature extraction (unsupervised)
    long_df = melt_to_long(data, resistance_cols)
    extracted_features = extract_tsfresh_features(long_df)

    # 3) Labels (one-hot → class index) with validity checks
    class_mat = data[class_cols].to_numpy()
    row_sum = class_mat.sum(axis=1)
    if not np.all(row_sum == 1):
        bad_idx = np.where(row_sum != 1)[0]
        raise ValueError(
            f"Invalid one-hot rows where row sum != 1. "
            f"First bad indices: {bad_idx[:10].tolist()} (total bad = {len(bad_idx)})"
        )
    y = np.argmax(class_mat, axis=1).astype(int)

    # Align y index to features’ index (tsfresh index == sample id)
    y_series = pd.Series(y, index=extracted_features.index)

    # 4) Split BEFORE any supervised step (consistent row order / stratified)
    n = extracted_features.shape[0]
    idx_all = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx_all,
        test_size=CFG.test_size,
        shuffle=True,
        stratify=y_series.values,
        random_state=CFG.random_state,
    )

    X_all = extracted_features
    y_all = y_series

    X_train_raw = X_all.iloc[train_idx]
    X_test_raw = X_all.iloc[test_idx]
    y_train = y_all.iloc[train_idx].to_numpy()
    y_test = y_all.iloc[test_idx].to_numpy()

    # 5) Imputation: fit on train, transform train/test (±inf → NaN)
    X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
    X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)

    X_train_imp, imputer = impute_fit_transform(X_train_raw)
    X_test_imp = impute_transform(X_test_raw, imputer)

    # 6) Supervised feature selection: fit on train only
    X_train_sel = select_supervised_on_train(X_train_imp, y_train)
    keep_cols = list(X_train_sel.columns[: CFG.top_k_features])

    # Align test columns; fill missing with 0.0 (features not selected from train)
    X_test_sel = X_test_imp.reindex(columns=keep_cols, fill_value=0.0)

    # 7) Standardization: fit on train only
    scaler = StandardScaler().fit(X_train_sel)
    X_train = scaler.transform(X_train_sel)
    X_test = scaler.transform(X_test_sel)

    # 8) Base model selection & soft-voting ensemble (train split only)
    svm_best, rf_best, lr_best = fit_best_base_models(X_train, y_train, seed=CFG.random_state)
    voter = build_voter(svm_best, rf_best, lr_best)

    # 9) Train & evaluate on the held-out test set
    voter.fit(X_train, y_train)
    y_pred = voter.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Voting Accuracy (hold-out test): {acc:.6f}")


if __name__ == "__main__":
    main()
