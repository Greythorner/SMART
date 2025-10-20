#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""


Summary
-------
We present an end-to-end pipeline for classifying 1D resistance time-series
acquired from steel-ball combination experiments. The pipeline comprises:
(i) per-sample Savitzky–Golay smoothing, (ii) global min–max normalization,
(iii) stratified train/validation splitting, (iv) supervised training of a
compact 1D CNN with early stopping and checkpointing, and (v) rigorous logging
of per-epoch metrics for downstream reporting.

Data Assumptions
----------------
- Excel file includes:
  - Resistance1 ... Resistance3500  (time-series per example)
  - class_label1 ... class_label5    (one-hot labels; 5 classes)
- Each row corresponds to one example.


"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam


# ============================== Configuration ==============================

# I/O
OUTPUT_DIR = r"C:/Users/wuqiushuo/Desktop/20241221/球/"
EXCEL_PATH = os.path.join(OUTPUT_DIR, "Five different sizes balls recognition algorithm data.xlsx")

# Series & labels
N_FEATURES = 3500
RESISTANCE_COLS = [f"Resistance{i}" for i in range(1, N_FEATURES + 1)]
N_CLASSES = 5
LABEL_COLS = [f"class_label{i}" for i in range(1, N_CLASSES + 1)]

# Savitzky–Golay smoothing
SG_WINDOW = 51   # must be odd and <= N_FEATURES
SG_POLY   = 3

# Split & training
TEST_SIZE = 0.20
RANDOM_STATE = 57
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 200
EARLY_STOP_PATIENCE = 200


# ============================ Reproducibility ===============================

def set_global_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Optional: make TF ops deterministic (can reduce performance)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "0")

set_global_seed(42)


# ================================ I/O Utils =================================

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    miss_x = [c for c in RESISTANCE_COLS if c not in df.columns]
    miss_y = [c for c in LABEL_COLS if c not in df.columns]
    if miss_x:
        raise ValueError(f"Missing resistance columns (first few): {miss_x[:5]} ...")
    if miss_y:
        raise ValueError(f"Missing label columns: {miss_y}")
    return df


# ============================== Pre-processing ==============================

def validate_sg_window(window: int, series_len: int) -> int:
    """Ensure SG window is odd and <= series length."""
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w > series_len:
        w = series_len if series_len % 2 == 1 else series_len - 1
    if w < 3:
        w = 3
    return w

def smooth_and_normalize(X: np.ndarray) -> np.ndarray:
    """
    Row-wise Savitzky–Golay smoothing + global min–max normalization.

    Parameters
    ----------
    X : (N, T) array

    Returns
    -------
    Xn : (N, T) float32
    """
    assert X.ndim == 2, "Expected 2D array (N, T)."
    _, T = X.shape
    w = validate_sg_window(SG_WINDOW, T)

    X_smooth = np.apply_along_axis(
        lambda row: savgol_filter(row, window_length=w, polyorder=SG_POLY),
        axis=1, arr=X
    )
    x_min, x_max = X_smooth.min(), X_smooth.max()
    if x_max <= x_min:
        return np.zeros_like(X_smooth, dtype=np.float32)
    Xn = (X_smooth - x_min) / (x_max - x_min)
    return Xn.astype(np.float32)


# ================================== Model ===================================

def build_improved_classification_model(input_dim: int, n_classes: int = N_CLASSES) -> tf.keras.Model:
    """
    1D CNN mirroring the user-provided architecture.
    Input shape: (T, 1) with T == input_dim.
    """
    inputs = Input(shape=(input_dim, 1))
    x = Conv1D(256, kernel_size=5, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=10, strides=2, padding='same')(x)

    x = Conv1D(128, kernel_size=5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=10, strides=2, padding='same')(x)

    x = Conv1D(64, kernel_size=5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=10, strides=2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Dropout(0.5)(x)  # keep user's dropout setting

    outputs = Dense(n_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs, name="BallCombo1D_CNN")


# ============================= Metrics Callback =============================

class MetricsCallback(Callback):
    """Collect per-epoch train/val metrics for external reporting."""
    def __init__(self):
        super().__init__()
        self.train_loss, self.val_loss = [], []
        self.train_acc,  self.val_acc  = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(float(logs.get('loss', np.nan)))
        self.train_acc.append(float(logs.get('accuracy', np.nan)))
        self.val_loss.append(float(logs.get('val_loss', np.nan)))
        self.val_acc.append(float(logs.get('val_accuracy', np.nan)))


# ================================== Main ====================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    # ----- Load & check -----
    df = load_dataframe(EXCEL_PATH)
    X_raw = df[RESISTANCE_COLS].values.astype(np.float32)  # (N, 3500)
    Y     = df[LABEL_COLS].values.astype(np.float32)       # (N, 5)

    # ----- Pre-process -----
    X = smooth_and_normalize(X_raw)                        # (N, 3500)

    # Stratify by one-hot argmax
    y_idx = np.argmax(Y, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_idx
    )

    # Keras Conv1D expects (N, T, 1)
    X_train_3d = np.expand_dims(X_train, axis=-1)
    X_val_3d   = np.expand_dims(X_val,   axis=-1)

    # ----- Model -----
    model = build_improved_classification_model(input_dim=N_FEATURES, n_classes=N_CLASSES)
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # ----- Callbacks -----
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model_weights.h5")
    metrics_cb = MetricsCallback()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=False),
        ModelCheckpoint(filepath=ckpt_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1, mode='max', save_weights_only=True),
        metrics_cb
    ]

    # ----- Train -----
    history = model.fit(
        X_train_3d, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_3d, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # ----- Export metrics -----
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(metrics_cb.train_loss) + 1)),
        "Train_Loss": metrics_cb.train_loss,
        "Train_Accuracy": metrics_cb.train_acc,
        "Val_Loss": metrics_cb.val_loss,
        "Val_Accuracy": metrics_cb.val_acc,
    })
    metrics_path = os.path.join(OUTPUT_DIR, "training_and_validation_metrics.xlsx")
    metrics_df.to_excel(metrics_path, index=False)

    print(f"[OK] Metrics saved to: {metrics_path}")
    print(f"[OK] Best weights saved to: {ckpt_path}")


if __name__ == "__main__":
    main()

