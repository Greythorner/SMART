#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Summary
-------
We present a compact 1D convolutional neural network (CNN) pipeline for
multi-class classification of resistance time-series acquired during a steel-ball
experiment. The pipeline performs (i) per-sample smoothing via a Savitzky–Golay
filter, (ii) global min–max normalization, (iii) stratified train/validation
splitting, (iv) supervised training with early stopping and checkpointing, and
(v) two-dimensional embedding of the latent data manifold using t-SNE for
qualitative assessment. The implementation is lightweight and reproducible.

Data Assumptions
----------------
- Input Excel file contains columns named:
  - Resistance1, Resistance2, ..., Resistance4499 (time-series)
  - class_label1, ..., class_label5 (one-hot encoding of class labels)
- Each row corresponds to a single example.

Reproducibility
---------------
We set NumPy and TensorFlow seeds. Determinism may still be affected by
non-deterministic GPU kernels. For strict determinism, consult the TF docs.

Notes
-----
- The CNN expects input with shape (N, T, 1). We explicitly expand the channel
  dimension.
- The smoothing window length must be odd and <= the series length; we guard it.
- The model architecture and the overall logic follow the user’s original code.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from scipy.signal import savgol_filter

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ----------------------------- Configuration ----------------------------- #

# Output directory and data source.
OUTPUT_DIR = r"C:/Users/wuqiushuo/Desktop/20241219/"
EXCEL_PATH = os.path.join(OUTPUT_DIR, "ball position recognition algorithm data.xlsx")

# Resistance series configuration.
N_FEATURES = 4499
RESISTANCE_COLS = [f"Resistance{i}" for i in range(1, N_FEATURES + 1)]

# One-hot label columns (5 classes).
N_CLASSES = 5
LABEL_COLS = [f"class_label{i}" for i in range(1, N_CLASSES + 1)]

# Savitzky–Golay smoothing.
SG_WINDOW = 51    # must be odd and <= N_FEATURES
SG_POLY = 3

# Train/val split.
TEST_SIZE = 0.20
RANDOM_STATE = 57

# Optimization and training.
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 200
EARLY_STOP_PATIENCE = 200


# ----------------------------- Reproducibility --------------------------- #

def set_global_seed(seed: int = 42) -> None:
    """Set global seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Optional: enforce TF determinism (may reduce performance)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "0")


set_global_seed(42)


# ------------------------------ I/O Utilities ---------------------------- #

def ensure_output_dir(path: str) -> None:
    """Ensure output directory exists."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load Excel file containing the dataset.

    Returns
    -------
    df : pd.DataFrame
        Must contain RESISTANCE_COLS and LABEL_COLS.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    missing_r = [c for c in RESISTANCE_COLS if c not in df.columns]
    missing_y = [c for c in LABEL_COLS if c not in df.columns]
    if missing_r:
        raise ValueError(f"Missing resistance columns: {missing_r[:5]} ...")
    if missing_y:
        raise ValueError(f"Missing label columns: {missing_y}")
    return df


# ----------------------------- Pre-processing ---------------------------- #

def validate_sg_window(window: int, series_len: int) -> int:
    """Ensure Savitzky–Golay window is odd and <= series length."""
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, series_len if series_len % 2 == 1 else series_len - 1)
    if w < 3:
        w = 3  # minimal odd window acceptable by savgol_filter
    return w


def smooth_and_normalize(X: np.ndarray) -> np.ndarray:
    """
    Apply Savitzky–Golay smoothing row-wise and global min–max normalization.

    Parameters
    ----------
    X : np.ndarray, shape (N, T)
        Raw resistance sequences.

    Returns
    -------
    X_norm : np.ndarray, shape (N, T)
        Smoothed and globally min–max normalized sequences.
    """
    assert X.ndim == 2, "Expected 2D array (N, T)."
    N, T = X.shape
    w = validate_sg_window(SG_WINDOW, T)
    # Row-wise smoothing
    X_smooth = np.apply_along_axis(
        lambda row: savgol_filter(row, window_length=w, polyorder=SG_POLY),
        axis=1,
        arr=X
    )
    # Global min–max normalization (across entire matrix)
    x_min = X_smooth.min()
    x_max = X_smooth.max()
    if x_max <= x_min:
        # Degenerate case: constant matrix
        return np.zeros_like(X_smooth, dtype=np.float32)
    X_norm = (X_smooth - x_min) / (x_max - x_min)
    return X_norm.astype(np.float32)


# ------------------------------- Model ----------------------------------- #

def build_improved_classification_model(input_dim: int, n_classes: int = N_CLASSES) -> tf.keras.Model:
    """
    1D CNN classifier. Architecture mirrors the user’s original design.

    Input shape: (T, 1), where T == input_dim.
    """
    inputs = Input(shape=(input_dim, 1))
    x = Conv1D(256, kernel_size=5, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = MaxPooling1D(pool_size=10, strides=2, padding="same")(x)

    x = Conv1D(128, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = MaxPooling1D(pool_size=10, strides=2, padding="same")(x)

    x = Conv1D(64, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = MaxPooling1D(pool_size=10, strides=2, padding="same")(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Maintain the original strong dropout as provided by the user.
    x = Dropout(0.9)(x)

    outputs = Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="SteelBall1D_CNN")
    return model


# ----------------------------- Visualization ----------------------------- #

def save_tsne_plot(emb: np.ndarray, labels: np.ndarray, title: str, path_png: str) -> None:
    """
    Save a scatter plot for t-SNE embeddings.

    Parameters
    ----------
    emb : (N, 2)
    labels : (N,)
    """
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="jet", s=12, alpha=0.85)
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close()


# --------------------------------- Main ---------------------------------- #

def main() -> None:
    """Run the full pipeline end-to-end."""
    ensure_output_dir(OUTPUT_DIR)

    # --- Load data -------------------------------------------------------- #
    data = load_dataframe(EXCEL_PATH)

    X_raw = data[RESISTANCE_COLS].values.astype(np.float32)   # (N, 4499)
    Y = data[LABEL_COLS].values.astype(np.float32)            # (N, 5), one-hot

    # --- Pre-processing --------------------------------------------------- #
    X = smooth_and_normalize(X_raw)                           # (N, 4499)

    # Train/val split (stratified via one-hot argmax).
    y_indices = np.argmax(Y, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_indices
    )

    # Keras expects (N, T, 1) for Conv1D.
    X_train_3d = np.expand_dims(X_train, axis=-1)             # (N, 4499, 1)
    X_val_3d   = np.expand_dims(X_val, axis=-1)               # (N, 4499, 1)

    # --- Model ------------------------------------------------------------ #
    model = build_improved_classification_model(input_dim=N_FEATURES, n_classes=N_CLASSES)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks: early stopping & best weights checkpoint.
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model_weights.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=False),
        ModelCheckpoint(filepath=ckpt_path, monitor="val_accuracy", save_best_only=True,
                        verbose=1, mode="max", save_weights_only=True)
    ]

    # --- Training --------------------------------------------------------- #
    history = model.fit(
        X_train_3d,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_3d, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # --- Metrics & Export ------------------------------------------------- #
    final_metrics_df = pd.DataFrame({
        "Loss": [history.history["loss"][-1]],
        "Accuracy": [history.history["accuracy"][-1]],
        "Val_Loss": [history.history["val_loss"][-1]],
        "Val_Accuracy": [history.history["val_accuracy"][-1]],
    })
    metrics_xlsx = os.path.join(OUTPUT_DIR, "final_metrics.xlsx")
    final_metrics_df.to_excel(metrics_xlsx, index=False)

    # --- t-SNE on original inputs (train split) --------------------------- #
    # Note: t-SNE is run on the preprocessed 2D features (not on CNN embeddings).
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    X_train_tsne = tsne.fit_transform(X_train)  # (N_train, 2)

    df_tsne_original = pd.DataFrame(X_train_tsne, columns=["tSNE1", "tSNE2"])
    df_tsne_original["Original_Label"] = np.argmax(y_train, axis=1).astype(int)
    original_tsne_xlsx = os.path.join(OUTPUT_DIR, "original_tsne_results.xlsx")
    df_tsne_original.to_excel(original_tsne_xlsx, index=False)

    original_tsne_png = os.path.join(OUTPUT_DIR, "original_tsne_plot.png")
    save_tsne_plot(
        emb=X_train_tsne,
        labels=df_tsne_original["Original_Label"].values,
        title="t-SNE Visualization — Original Data (train split)",
        path_png=original_tsne_png
    )

    # --- t-SNE colored by model predictions (on train split) -------------- #
    y_pred_train = model.predict(np.expand_dims(X_train, axis=-1), verbose=0)
    y_pred_labels = np.argmax(y_pred_train, axis=1)

    df_tsne_pred = pd.DataFrame(X_train_tsne, columns=["tSNE1", "tSNE2"])
    df_tsne_pred["Predicted_Label"] = y_pred_labels.astype(int)
    predicted_tsne_xlsx = os.path.join(OUTPUT_DIR, "predicted_tsne_results.xlsx")
    df_tsne_pred.to_excel(predicted_tsne_xlsx, index=False)

    predicted_tsne_png = os.path.join(OUTPUT_DIR, "predicted_tsne_plot.png")
    save_tsne_plot(
        emb=X_train_tsne,
        labels=df_tsne_pred["Predicted_Label"].values,
        title="t-SNE Visualization — Predicted Labels (train split)",
        path_png=predicted_tsne_png
    )

    # --- Console report --------------------------------------------------- #
    print(f"[OK] Final metrics saved to {metrics_xlsx}")
    print(f"[OK] Best weights saved to {ckpt_path}")
    print(f"[OK] Original t-SNE: {original_tsne_xlsx} / {original_tsne_png}")
    print(f"[OK] Predicted t-SNE: {predicted_tsne_xlsx} / {predicted_tsne_png}")


if __name__ == "__main__":
    main()

