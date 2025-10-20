#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout,
                                     BatchNormalization, concatenate, LSTM, Reshape)
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import resample

# ----------------------------------------------------
# 1) Load and preprocess data
# ----------------------------------------------------
data_path = 'E:/Force regeneration Data resistance.xlsx'
data = pd.read_excel(data_path, engine='openpyxl')

# Load resistance and force values
resistance_columns = [f'Resistance{i}' for i in range(1, 10001)]
force_columns = [f'Force{i}' for i in range(1, 10001)]

resistance_values = data[resistance_columns].values
force_values = pd.read_excel(
    'E:/Force regeneration Data force.xlsx',
    engine='openpyxl',
    usecols=force_columns
).values

# Downsample (e.g., 10,000 → 50 points)
target_num_points = 50
reshaped_values_50 = resample(resistance_values, target_num_points, axis=1)
force_values_50 = resample(force_values, target_num_points, axis=1)

# ----------------------------------------------------
# 2) Data augmentation — add BOTH Gaussian noise AND white noise
#    (white noise here is zero-mean uniform noise)
# ----------------------------------------------------
def augment_sequences(sequences, num_augmented, gaussian_std=0.01, white_amp=0.01):
    """
    Create augmented copies by adding two independent noise components:
      - Gaussian noise ~ N(0, gaussian_std^2)
      - White (uniform) noise ~ U(-white_amp, +white_amp)
    Args:
        sequences: np.ndarray, shape (N, T, 1) or (N, T)
        num_augmented: int, number of augmented samples per original sample
        gaussian_std: float, std of the Gaussian component
        white_amp: float, half-amplitude of the uniform component
    Returns:
        np.ndarray with shape (N * num_augmented, T, 1) or (N * num_augmented, T)
    """
    augmented = []
    for seq in sequences:
        for _ in range(num_augmented):
            g_noise = np.random.normal(loc=0.0, scale=gaussian_std, size=seq.shape)
            w_noise = np.random.uniform(low=-white_amp, high=white_amp, size=seq.shape)
            aug_seq = seq + g_noise + w_noise
            augmented.append(aug_seq)
    return np.asarray(augmented)

# ----------------------------------------------------
# 3) Build model
# ----------------------------------------------------
def build_model(input_shape, target_points):
    sequence_input = Input(shape=input_shape)

    # CNN stack over temporal axis
    x = Reshape((input_shape[0], 1, 1))(sequence_input)
    conv1 = Conv2D(64, (3, 1), activation='relu', padding='same')(x)
    conv2 = Conv2D(128, (3, 1), activation='relu', padding='same')(conv1)
    conv3 = MaxPooling2D((2, 1))(BatchNormalization()(conv2))

    # LSTM over downsampled temporal features
    lstm_in = Reshape((input_shape[0] // 2, -1))(conv3)
    lstm_out = LSTM(64, return_sequences=True)(lstm_in)   # 1st LSTM
    lstm_out = LSTM(128, return_sequences=True)(lstm_out) # 2nd LSTM
    lstm_out = LSTM(64, return_sequences=False)(lstm_out) # 3rd LSTM

    # Dense head
    dense1 = Dense(128, activation='relu')(lstm_out)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    final_output = Dense(target_points)(dropout2)

    model = Model(inputs=sequence_input, outputs=final_output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ----------------------------------------------------
# 4) 5-fold cross-validation (unchanged logic)
# ----------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics_list = []

num_augmented = 5      # per-sample augmentations
gaussian_std = 0.01    # std for Gaussian noise
white_amp = 0.01       # half-amplitude for uniform white noise

fold = 1
for train_index, val_index in kf.split(reshaped_values_50):
    print(f'\nStarting Fold {fold}')

    # Split
    X_train, X_val = reshaped_values_50[train_index], reshaped_values_50[val_index]
    y_train, y_val = force_values_50[train_index], force_values_50[val_index]

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]

    # Augmentation: BOTH Gaussian + white noise
    augmented_sequences = augment_sequences(
        X_train,
        num_augmented=num_augmented,
        gaussian_std=gaussian_std,
        white_amp=white_amp
    )
    augmented_labels = np.repeat(y_train, num_augmented, axis=0)

    # Concatenate original + augmented
    X_train_augmented = np.concatenate([X_train, augmented_sequences], axis=0)
    y_train_augmented = np.concatenate([y_train, augmented_labels], axis=0)

    # Build model
    model = build_model(input_shape=(target_num_points, 1), target_points=target_num_points)

    # Checkpoint
    checkpoint_filepath = f'/tmp/best_model_fold_{fold}.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_mae',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # Train
    history = model.fit(
        X_train_augmented, y_train_augmented,
        epochs=200,
        batch_size=64,
        callbacks=[model_checkpoint_callback],
        validation_data=(X_val, y_val),
        verbose=2
    )

    # Evaluate best checkpoint
    best_model = tf.keras.models.load_model(checkpoint_filepath)
    y_pred = best_model.predict(X_val, verbose=0)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    print(f'Fold {fold} -- MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R²: {r2}')
    metrics_list.append({'Fold': fold, 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

    fold += 1

# ----------------------------------------------------
# 5) Save metrics (unchanged)
# ----------------------------------------------------
metrics_df = pd.DataFrame(metrics_list)
metrics_summary = metrics_df.agg(['mean', 'std']).reset_index()
metrics_summary.rename(columns={'index': 'Metric'}, inplace=True)

with pd.ExcelWriter('cross_validation_metrics.xlsx') as writer:
    metrics_df.to_excel(writer, sheet_name='Fold Metrics', index=False)
    metrics_summary.to_excel(writer, sheet_name='Summary Metrics', index=False)

print("\nCross-validation metrics saved to 'cross_validation_metrics.xlsx'.")

