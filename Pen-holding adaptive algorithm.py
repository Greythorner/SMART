#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from scipy.signal import savgol_filter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -------------------------------
# Output directory
# -------------------------------
output_dir = 'C:/Users/wuqiushuo/Desktop/20250116/'
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Read dataset from Excel
# -------------------------------
data_path = f'{output_dir}Pen-holding adaptive algorithm data.xlsx'
data = pd.read_excel(data_path, engine='openpyxl')

# Resistance sequence columns (length 4,999 to match model input)
resistance_columns = [f'Resistance{i}' for i in range(1, 5000)]
resistance_values = data[resistance_columns].values.astype(np.float32)

# One-hot class labels (3 classes)
class_labels_columns = [f'class_label{i}' for i in range(1, 4)]
class_labels = data[class_labels_columns].values.astype(np.float32)

# =====================================================
# Preprocessing
# =====================================================
# Savitzky–Golay smoothing along each sequence
window_size = 51   # must be odd and < 4999
poly_order  = 3
smoothed_values = np.apply_along_axis(
    lambda row: savgol_filter(row, window_length=window_size, polyorder=poly_order),
    axis=1,
    arr=resistance_values
).astype(np.float32)

# Global min–max normalization (to [0,1])
global_min = smoothed_values.min()
global_max = smoothed_values.max()
normalized_values = (smoothed_values - global_min) / (global_max - global_min + 1e-12)
normalized_values = normalized_values.astype(np.float32)

# =====================================================
# Train / validation split
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(
    normalized_values,
    class_labels,
    test_size=0.2,
    random_state=57,
    stratify=np.argmax(class_labels, axis=1)
)

# Add channel dimension for Conv1D
X_train_cnn = X_train[..., np.newaxis]  # (N, 4999, 1)
X_val_cnn   = X_val[..., np.newaxis]    # (N, 4999, 1)

# =====================================================
# Model (unchanged architecture)
# =====================================================
def build_improved_classification_model(input_dim, l2_value=0.001):
    input_layer = Input(shape=(input_dim, 1))
    x = Conv1D(256, kernel_size=3, strides=2, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=10, strides=2, padding='same')(x)

    x = Conv1D(128, kernel_size=4, strides=2, padding='same')(x)
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
    x = Dropout(0.5)(x)

    output = Dense(3, activation='softmax')(x)
    return models.Model(inputs=input_layer, outputs=output)

model = build_improved_classification_model(4999)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# =====================================================
# Callbacks (unchanged)
# =====================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=False),
    ModelCheckpoint(
        filepath=f'{output_dir}best_model_weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max',
        save_weights_only=True
    )
]

# =====================================================
# Training (unchanged)
# =====================================================
history = model.fit(
    X_train_cnn,
    y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_val_cnn, y_val),
    callbacks=callbacks,
    verbose=2
)

# Save final metrics
final_metrics_df = pd.DataFrame({
    'Loss': [history.history['loss'][-1]],
    'Accuracy': [history.history['accuracy'][-1]],
    'Val_Loss': [history.history['val_loss'][-1]],
    'Val_Accuracy': [history.history['val_accuracy'][-1]]
})
metrics_filename = f'{output_dir}final_metrics.xlsx'
final_metrics_df.to_excel(metrics_filename, index=False)

# =====================================================
# t-SNE (unchanged; operates on 2D features)
# =====================================================
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_train_tsne = tsne.fit_transform(X_train)

# Original-label t-SNE export/plot
df_tsne_original = pd.DataFrame(X_train_tsne, columns=['tSNE1', 'tSNE2'])
df_tsne_original['Original_Label'] = np.argmax(y_train, axis=1)
original_tsne_filename = f'{output_dir}original_tsne_results.xlsx'
df_tsne_original.to_excel(original_tsne_filename, index=False)

plt.figure(figsize=(10, 6))
scatter_original = plt.scatter(df_tsne_original['tSNE1'], df_tsne_original['tSNE2'],
                               c=df_tsne_original['Original_Label'], cmap='jet')
plt.colorbar(scatter_original)
plt.title('t-SNE Visualization - Original Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig(f'{output_dir}original_tsne_plot.png', dpi=200, bbox_inches='tight')
plt.close()

# Predicted-label t-SNE export/plot
y_pred = model.predict(X_train_cnn, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)

df_tsne_predicted = pd.DataFrame(X_train_tsne, columns=['tSNE1', 'tSNE2'])
df_tsne_predicted['Predicted_Label'] = y_pred_labels
predicted_tsne_filename = f'{output_dir}predicted_tsne_results.xlsx'
df_tsne_predicted.to_excel(predicted_tsne_filename, index=False)

plt.figure(figsize=(10, 6))
scatter_pred = plt.scatter(df_tsne_predicted['tSNE1'], df_tsne_predicted['tSNE2'],
                           c=df_tsne_predicted['Predicted_Label'], cmap='jet')
plt.colorbar(scatter_pred)
plt.title('t-SNE Visualization - Predicted Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig(f'{output_dir}predicted_tsne_plot.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------
# Output logs
# -------------------------------
print(f"Final metrics saved to {metrics_filename}")
print(f"Original data t-SNE results saved to {original_tsne_filename}")
print(f"Original data t-SNE plot saved to {output_dir}original_tsne_plot.png")
print(f"Predicted data t-SNE results saved to {predicted_tsne_filename}")
print(f"Predicted data t-SNE plot saved to {output_dir}predicted_tsne_plot.png")


# In[ ]:




