# -*- coding: utf-8 -*-

"""
---------------------
STM32 emits lines of the form:
    `Sample <idx>, Channel 1: <adc1>, Channel 2: <adc2>`

Example:
    `Sample 137, Channel 1: 3021, Channel 2: 1983`

The application ingests 2,500 samples per channel per inference window
(total 5,000 points), then performs end-to-end processing and classifies
into one of {1, 2, 3}. The resulting class id can be written back to the
STM32 as a single ASCII digit (`'1'|'2'|'3'`).

Reproducibility & Environment
-----------------------------
- Python 3.9+
- Packages: numpy, tensorflow (keras), pyqt5, pyqtgraph, scipy, pyserial
- GPU acceleration (optional) via TensorFlow if available.

"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import List, Optional

import numpy as np
import serial
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras import layers, models

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QSizePolicy,
    QProgressBar,
)

import pyqtgraph as pg


# ============================================================================
# Configuration
# ============================================================================

# --- Serial I/O ---
SERIAL_PORT: str = "COM5"          # Adjust to your STM32 port
BAUD_RATE: int = 921600            # Must match STM32 firmware

# --- Output assets (class images, model weights, logs) ---
OUTPUT_DIR: str = "C:/Users/wuqiushuo/Desktop/20250116/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Preprocessing (Savitzky–Golay) ---
SG_WINDOW: int = 51                # Must be odd
SG_POLY: int = 3

# --- ADC parameters ---
ADC_RESOLUTION: int = 4095         # 12-bit: 0..4095
VREF: float = 5.0                  # Reference voltage (V)

# --- Model / windowing ---
SAMPLES_PER_CHANNEL: int = 2500    # per inference window
N_CHANNELS: int = 2
TOTAL_SAMPLES: int = SAMPLES_PER_CHANNEL * N_CHANNELS  # 5000
MODEL_INPUT_LENGTH: int = 4999     # training-time input length (compat.)
N_CLASSES: int = 3

# --- Weight file (must match model architecture) ---
WEIGHTS_PATH: str = os.path.join(OUTPUT_DIR, "best_model_weights.h5")

# --- Regex for parsing STM32 lines ---
LINE_REGEX = re.compile(
    r"^Sample\s+(?P<idx>\d+),\s*Channel\s+1:\s*(?P<c1>\d+),\s*Channel\s+2:\s*(?P<c2>\d+)\s*$"
)


# ============================================================================
# Model definition (must mirror training)
# ============================================================================

def build_classification_model(input_dim: int, l2_value: float = 0.001) -> tf.keras.Model:
    """
    1-D CNN classifier mirroring the training architecture.

    Parameters
    ----------
    input_dim : int
        Temporal length of the 1-D sequence.
    l2_value : float
        (Unused placeholder for parity with prior signatures.)

    Returns
    -------
    tf.keras.Model
        Compiled Keras model (without weights).
    """
    inp = layers.Input(shape=(input_dim, 1), name="input_layer")

    x = layers.Conv1D(256, kernel_size=3, strides=2, padding="same", name="conv1d_1")(inp)
    x = layers.BatchNormalization(name="batchnorm_1")(x)
    x = layers.Activation("relu", name="activation_1")(x)
    x = layers.MaxPooling1D(pool_size=10, strides=2, padding="same", name="maxpool_1")(x)

    x = layers.Conv1D(128, kernel_size=4, strides=2, padding="same", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="batchnorm_2")(x)
    x = layers.Activation("relu", name="activation_2")(x)
    x = layers.MaxPooling1D(pool_size=10, strides=2, padding="same", name="maxpool_2")(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same", name="conv1d_3")(x)
    x = layers.BatchNormalization(name="batchnorm_3")(x)
    x = layers.Activation("relu", name="activation_3")(x)
    x = layers.MaxPooling1D(pool_size=10, strides=2, padding="same", name="maxpool_3")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, name="dense_1")(x)
    x = layers.BatchNormalization(name="batchnorm_4")(x)
    x = layers.Activation("relu", name="activation_4")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    out = layers.Dense(N_CLASSES, activation="softmax", name="output")(x)
    return models.Model(inputs=inp, outputs=out, name="stm32_stream_cnn")


# Instantiate and load weights (fail fast if unavailable)
model = build_classification_model(MODEL_INPUT_LENGTH)
print("Model summary:")
model.summary(line_length=120)

if os.path.exists(WEIGHTS_PATH):
    try:
        model.load_weights(WEIGHTS_PATH)
        print(f"Loaded weights from: {WEIGHTS_PATH}")
    except Exception as exc:
        print(f"Error loading weights: {exc}")
        sys.exit(1)
else:
    print(f"Weights not found at: {WEIGHTS_PATH}")
    sys.exit(1)


# ============================================================================
# Utility functions
# ============================================================================

def adc_to_volts(adc_val: int) -> float:
    """Convert raw 12-bit ADC reading to voltage."""
    return float(adc_val) / ADC_RESOLUTION * VREF


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min–max normalize an array to [0, 1] with safe guards."""
    a_min = arr.min()
    a_max = arr.max()
    if a_max <= a_min:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)


# ============================================================================
# Qt Main Window
# ============================================================================

class MainWindow(QMainWindow):
    """
    PyQt5 application for real-time STM32 serial streaming, denoising,
    CNN inference, and visual feedback.
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Window & theme ---
        self.setWindowTitle("STM32 Streaming: Denoising + CNN Classification")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #2C3E50;")  # dark theme

        # --- Central layout ---
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- Class image placeholder ---
        self.image_label = QLabel("Prediction image will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(
            "background-color: #2C3E50; color: #ECF0F1; font-size: 16px;"
        )

        # --- Status / prediction text ---
        self.text_label = QLabel("Status: idle")
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("font-size: 18px; color: #3498DB;")

        # --- Progress bar ---
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar {background-color: #E0E0E0; border: 1px solid grey; border-radius: 5px;}"
            "QProgressBar::chunk {background-color: #4CAF50; width: 10px;}"
        )

        # --- Control button ---
        self.start_button = QPushButton("Start streaming", self)
        self.start_button.setStyleSheet(
            "QPushButton {background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; border-radius: 5px;}"
            "QPushButton:hover {background-color: #45A049;}"
        )
        self.start_button.clicked.connect(self.start_reading)

        # --- Assemble top widgets ---
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.start_button)

        # --- Plotting widgets (pyqtgraph) ---
        self.plot_widget = pg.GraphicsLayoutWidget(self)
        self.layout.addWidget(self.plot_widget)

        # Channel 1
        self.channel1_plot = self.plot_widget.addPlot(title="Channel 1 (V)")
        self.channel1_curve = self.channel1_plot.plot(pen=pg.mkPen("b", width=2))
        self.channel1_plot.setLabel("left", "Voltage (V)")
        self.channel1_plot.setLabel("bottom", "Sample")

        # Channel 2
        self.plot_widget.nextRow()
        self.channel2_plot = self.plot_widget.addPlot(title="Channel 2 (V)")
        self.channel2_curve = self.channel2_plot.plot(pen=pg.mkPen("r", width=2))
        self.channel2_plot.setLabel("left", "Voltage (V)")
        self.channel2_plot.setLabel("bottom", "Sample")

        # --- Serial init ---
        self.serial: Optional[serial.Serial] = None
        try:
            self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            self.text_label.setText(f"Serial port {SERIAL_PORT} opened successfully.")
        except Exception as exc:
            self.text_label.setText(f"Serial port error: {exc}")
            self.serial = None

        # --- Timer for polling serial ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial_data)

        # --- Data buffers (per window) ---
        self.channel1_data: List[float] = []
        self.channel2_data: List[float] = []

        # --- Inference result ---
        self.predicted_category: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #

    def start_reading(self) -> None:
        """Begin periodic serial polling if the port is open."""
        if self.serial and self.serial.is_open:
            self.text_label.setText("Reading data from serial port…")
            # Poll at 1 ms; adjust as necessary for your host/firmware throughput
            self.timer.start(1)
        else:
            self.text_label.setText("Error: serial port not open.")

    # ------------------------------------------------------------------ #
    # Serial handling
    # ------------------------------------------------------------------ #

    def read_serial_data(self) -> None:
        """Read one line from the UART, parse, convert, and collect."""
        try:
            if not (self.serial and self.serial.is_open):
                self.text_label.setText("Error: serial port not open.")
                return

            raw = self.serial.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                return

            m = LINE_REGEX.match(raw)
            if not m:
                return  # silently ignore malformed lines

            # Parse integers
            _ = int(m.group("idx"))  # sample index (unused for now)
            c1_raw = int(m.group("c1"))
            c2_raw = int(m.group("c2"))

            # Raw ADC -> volts
            c1_v = adc_to_volts(c1_raw)
            c2_v = adc_to_volts(c2_raw)

            # Buffer the frame
            self.channel1_data.append(c1_v)
            self.channel2_data.append(c2_v)

            # Trigger processing when we have a full window
            if len(self.channel1_data) >= SAMPLES_PER_CHANNEL and len(self.channel2_data) >= SAMPLES_PER_CHANNEL:
                self.process_window()

        except Exception as exc:
            self.text_label.setText(f"Read error: {exc}")

    def send_signal_to_stm32(self, signal: int) -> None:
        """Send a single ASCII digit [1|2|3] to the STM32 as feedback."""
        if not (self.serial and self.serial.is_open):
            self.text_label.setText("Error: serial port not open.")
            return
        if signal not in (1, 2, 3):
            self.text_label.setText("Error: invalid feedback value.")
            return
        try:
            self.serial.write(str(signal).encode("utf-8"))
        except Exception as exc:
            self.text_label.setText(f"Write error: {exc}")

    # ------------------------------------------------------------------ #
    # Processing & inference
    # ------------------------------------------------------------------ #

    def process_window(self) -> None:
        """
        Process the latest 2×N window:
        SG denoising → min–max normalization → length adjustment →
        CNN inference → feedback + UI update.
        """
        t0 = time.time()
        self.progress_bar.setValue(0)

        # 1) Extract exactly N samples per channel (drop extras if any)
        ch1 = np.asarray(self.channel1_data[:SAMPLES_PER_CHANNEL], dtype=np.float32)
        ch2 = np.asarray(self.channel2_data[:SAMPLES_PER_CHANNEL], dtype=np.float32)

        # Trim buffers (leave any extra for the next window)
        self.channel1_data = self.channel1_data[SAMPLES_PER_CHANNEL:]
        self.channel2_data = self.channel2_data[SAMPLES_PER_CHANNEL:]

        # 2) Savitzky–Golay denoising (channel-wise)
        ch1_sg = savgol_filter(ch1, window_length=SG_WINDOW, polyorder=SG_POLY, mode="interp")
        ch2_sg = savgol_filter(ch2, window_length=SG_WINDOW, polyorder=SG_POLY, mode="interp")
        self.progress_bar.setValue(25)

        # 3) Stack and normalize over all points jointly (2×N → (2N,))
        stacked = np.vstack([ch1_sg, ch2_sg])
        stacked = np.apply_along_axis(
            lambda row: savgol_filter(row, window_length=SG_WINDOW, polyorder=SG_POLY, mode="interp"),
            axis=1,
            arr=stacked,
        )
        self.progress_bar.setValue(50)

        flat = stacked.flatten()  # length = 5000
        flat_norm = minmax_normalize(flat)
        self.progress_bar.setValue(70)

        # 4) Model input shaping: 5000 → 4999 to match training
        if flat_norm.size == TOTAL_SAMPLES:
            # Drop one element (random index) to match the training input length
            drop_idx = np.random.randint(0, flat_norm.size)
            flat_norm = np.delete(flat_norm, drop_idx)

        x = flat_norm.reshape(1, MODEL_INPUT_LENGTH, 1).astype(np.float32)

        # 5) Inference
        pred = model.predict(x, verbose=0)
        pred_class = int(np.argmax(pred, axis=1)[0])  # 0-based
        self.predicted_category = pred_class + 1      # map to {1,2,3}
        self.progress_bar.setValue(90)

        # 6) Feedback to STM32
        self.send_signal_to_stm32(self.predicted_category)

        # 7) Plot update (denoised channels)
        self.channel1_curve.setData(ch1_sg)
        self.channel2_curve.setData(ch2_sg)
        self.progress_bar.setValue(100)

        # 8) Status + image
        dt = time.time() - t0
        self.text_label.setText(
            f"Predicted class: {self.predicted_category} | Inference window: {SAMPLES_PER_CHANNEL} samples/channel "
            f"| Runtime: {dt:.2f}s"
        )
        self.display_prediction_image()

    # ------------------------------------------------------------------ #
    # UI helpers
    # ------------------------------------------------------------------ #

    def display_prediction_image(self) -> None:
        """Load and display the class-specific image if available."""
        if self.predicted_category is None:
            self.text_label.setText("No prediction to display.")
            return

        image_path = os.path.join(OUTPUT_DIR, f"{self.predicted_category}.png")
        if not os.path.exists(image_path):
            self.text_label.setText(
                f"Prediction image not found for class {self.predicted_category}: {image_path}"
            )
            # Keep previous pixmap (if any)
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.text_label.setText(f"Failed to load image: {image_path}")
            return

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        # Keep status line as-is (already set in process_window)


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
