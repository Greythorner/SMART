# -*- coding: utf-8 -*-
"""


Safety & Notes
--------------
- Motion occurs immediately after pressing “Run Full Procedure”. Keep clear of moving parts.
- The code assumes RSE terminal configuration on AI0 and AI2 at ±10 V.
- The logic favors code clarity and traceability over micro-optimizations; the
  experimental protocol (command → move → record → localize → back-off → notify → save)
  is preserved verbatim from the legacy implementation.

Author / Affiliation
--------------------

"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import serial
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Matplotlib for in-GUI plotting
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# NI-DAQmx (requires NI drivers)
import nidaqmx
import nidaqmx.constants
from nidaqmx.system import Device


# =============================== Parameters =============================== #

# --- Serial endpoints (adapt to your setup) ---
MOTOR_COM_PORT = "COM10"   # Stepper motor controller
STM32_COM_PORT = "COM13"   # STM32 notification endpoint

# --- DAQ configuration ---
DAQ_DEVICE_NAME = "Dev5"
DAQ_CHANNELS = ("ai0", "ai2")                        # Two analog inputs
DAQ_TERMINAL_CFG = nidaqmx.constants.TerminalConfiguration.RSE
DAQ_SAMPLE_RATE_HZ = 1000

# --- Motion & timing ---
STEPS_PER_MM = 1000          # Controller microstep multiplier (as used in legacy code)
SPEED_DEFAULT_STEPS_PER_S = 50
DIST_DEFAULT_MM = 6.0
MOVE_TIME_CORRECTION = 0.03   # Empirical factor retained from original code

# --- Plot appearance ---
FIG_SIZE = (10, 4)
FIG_DPI = 100


# ================================ Serial I/O ============================== #

class SerialController:
    """
    Minimal, robust wrapper around a serial endpoint with ASCII command protocol.

    Parameters
    ----------
    port : str
        COM port (e.g., 'COM10').
    baudrate : int
        Baud rate; default 9600.
    timeout : float
        Read timeout in seconds.

    Notes
    -----
    - Commands are written with trailing '\\r' (carriage return), mirroring legacy
      NC/PLC protocols. Data writes use raw ASCII (no terminator).
    - Failures do not raise to the GUI thread; they are surfaced via message boxes
      during construction and via returned error strings afterward.
    """

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0) -> None:
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(0.3)  # allow controller to settle
        except serial.SerialException as e:
            messagebox.showerror("Serial Error", f"Cannot open serial port {port}: {e}")
            self.ser = None

    def send_command(self, cmd: str) -> str:
        """
        Send a line-based command with trailing CR and return all available response.

        Returns
        -------
        str
            Response decoded as ASCII (errors ignored); or error message.
        """
        if not self.ser:
            return "Serial not open"
        try:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + "\r").encode("ascii"))
            time.sleep(0.05)
            resp = self.ser.read_all().decode("ascii", errors="ignore").strip()
            return resp
        except Exception as e:
            return f"Error: {e}"

    def send_data(self, data: str) -> str:
        """
        Send raw ASCII data (no automatic terminator).

        Returns
        -------
        str
            "OK" on success or an error string.
        """
        if not self.ser:
            return "Serial not open"
        try:
            self.ser.write(data.encode("ascii"))
            return "OK"
        except Exception as e:
            return f"Send failed: {e}"

    def close(self) -> None:
        """Close underlying serial port if open."""
        if self.ser:
            self.ser.close()


# =============================== Main GUI App ============================= #

class MotorControlApp:
    """
    Tkinter application orchestrating the experimental protocol:

    Steps
    -----
    1) Home stage (HZ0).
    2) Set speed (V<steps/s>).
    3) Synchronous acquisition & forward motion (Z+<steps>).
    4) Localize extremum across AI0/AI2; compute depth.
    5) Back off to extremum depth (Z-<steps>).
    6) Notify STM32 ("REACH\\n").
    7) Optional: export waveforms/time to Excel.

    Design Choices
    --------------
    - All long-running actions execute in a daemon thread to keep the GUI responsive.
    - The NI-DAQ task is configured as continuous acquisition but stopped after motion
      completes; a fixed number of samples is then read.
    - The “move time” prediction uses the empirical correction MOVE_TIME_CORRECTION to
      maintain behavioral parity with legacy runs.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Z-axis Stepper + NI USB-6009 Acquisition")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        # Serial endpoints (opened at startup; failures are shown via message box)
        self.motor_ctl = SerialController(port=MOTOR_COM_PORT)
        self.stm32_ctl = SerialController(port=STM32_COM_PORT)

        # Global font
        default_font = ("Segoe UI", 11)
        root.option_add("*Font", default_font)

        # ------------- Control strip -------------
        frm = ttk.Frame(root)
        frm.pack(padx=20, pady=10, fill="x")

        ttk.Label(frm, text="Travel (mm):").grid(row=0, column=0, sticky="e")
        self.dist_entry = ttk.Entry(frm, width=8)
        self.dist_entry.insert(0, f"{DIST_DEFAULT_MM:g}")
        self.dist_entry.grid(row=0, column=1, padx=10)

        ttk.Label(frm, text="Speed (steps/s):").grid(row=0, column=2, sticky="e")
        self.spd_entry = ttk.Entry(frm, width=8)
        self.spd_entry.insert(0, f"{SPEED_DEFAULT_STEPS_PER_S:d}")
        self.spd_entry.grid(row=0, column=3, padx=10)

        self.btn_run = ttk.Button(root, text="🚀 Run Full Procedure", command=self.run_full_process_thread)
        self.btn_run.pack(fill="x", padx=20, pady=10, ipady=6)

        # ------------- Progress bar -------------
        self.progress = ttk.Progressbar(root, length=400, mode="determinate")
        self.progress.pack(padx=20, pady=(0, 10), fill="x")

        # ------------- Plot panel -------------
        plot_frame = ttk.LabelFrame(root, text="📈 Waveform")
        plot_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.fig = Figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ------------- Log panel -------------
        log_frame = ttk.LabelFrame(root, text="📋 Log")
        log_frame.pack(fill="both", expand=False, padx=20, pady=10)

        self.log = tk.Text(log_frame, height=10, bg="#f0f0f0", font=("Courier New", 10))
        self.log.pack(fill="both", expand=True)

        # Clean shutdown
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------- Small UI helpers ----------------------- #

    def log_msg(self, msg: str) -> None:
        """Append a line to the log textbox; auto-scroll."""
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def update_progress(self, value: float) -> None:
        """Set progress bar to `value` (0–100)."""
        self.progress["value"] = value
        self.root.update_idletasks()

    # ------------------------ Thread launcher ----------------------- #

    def run_full_process_thread(self) -> None:
        """Launch the full protocol in a background daemon thread."""
        threading.Thread(target=self.run_full_process, daemon=True).start()

    # ------------------------- Core Protocol ------------------------ #

    def run_full_process(self) -> None:
        """Execute the full acquisition & motion protocol; GUI-safe logging."""
        self.update_progress(0)

        # --- Parse inputs ---
        try:
            distance_mm = float(self.dist_entry.get())
            speed = float(self.spd_entry.get())
            if speed <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Input Error", "Please check your inputs; speed must be > 0.")
            return

        total_steps = max(1, int(distance_mm * STEPS_PER_MM))
        sample_rate = DAQ_SAMPLE_RATE_HZ

        # Empirically retained factor from legacy code
        move_time = total_steps / speed * MOVE_TIME_CORRECTION

        # --- 1) Home stage ---
        self.log_msg("1) Homing...")
        resp = self.motor_ctl.send_command("HZ0")
        self.log_msg(f">>> HZ0 | {resp}")
        time.sleep(2)  # allow homing to complete
        self.update_progress(10)

        # --- 2) Set speed ---
        self.log_msg(f"2) Set speed: V{int(speed)}")
        resp = self.motor_ctl.send_command(f"V{int(speed)}")
        self.log_msg(f">>> V{int(speed)} | {resp}")
        self.update_progress(20)

        # --- 3) Acquisition + forward motion ---
        self.log_msg(f"3) Acquiring + Move: Z+{total_steps} steps")

        try:
            # Reset device to a clean state (preserves parity with original code path)
            Device(DAQ_DEVICE_NAME).reset_device()

            with nidaqmx.Task() as task:
                # Two AI channels in RSE mode
                for ch in DAQ_CHANNELS:
                    task.ai_channels.add_ai_voltage_chan(
                        f"{DAQ_DEVICE_NAME}/{ch}",
                        terminal_config=DAQ_TERMINAL_CFG
                    )
                task.timing.cfg_samp_clk_timing(
                    rate=sample_rate,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
                )

                task.start()
                self.motor_ctl.send_command(f"Z+{total_steps}")

                self.log_msg(f"Waiting for motion to complete: ~{move_time:.2f}s + 1.0s buffer")
                time.sleep(move_time + 1.0)

                task.stop()
                num_samples_to_read = int(sample_rate * (move_time + 1.0))
                samples = task.read(number_of_samples_per_channel=num_samples_to_read)

        except Exception as e:
            self.log_msg(f"❌ Acquisition failed: {e}")
            self.update_progress(0)
            return

        # Convert to arrays & build time base
        data1 = np.array(samples[0])
        data2 = np.array(samples[1])
        actual_duration = len(data1) / sample_rate
        times = np.linspace(0, actual_duration, len(data1), dtype=float)
        self.update_progress(50)

        self.log_msg("✔ Acquisition complete.")
        self.log_msg(f"⏱ Actual acquisition time: {actual_duration:.3f} s")
        self.log_msg(f"🧮 Number of samples: {len(data1)}")

        # --- 4) Extremum localization across channels ---
        idx1 = int(np.argmax(np.abs(data1)))
        idx2 = int(np.argmax(np.abs(data2)))
        t1, v1 = times[idx1], float(data1[idx1])
        t2, v2 = times[idx2], float(data2[idx2])

        # Choose the channel with the larger absolute extremum
        critical_time = t1 if abs(v1) >= abs(v2) else t2
        critical_dist_mm = (critical_time / actual_duration) * distance_mm if actual_duration > 0 else 0.0

        self.log_msg(f"🔎 AI0 extremum {v1:.3f} V @ idx={idx1}")
        self.log_msg(f"🔎 AI2 extremum {v2:.3f} V @ idx={idx2}")
        self.log_msg(f"➡  time of strongest response ≈ {critical_time:.3f} s")
        self.log_msg(f"➡  depth at strongest response ≈ {critical_dist_mm:.2f} mm")

        # --- Plot waveforms & annotate extremums ---
        self.ax.clear()
        self.ax.plot(times, data1, label="AI0", linewidth=1.5)
        self.ax.plot(times, data2, label="AI2", linewidth=1.5)

        # Mark AI0 extremum
        self.ax.plot(t1, v1, "ro")
        self.ax.annotate(
            f"AI0 Max\n{v1:.2f} V",
            xy=(t1, v1), xytext=(t1 + 0.1, v1),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=10, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red")
        )

        # Mark AI2 extremum
        self.ax.plot(t2, v2, "ro")
        self.ax.annotate(
            f"AI2 Max\n{v2:.2f} V",
            xy=(t2, v2), xytext=(t2 + 0.1, v2),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=10, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red")
        )

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.set_title("Full Acquisition Trace")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()
        self.update_progress(70)

        # --- 5) Back off to estimated extremum depth ---
        back_steps = total_steps - int(critical_dist_mm * STEPS_PER_MM)
        if back_steps > 0:
            self.log_msg(f"4) Back off: Z-{back_steps}")
            resp = self.motor_ctl.send_command(f"Z-{back_steps}")
            self.log_msg(f">>> Z-{back_steps} | {resp}")
            # Nominal wait: distance/speed (no correction factor used for return)
            time.sleep(max(0.0, back_steps / speed))
        else:
            self.log_msg("⚠ Already near the extremum depth; skipping back-off.")

        self.update_progress(85)

        # --- 6) Notify STM32 ---
        self.log_msg("5) Notify STM32: REACH")
        resp = self.stm32_ctl.send_data("REACH\n")
        self.log_msg(f"STM32 response: {resp}")
        self.update_progress(90)

        # --- 7) Optional: save to Excel ---
        df = pd.DataFrame({"Time (s)": times, "AI0 (V)": data1, "AI2 (V)": data2})
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
            title="Save data file",
            initialfile=f"acquisition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if file_path:
            try:
                df.to_excel(file_path, index=False)
                self.log_msg(f"✅ Data saved: {file_path}")
            except Exception as e:
                self.log_msg(f"⚠ Save failed: {e}")

        self.update_progress(100)
        time.sleep(0.5)
        self.update_progress(0)

    # ----------------------------- Shutdown ----------------------------- #

    def on_close(self) -> None:
        """Ensure both serial links are closed before destroying the window."""
        self.motor_ctl.close()
        self.stm32_ctl.close()
        self.root.destroy()


# ============================== Entrypoint ================================ #

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    app = MotorControlApp(root)
    root.mainloop()
