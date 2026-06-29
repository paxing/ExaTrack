# -*- coding: utf-8 -*-
"""
acquire_emg.py  —  Step 1 of 3

Records EMG data from two channels using the ADS1115 ADC on a Raspberry Pi,
following a predetermined gesture sequence with automatic labelling.
The output CSV is used as input for train_exatrack.py (Step 2).

Usage
─────
  python acquire_emg.py

The saved CSV path is printed at the end — copy it into train_exatrack.py.
"""

import time
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import board
import busio
import RPi.GPIO as GPIO
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ===========================================================================
# CONFIGURATION — edit here
# ===========================================================================
OUTPUT_DIR  = 'data/'           # folder where CSV files are saved (must exist)
FILE_PREFIX = 'training_acquisition'

# Gesture sequence: 0=rest, 1=flexion, -1=extension.
# Each entry is one event; NTILE windows are recorded per event.
LABEL_SEQ = [0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0,
             0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0]

NTILE       = 20    # windows recorded per event in LABEL_SEQ
WINDOW_SIZE = 50    # samples per window

# ===========================================================================
# 1.  HARDWARE SETUP
# ===========================================================================
i2c  = busio.I2C(board.SCL, board.SDA)
ads  = ADS.ADS1115(i2c, data_rate=860, mode=256)
chan1 = AnalogIn(ads, ADS.P0)
chan2 = AnalogIn(ads, ADS.P1)

GPIO.setup(12, GPIO.OUT)
GPIO.output(12, 0)
buzzer = GPIO.PWM(12, 10)

# ===========================================================================
# 2.  BUILD FULL LABEL VECTOR
#     Each event in LABEL_SEQ is repeated NTILE times so the label vector
#     aligns one-to-one with the acquired windows.
# ===========================================================================
label_array = np.asarray(LABEL_SEQ)[:, np.newaxis]
training_label = np.tile(label_array, (1, NTILE)).flatten()   # length = n_events * NTILE
n_windows = len(training_label)

print(f"Acquisition plan:")
print(f"  Events       : {len(LABEL_SEQ)}")
print(f"  Windows/event: {NTILE}")
print(f"  Total windows: {n_windows}  ({n_windows * WINDOW_SIZE} samples)")
print(f"  Window size  : {WINDOW_SIZE} samples")
print()

# ===========================================================================
# 3.  HELPER FUNCTIONS
# ===========================================================================
def create_csv(path, prefix):
    """Open a new timestamped CSV and write the header. Returns the file path."""
    ts        = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    file_name = f"{path}{prefix}_{ts}.csv"
    with open(file_name, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=['voltage1 (V)', 'voltage2 (V)']).writeheader()
    return file_name


def acquire_window(writer):
    """Acquire one window of WINDOW_SIZE samples and append to open CSV writer."""
    data1, data2 = [], []
    for _ in range(WINDOW_SIZE):
        v1, v2 = chan1.voltage, chan2.voltage
        data1.append(v1); data2.append(v2)
        writer.writerow([v1, v2])
    return np.array(data1), np.array(data2)


def acquire_dataset(file_name, labels):
    """Acquire all windows sequentially, printing the current gesture label."""
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        print("Starting acquisition — follow the gesture sequence below.")
        t_start = time.time()
        for lbl in labels:
            symbol = {1: '↑ FLEX', -1: '↓ EXTEND', 0: '── REST'}[lbl]
            print(f"  {symbol}")
            acquire_window(writer)
        duration = time.time() - t_start

    actual_fs = n_windows / duration
    print(f"\nDone. Duration: {duration:.1f}s  "
          f"Effective rate: {actual_fs:.1f} windows/s\n")


# ===========================================================================
# 4.  ACQUISITION
# ===========================================================================
file_name = create_csv(OUTPUT_DIR, FILE_PREFIX)
print(f"Saving to: {file_name}\n")

buzzer.start(50)
time.sleep(0.3)
buzzer.stop()

acquire_dataset(file_name, training_label)

buzzer.start(50)
time.sleep(0.3)
buzzer.stop()

# ===========================================================================
# 5.  VISUAL VALIDATION
#     Shows raw signals and the envelope aligned with the label sequence.
#     Press Enter to confirm quality; Ctrl-C to abort and re-acquire.
# ===========================================================================
data  = pd.read_csv(file_name)
chan1_raw = np.asarray(data['voltage1 (V)'])
chan2_raw = np.asarray(data['voltage2 (V)'])

# Per-window envelope (mean absolute deviation from channel mean)
env1 = np.abs(chan1_raw - np.mean(chan1_raw))
env2 = np.abs(chan2_raw - np.mean(chan2_raw))
env1_norm = np.mean(np.reshape(env1, (-1, WINDOW_SIZE)), axis=1)
env2_norm = np.mean(np.reshape(env2, (-1, WINDOW_SIZE)), axis=1)
env1_norm /= np.max(env1_norm) + 1e-8
env2_norm /= np.max(env2_norm) + 1e-8

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

axes[0].plot(chan1_raw, lw=0.4, label='CH1')
axes[0].plot(chan2_raw, lw=0.4, label='CH2')
axes[0].set_xlabel('Samples'); axes[0].set_ylabel('Voltage (V)')
axes[0].set_title('Raw EMG signal'); axes[0].legend()

axes[1].plot(env1_norm, lw=1.0, label='CH1 envelope')
axes[1].plot(env2_norm, lw=1.0, label='CH2 envelope')
axes[1].plot(training_label / max(abs(training_label.max()), 1),
             lw=1.5, color='red', label='Label')
axes[1].set_xlabel('Window'); axes[1].set_ylabel('Normalised amplitude')
axes[1].set_title('Per-window envelope vs label sequence'); axes[1].legend()

plt.tight_layout()
plt.show()

try:
    input('\nPress Enter to confirm this acquisition, or Ctrl-C to discard.')
except KeyboardInterrupt:
    print("\nAcquisition discarded. Re-run the script to try again.")
    sys.exit(0)

# ===========================================================================
# 6.  REPORT
# ===========================================================================
print(f"\nAcquisition saved:")
print(f"  {file_name}")
print(f"\nNext step: open train_exatrack.py and set")
print(f"  CSV_PATH = '{file_name}'")
print(f"then run:  python train_exatrack.py")
