# -*- coding: utf-8 -*-
"""
emg_live_scope.py — quick diagnostic, NOT part of the acquire/train/play pipeline.

Streams raw voltage from both EMG channels to a live-updating plot so you can
visually confirm electrode contact before running a full acquisition.

Usage
─────
  python emg_live_scope.py

  Flex hard a few times while watching the plot:
    - Good contact  → wide voltage swings (e.g. 0.5 V to 3 V) during contraction,
                       flat near baseline at rest.
    - Bad contact    → signal stays compressed in a narrow band (e.g. <0.1 V range)
                       even during a hard flex. This points to electrode/wiring,
                       not weak muscle signal.

  Ctrl+C to stop.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ===========================================================================
# CONFIGURATION
# ===========================================================================
BUFFER_SECONDS = 5      # how many seconds of signal to show on screen
FS_ESTIMATE    = 860    # approximate sampling rate (matches data_rate below)

# ===========================================================================
# HARDWARE SETUP
# ===========================================================================
i2c   = busio.I2C(board.SCL, board.SDA)
ads   = ADS.ADS1115(i2c, data_rate=860, mode=256)
chan1 = AnalogIn(ads, ADS.P0)
chan2 = AnalogIn(ads, ADS.P1)

# ===========================================================================
# ROLLING BUFFERS
# ===========================================================================
n_points = BUFFER_SECONDS * FS_ESTIMATE
buf1 = deque(maxlen=n_points)
buf2 = deque(maxlen=n_points)

# ===========================================================================
# LIVE PLOT
# ===========================================================================
plt.ion()
fig, ax = plt.subplots(figsize=(12, 5))
line1, = ax.plot([], [], lw=0.6, label='Channel 1')
line2, = ax.plot([], [], lw=0.6, label='Channel 2')
ax.set_xlabel('Samples (most recent)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Live EMG — flex hard and watch for swings. Ctrl+C to stop.')
ax.legend(loc='upper right')
ax.set_ylim(0, 3.3)   # full ADS1115 single-ended range for quick visual reference
fig.tight_layout()

print("Streaming... flex hard a few times. Ctrl+C to stop.")
t_start = time.time()
n_acquired = 0

try:
    while True:
        v1, v2 = chan1.voltage, chan2.voltage
        buf1.append(v1)
        buf2.append(v2)
        n_acquired += 1

        # Update plot every ~50 samples to keep it responsive without
        # redrawing on every single sample (which would be too slow).
        if n_acquired % 50 == 0:
            x = np.arange(len(buf1))
            line1.set_data(x, buf1)
            line2.set_data(x, buf2)
            ax.set_xlim(0, max(len(buf1), 1))

            # Auto-report swing size over the current buffer so you don't
            # have to eyeball it — this is the actual diagnostic number.
            if len(buf1) > 10:
                swing1 = max(buf1) - min(buf1)
                swing2 = max(buf2) - min(buf2)
                ax.set_title(
                    f'Live EMG — CH1 swing: {swing1:.3f} V   '
                    f'CH2 swing: {swing2:.3f} V   (Ctrl+C to stop)')

            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    elapsed   = time.time() - t_start
    actual_fs = n_acquired / elapsed
    print(f"\nStopped. {n_acquired} samples in {elapsed:.1f}s "
          f"({actual_fs:.0f} sps actual rate)")
    if len(buf1) > 10:
        swing1 = max(buf1) - min(buf1)
        swing2 = max(buf2) - min(buf2)
        print(f"Final buffer swing — CH1: {swing1:.3f} V   CH2: {swing2:.3f} V")
        if swing1 < 0.2 and swing2 < 0.2:
            print("\n⚠ Both channels show <0.2V swing even after flexing.")
            print("  This points to electrode contact / wiring, not weak muscle signal.")
            print("  Check: gel freshness, skin prep, lead connections, reference electrode.")
