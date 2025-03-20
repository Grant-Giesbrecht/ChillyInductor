import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz
import time
import os
import sys

f1 = 20
f2 = 20

t = np.linspace(0, 10, 1001)

v1 = np.sin(2*np.pi*f1*t)
v2 = np.sin(2*np.pi*f2*t + np.pi)

fig1 = plt.figure(1, figsize=(16, 9))
gs = fig1.add_gridspec(5, 1)

ax1a = fig1.add_subplot(gs[0:3, 0])
ax1b = fig1.add_subplot(gs[3, 0])
ax1c = fig1.add_subplot(gs[4, 0])

ax1a.plot(t, v2-v1, linestyle=':', marker='.', color=(0.65, 0, 0))
ax1a.set_xlabel("Time")
ax1a.set_ylabel("Amplitude, v2-v1")
ax1a.grid(True)

ax1b.plot(t, v1, linestyle=':', marker='.', color=(0.65, 0, 0))
ax1b.set_xlabel("Time")
ax1b.set_ylabel("Amplitude, v1")
ax1b.grid(True)

ax1c.plot(t, v2, linestyle=':', marker='.', color=(0.65, 0, 0))
ax1c.set_xlabel("Time")
ax1c.set_ylabel("Amplitude, v2")
ax1c.grid(True)

fig1.tight_layout()

plt.show()