import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz

DATADIR = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Time Domain Measurements"

pi = 3.1415926535
merge = True
include_3rd = False

trim_time = True
use_lpfilter = True

# dfA1 = pd.read_csv(f"{DATADIR}/C1NOBIAS-F4,8GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')
dfB1 = pd.read_csv(f"{DATADIR}/C10,1VBIAS-F2,4GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')
dfB2 = pd.read_csv(f"{DATADIR}/C20,1VBIAS-F2,4GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')

dfC1 = pd.read_csv(f"{DATADIR}/C1NOBIAS-F4,8GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')
# dfD1 = pd.read_csv(f"{DATADIR}/C10,275VBIAS-F2,4GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')
# dfD2 = pd.read_csv(f"{DATADIR}/C20,275VBIAS-F2,4GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')

def find_closest_index(lst, X):
	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
	return closest_index

red = (0.6, 0, 0)
blue = (0, 0, 0.7)
green = (0, 0.5, 0)
orange = (1, 0.5, 0.05)

#====================== Select Data to Work with =================

time_ns_full = list(dfC1['Time']*1e9)
ampl_mV_full = list(dfC1['Ampl']*1e3)
supertitle = "No Doubler"
pulse_range_ns = (10, 48)

time_ns_full = list(dfB1['Time']*1e9)
ampl_mV_full = list(dfB1['Ampl']*1e3)
if include_3rd:
	ampl_mV_full = ampl_mV_full + dfB2['Ampl']*1e3
supertitle = "Using Doubler"
pulse_range_ns = (20, 40)

# Trim timeseries
if trim_time:
	idx_start = find_closest_index(time_ns_full, pulse_range_ns[0])
	idx_end = find_closest_index(time_ns_full, pulse_range_ns[1])
	time_ns = time_ns_full[idx_start:idx_end+1]
	ampl_mV = ampl_mV_full[idx_start:idx_end+1]
else:
	time_ns = time_ns_full
	ampl_mV = ampl_mV_full

#===================== Apply Low-pass fitler =========================

# Calculate sample rate
delta_t = (time_ns[1] - time_ns[0])*1e-9
fs = 1/delta_t

# Create filter parameters
# b, a = butter(3, 5e9, fs=fs, btype='low', analog=False)
b, a = butter(7, 5e9, fs=fs, btype='low', analog=False)

# Apply filter to data
if use_lpfilter:
	filt_ampl_mV = lfilter(b, a, ampl_mV)
else:
	filt_ampl_mV = ampl_mV

# fig2 = plt.figure(figsize=(5, 3))
# gs = fig2.add_gridspec(1, 1)
# ax0 = fig2.add_subplot(gs[0, 0])

# ax0.plot(time_ns, ampl_mV, linestyle=':', marker='.', alpha=0.2, color=(0, 0, 0), label='Original')
# ax0.plot(time_ns, filt_ampl_mV, linestyle=':', marker='.', alpha=0.2, color=red, label='Filtered')

# ax0.set_xlabel("Time (ns)")
# ax0.set_ylabel("Amplitude (mV)")
# ax0.set_title("Using Doubler")
# ax0.grid(True)
# ax0.legend()

# plt.tight_layout()

#===================== Perform Hilbert transform ==========================

# Perform hilbert transform and calculate derivative parameters
a_signal = hilbert(filt_ampl_mV)
ampl_env = np.abs(a_signal)
inst_phase = np.unwrap(np.angle(a_signal))
inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs

ampl_norm = filt_ampl_mV/ampl_env

# #===================== Perform 2nd Hilbert transform ==========================

# anorm_signal = hilbert(ampl_norm)
# ampl_norm_env = np.abs(anorm_signal)
# inst_norm_phase = np.unwrap(np.angle(anorm_signal))
# inst_norm_freq = np.diff(inst_norm_phase) / (2 * np.pi) * fs


# #====================== Perform plotting ========================

fig1 = plt.figure(figsize=(8,6))
gs = fig1.add_gridspec(3, 1)

ax0 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0])
ax1 = fig1.add_subplot(gs[2, 0])


ax0.plot(time_ns, ampl_mV, linestyle=':', marker='.', alpha=0.2, color=red, label="Original signal")
ax0.plot(time_ns, filt_ampl_mV, linestyle=':', marker='.', alpha=0.2, color=green, label="Filtered signal")
ax0.plot(time_ns, ampl_env, linestyle='-', color=orange, label="Filtered signal")

ax2.plot(time_ns, ampl_norm, linestyle=':', marker='.', alpha=0.2, color=green)

ax1.plot(time_ns[1:], inst_freq/1e9, linestyle=':', marker='.', alpha=0.2, color=blue)
# ax1.plot(time_ns[1:], inst_norm_freq/1e9, linestyle=':', marker='.', alpha=0.2, color=green, label='Hilbert 2')

ax0.set_xlabel("Time (ns)")
ax0.set_ylabel("Amplitude (mV)")
ax0.set_title("Time Domain Data")
ax0.grid(True)
ax0.legend()

ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Amplitude (mV)")
ax2.set_title("Normalized Signal")
ax2.grid(True)
ax2.set_ylim([-1.5, 1.5])

ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Frequency (GHz)")
ax1.set_title("Hilbert Transform")
ax1.grid(True)
ax1.set_ylim([4, 6])

plt.tight_layout()

plt.show()





