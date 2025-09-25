''' Uses 1-full period (3 zero crossings) instead to make the analysis robust to
any y-offset.
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz
import time
import os

def get_colormap_colors(colormap_name, n):
	"""
	Returns 'n' colors as tuples that are evenly spaced in the specified colormap.
	
	Parameters:
	colormap_name (str): The name of the colormap.
	n (int): The number of colors to return.
	
	Returns:
	list: A list of 'n' colors as tuples.
	"""
	cmap = plt.get_cmap(colormap_name)
	colors = [cmap(i / (n - 1)) for i in range(n)]
	return colors

#====================== Program options =================

pi = 3.1415926535
merge = True
include_3rd = False

trim_time = False
use_hilbert_norm = False

# Window fit options
window_size_ns = 3.5
window_step_points = 10



#====================== Load data =================

# DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "13_Feb_2025")
DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_Feb_2025")
print(f"DATA DIRECTORY: {DATADIR}")
# DATADIR = os.path.join("/Volumes/M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "13_Feb_2025")

# df_raw = []
# df_raw.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,

# DATADIR = os.path.join("/Volumes/M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_feb_2025")

# df_raw.append(pd.read_csv(f"{DATADIR}/C1 BIAS0,30V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8'))
# trim_times = [-300, -225]
# df_raw = pd.read_csv(f"{DATADIR}/C1 BIAS0,15V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8')
df_raw = pd.read_csv(f"{DATADIR}/C1 BIAS0,15V_2,368GHz_HalfPiOut_-14dBm00000.txt", skiprows=4, encoding='utf-8')
trim_times = [-300, -225]


def find_closest_index(lst, X):
	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
	return closest_index

red = (0.6, 0, 0)
blue = (0, 0, 0.7)
green = (0, 0.5, 0)
orange = (1, 0.5, 0.05)

def fft_freq_analysis(dframe, pulse_range_ns = (10, 48)):
	
	print(f"Beginning FFT.")
	
	time_ns_full = dframe['Time']*1e9
	ampl_mV_full = dframe['Ampl']*1e3
	
	sample_rate = 1e9/(time_ns_full[1]-time_ns_full[0])
	
	# Trim timeseries
	if trim_time:
		idx_start = find_closest_index(time_ns_full, pulse_range_ns[0])
		idx_end = find_closest_index(time_ns_full, pulse_range_ns[1])
		time_ns = time_ns_full[idx_start:idx_end+1]
		ampl_mV = ampl_mV_full[idx_start:idx_end+1]
	else:
		time_ns = time_ns_full
		ampl_mV = ampl_mV_full
	
	spectrum_double = np.fft.fft(ampl_mV)
	freq_double = np.fft.fftfreq(len(spectrum_double), d=1/sample_rate)
	
	# Get single-ended results
	freq = freq_double[:len(freq_double)//2]
	spectrum = np.abs(spectrum_double[:len(freq_double)//2])
	
	return {'freq': freq, 'spec':spectrum, 't_trim':time_ns, 'y_trim':ampl_mV}

# #====================== Crunch numbers ========================


data_analyzed = fft_freq_analysis(df_raw, pulse_range_ns=trim_times)

# #====================== Perform plotting ========================

fig1 = plt.figure(1, figsize=(10, 6.5))
gs = fig1.add_gridspec(4, 1)
ax1a = fig1.add_subplot(gs[0:2, 0])
ax1b = fig1.add_subplot(gs[2, 0])
ax1c = fig1.add_subplot(gs[3, 0])

ax1a.plot(data_analyzed['freq']/1e9, data_analyzed['spec'], linestyle=':', marker='.', color=(0, 0, 0.65))
ax1a.set_title("FFT")
ax1a.set_xlabel("Frequency (GHz)")
ax1a.set_ylabel("Power")
ax1a.grid(True)
ax1a.set_xlim([4.775, 4.85])

ax1b.plot(data_analyzed['freq']/1e9, data_analyzed['spec'], linestyle=':', marker='.', color=(0, 0, 0.65))
ax1b.set_title("FFT")
ax1b.set_xlabel("Frequency (GHz)")
ax1b.set_ylabel("Power")
ax1b.grid(True)

ax1c.plot(data_analyzed['t_trim'], data_analyzed['y_trim'], linestyle=':', marker='.', color=(0, 0.6, 0), alpha=1)
ax1c.set_title("Trimmed Waveform")
ax1c.set_xlabel("Time")
ax1c.set_ylabel("Amplitude")
ax1c.grid(True)

fig1.tight_layout()

plt.show()