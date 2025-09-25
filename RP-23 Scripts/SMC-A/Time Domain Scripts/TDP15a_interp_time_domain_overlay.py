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
from scipy.interpolate import interp1d
import time
import os
import sys

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

trim_time = True

rescale = True
offset = 0.8
scaling = 1.48


#====================== Load data =================

# DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "13_Feb_2025")
# DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_Feb_2025")
# DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-19")
# DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-18")
DATADIR = os.path.join("/", "Volumes", "M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-18")
# DATADIR = os.path.join("/Volumes/M7 PhD Data", "18_March_2025 Data", "Time Domain")
print(f"DATA DIRECTORY: {DATADIR}")

# df_double = []
# df_double.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,

# DATADIR = os.path.join("/Volumes/M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_feb_2025")

# df_double.append(pd.read_csv(f"{DATADIR}/C1 BIAS0,30V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8'))
# trim_times = [-300, -225]
# df_double = pd.read_csv(f"{DATADIR}/C1 BIAS0,15V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8')

# # NOTE: From 19_March_2025, Should not have 40 MHz beat (if r9 used as straight)
# df_double = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,275V_-11,13dBm_2,3679GHz_15Pi_r6a_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,0V_-23dBm_4,7758GHz_15Pi_r9_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [-3500, -2900]

# # NOTE: From 18_March_2025, contained 40 MHz beat
df_double = pd.read_csv(os.path.join(DATADIR, "C1Long_waveform_0,275V_-11,13dBm_2,3679GHz_100Pi_r2_00000.txt"), skiprows=4, encoding='utf-8')
df_straight = pd.read_csv(os.path.join(DATADIR,"C1Long_waveform_0,0V_-23dBm_4,7358GHz_100Pi_r3_00000.txt"), skiprows=4, encoding='utf-8')
trim_times = [-22197, -22185]


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

def interpolate_common_times(t1, y1, t2, y2):
	"""
	Interpolates two time series onto a common time grid where they overlap.

	Parameters:
		t1, y1 : np.array - Time points and amplitudes for series 1
		t2, y2 : np.array - Time points and amplitudes for series 2

	Returns:
		T : np.array - Common time grid over the overlap
		Y1 : np.array - Interpolated amplitudes for series 1
		Y2 : np.array - Interpolated amplitudes for series 2
	"""
	# Convert lists to arrays if needed
	t1, y1, t2, y2 = map(np.array, (t1, y1, t2, y2))
	
	# Determine the overlapping time range
	t_min = max(t1[0], t2[0])
	t_max = min(t1[-1], t2[-1])
	
	dt = t_max - t_min
	srate = len(t1)/(t1[-1]-t1[0])
	n_points = int(dt*srate)
	
	# Create a common time grid in the overlapping region
	T = np.linspace(t_min, t_max, num=n_points)  # Adjust resolution as needed
	
	# Interpolate y1 and y2 onto the common time grid
	interp_y1 = interp1d(t1, y1, kind='linear', fill_value="extrapolate")
	interp_y2 = interp1d(t2, y2, kind='linear', fill_value="extrapolate")
	
	Y1 = interp_y1(T)
	Y2 = interp_y2(T)

	return T, Y1, Y2

# #====================== Crunch numbers ========================


# data_analyzed = fft_freq_analysis(df_double, pulse_range_ns=trim_times)

# #====================== Perform plotting ========================

# fig1 = plt.figure(1, figsize=(16, 9))
fig1 = plt.figure(1, figsize=(10, 7))
gs = fig1.add_gridspec(5, 1)
ax1a = fig1.add_subplot(gs[1:3, 0])
ax1b = fig1.add_subplot(gs[3:5, 0])
ax1c = fig1.add_subplot(gs[0, 0])

t_doub = np.array(df_double['Time']*1e9)
t_stra = np.array(df_straight['Time']*1e9)

v_doub = np.array(df_double['Ampl']*1e3)
v_stra = np.array(df_straight['Ampl']*1e3)

if rescale:
	v_stra = v_stra+offset
	v_stra = v_stra*scaling

t_univ, v_univ_doub, v_univ_stra = interpolate_common_times(t_doub, v_doub, t_stra, v_stra)
	
ax1a.plot(t_doub, v_doub, linestyle=':', marker='.', color=(0, 0, 0.65), label="Doubler")
ax1a.plot(t_stra, v_stra, linestyle=':', marker='.', color=(0, 0.65, 0), label="No doubler")
ax1a.set_title("Zoomed-In")
ax1a.set_xlabel("Time (ns))")
ax1a.set_ylabel("Voltage (mV)")
ax1a.grid(True)
ax1a.legend()
if trim_time:
	ax1a.set_xlim(trim_times)

ax1b.plot(t_univ, (v_univ_stra)-(v_univ_doub), linestyle=':', marker='.', color=(0.65, 0, 0))
ax1b.set_title("Subtracted")
ax1b.set_xlabel("Time (ns))")
ax1b.set_ylabel("Voltage (mV)")
ax1b.grid(True)
if trim_time:
	ax1b.set_xlim(trim_times)

ax1c.plot(t_doub, v_doub, linestyle=':', marker='.', color=(0, 0, 0.65), label="Doubler")
ax1c.plot(t_stra, v_stra, linestyle=':', marker='.', color=(0, 0.65, 0), label="No doubler")
ax1c.set_title("Time Domain Comparison - Full")
ax1c.set_xlabel("Time (ns))")
ax1c.set_ylabel("Voltage (mV)")
ax1c.grid(True)

fig1.tight_layout()

plt.show()