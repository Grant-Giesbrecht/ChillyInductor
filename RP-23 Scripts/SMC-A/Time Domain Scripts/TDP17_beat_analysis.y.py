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

rescale = False
offset = 0.8
scaling = 1.48
time_shift_ns = 0


#====================== Load data =================

print(f"Loading files...")

# DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "13_Feb_2025")
# DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_Feb_2025")
# DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-19")
DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-25")
# DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-18")
# DATADIR = os.path.join("/Volumes/M7 PhD Data", "18_March_2025 Data", "Time Domain")
# DATADIR = os.path.join("/", "Volumes", "M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-19")
DATADIR = os.path.join("/", "Volumes", "M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-18")
print(f"DATA DIRECTORY: {DATADIR}")

# df_double = []
# df_double.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,

# DATADIR = os.path.join("/Volumes/M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_feb_2025")

# df_double.append(pd.read_csv(f"{DATADIR}/C1 BIAS0,30V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8'))
# trim_times = [-300, -225]
# df_double = pd.read_csv(f"{DATADIR}/C1 BIAS0,15V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8')

#NOTE: Comparing two direct-drive pulses to see how good subtraction can look.
df_double = pd.read_csv(f"{DATADIR}/C1Medwav_0,0V_-17dBm_4,7758GHz_15Pi_sig25ns_r24_00000.txt", skiprows=4, encoding='utf-8')
df_straight = pd.read_csv(f"{DATADIR}/C1Medwav_0,0V_-19dBm_4,7758GHz_15Pi_sig25ns_r23_00000.txt", skiprows=4, encoding='utf-8')
trim_times = [-300, 600]
rescale = True
offset = 3.07
scaling = 1.242
time_shift_ns = -0.1
rescale_doubler = True
offset_doubler = 3.015
void_threshold = 0.75

# #NOTE: Trying to look at de-chirp pulse:
# df_double = pd.read_csv(f"{DATADIR}/C1Medwav_0,075V_-4dBm_2,3679GHz_15Pi_sig35ns_r13_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Medwav_dechirp0.01K_0,275V_-9dBm_2,3679GHz_15Pi_sig35ns_r22_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [-600, 400]
# rescale = True
# offset = 3.63
# scaling = 1.38
# rescale_doubler = True
# offset_doubler = 3
# void_threshold = 0.75

# #NOTE: COrrected sigmas, trying other cases - see 3 MHz offset
# df_double = pd.read_csv(f"{DATADIR}/C1Medwav_0,075V_-4dBm_2,3679GHz_15Pi_sig35ns_r13_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Medwav_0,0V_-21dBm_4,7758GHz_15Pi_sig25ns_r17_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [-600, 400]
# rescale = True
# offset = 3.63
# scaling = 1.38
# rescale_doubler = True
# offset_doubler = 3
# void_threshold = 0.75

# # NOTE: First with corrected sigmas, looked weird?
# df_double = pd.read_csv(f"{DATADIR}/C1Medwav_0,275V_-9dBm_2,3679GHz_15Pi_sig35ns_r15_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Medwav_0,0V_-16dBm_4,7758GHz_15Pi_sig25ns_r18_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [-600, 400]
# rescale = True
# offset = 0.8
# scaling = 1.45
# void_threshold = 0.75


# # NOTE: From 19_March_2025, should have strongest nonlinearity
# df_double = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,070V_-4dBm_2,3679GHz_15Pi_r8_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,0V_-23dBm_4,7758GHz_15Pi_r9_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [-3500, -2900]
# rescale = True
# offset = 0.8
# scaling = 1.45
# void_threshold = 0.75

# # NOTE: From 19_March_2025, Should not have 40 MHz beat (if r9 used as straight)
# df_double = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,275V_-11,13dBm_2,3679GHz_15Pi_r6a_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,0V_-23dBm_4,7758GHz_15Pi_r9_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [-3500, -2900]
# rescale = True
# offset = 0.8
# scaling = 1.48
# void_threshold = 0.75


# NOTE: From 18_March_2025, contained 40 MHz beat
df_double = pd.read_csv(os.path.join(DATADIR, "C1Long_waveform_0,275V_-11,13dBm_2,3679GHz_100Pi_r2_00000.txt"), skiprows=4, encoding='utf-8')
df_straight = pd.read_csv(os.path.join(DATADIR, "C1Long_waveform_0,0V_-23dBm_4,7358GHz_100Pi_r3_00000.txt"), skiprows=4, encoding='utf-8')
# trim_times = [-22197, -22185]
trim_times = [-22000, -21500]
rescale = True
offset = -0.65
scaling = 1.234
void_threshold = 0.75

print(f"  --> Files loaded.")

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
	
	# Estimate required resolution:
	pts_per_sec = len(t1)/(t1[-1]-t1[0])
	num_pts = int((t_max-t_min)*pts_per_sec)
	print(f"Interpolating with: {num_pts} points")
	
	# Create a common time grid in the overlapping region
	T = np.linspace(t_min, t_max, num=num_pts)  # Adjust resolution as needed

	# Interpolate y1 and y2 onto the common time grid
	interp_y1 = interp1d(t1, y1, kind='linear', fill_value="extrapolate")
	interp_y2 = interp1d(t2, y2, kind='linear', fill_value="extrapolate")

	Y1 = interp_y1(T)
	Y2 = interp_y2(T)

	return T, Y1, Y2

def trim_time_series(t, y, t_start, t_end):
	"""
	Trims a time series to keep only values where t is within [t_start, t_end].

	Parameters:
		t : np.array - Time values
		y : np.array - Corresponding data values
		t_start : float - Start time
		t_end : float - End time

	Returns:
		t_trimmed, y_trimmed : np.array - Trimmed time and data arrays
	"""
	mask = (t >= t_start) & (t <= t_end)  # Create a boolean mask
	return t[mask], y[mask]  # Apply mask to both arrays

def find_zero_crossings(x, y):
	''' Finds zero crossings of x, then uses y-data to interpolate between the points.'''
	
	signs = np.sign(y)
	sign_changes = np.diff(signs)
	zc_indices = np.where(sign_changes != 0)[0]
	
	# Trim end points - weird things can occur if the waveform starts or ends at zero
	zc_indices = zc_indices[1:-1]
	
	# Interpolate each zero-crossing
	cross_times = []
	for zci in zc_indices:
		dx = x[zci+1] - x[zci]
		dy = y[zci+1] - y[zci]
		frac = np.abs(y[zci]/dy)
		cross_times.append(x[zci]+dx*frac)
	
	return cross_times

def zero_cross_freq_analysis(t_ns, V_mv, N_avg):
	
	print(f"Beginning windowed fit (linear).")
	t0 = time.time()
	
	time_ns = t_ns
	ampl_mV = V_mv
	
	#===================== Perform zero crossing analysis ==========================
	
	tzc = find_zero_crossings(time_ns, ampl_mV)
	
	# Select every other zero crossing to see full periods and become insensitive to y-offsets.
	tzc_fullperiod = tzc[::2*N_avg]
	periods = np.diff(tzc_fullperiod)
	freqs = (1/periods)*N_avg

	t_freqs = tzc_fullperiod[:-1] + periods/2
	
	return t_freqs, freqs

# #====================== Crunch numbers ========================


# data_analyzed = fft_freq_analysis(df_double, pulse_range_ns=trim_times)

# #====================== Perform plotting ========================

print(f"Converting data to arrays...")
t_doub = np.array(df_double['Time']*1e9)
t_stra = np.array(df_straight['Time']*1e9)

v_doub = np.array(df_double['Ampl']*1e3)
v_stra = np.array(df_straight['Ampl']*1e3)
print(f"  --> Conversion complete.")

print(f"Trimming time points....")
if trim_time:
	
	#Perform time shift
	t_stra = t_stra+time_shift_ns
	
	# Perform trim
	t_doub, v_doub = trim_time_series(t_doub, v_doub, trim_times[0], trim_times[1])
	t_stra, v_stra = trim_time_series(t_stra, v_stra, trim_times[0], trim_times[1])
print(f"  --> Trim complete.")

print(f"Scaling data...")
if rescale:
	v_stra = v_stra+offset
	v_stra = v_stra*scaling
if rescale_doubler:
	v_doub = v_doub+offset_doubler
print(f"  --> Scaling complete.")

print(f"Interpolating times to universal axis...")
t_univ, v_univ_doub, v_univ_stra = interpolate_common_times(t_doub, v_doub, t_stra, v_stra)
delta = v_univ_stra-v_univ_doub
print(f"  --> Interpolation complete.")

print(f"Running hilbert transforms...")
# Run hilbert on delta
an_delta = hilbert(delta)
delta_env = np.abs(an_delta)

# Run hilbert on doubled-signal
an_vud = hilbert(v_univ_doub)
vud_env = np.abs(an_vud)

# Run hilbert on doubled-signal
an_vus = hilbert(v_univ_stra)
vus_env = np.abs(an_vus)

# Normalize delta to doubled-envelope
vu_avg_env = (vus_env + vud_env)/2
norm_delta = delta/vu_avg_env

# Remove points with low envelope
weak_mask = (vu_avg_env < void_threshold)
norm_delta[weak_mask] = np.nan
print(f"  --> Hilbert transforms complete.")

print(f"Analyzing frequency...")
tzc_doub, freq_doub = zero_cross_freq_analysis(t_univ, v_univ_doub, 60)
tzc_stra, freq_stra = zero_cross_freq_analysis(t_univ, v_univ_stra, 60)
# freq_stra[weak_mask] = np.nan
# freq_doub[weak_mask] = np.nan
# t_stra[weak_mask] = np.nan
# t_doub[weak_mask] = np.nan
print(f"  --> Frequency analysis complete.")

# fig1 = plt.figure(1, figsize=(16, 9))
fig1 = plt.figure(1, figsize=(10, 11))
gs = fig1.add_gridspec(5, 1)
ax1c = fig1.add_subplot(gs[0, 0])
ax1a = fig1.add_subplot(gs[1, 0])
ax1b = fig1.add_subplot(gs[2, 0])
ax1d = fig1.add_subplot(gs[3, 0])
ax1e = fig1.add_subplot(gs[4, 0])

ax1a.plot(t_doub, v_doub, linestyle=':', marker='.', color=(0, 0, 0.65), label="Doubler")
ax1a.plot(t_stra, v_stra, linestyle=':', marker='.', color=(0, 0.65, 0), label="No doubler")
ax1a.set_title("Zoomed-In")
ax1a.set_xlabel("Time (ns))")
ax1a.set_ylabel("Voltage (mV)")
ax1a.grid(True)
ax1a.legend()
# if trim_time:
# 	ax1a.set_xlim(trim_times)

ax1b.plot(t_univ, delta, linestyle=':', marker='.', color=(0.65, 0, 0))
# ax1b.plot(t_univ, delta_env, linestyle='-', color=(1, 0.7, 0))
ax1b.plot(t_univ, vud_env, linestyle='--', color=(0.6, 0.8, 0), label='doubler env')
ax1b.plot(t_univ, vus_env, linestyle='--', color=(0.8, 0.6, 0), label='straight env')
ax1b.plot(t_univ, vu_avg_env, linestyle='-', color=(1, 0.7, 0), label='Averaged env')
ax1b.set_title("Subtracted")
ax1b.set_xlabel("Time (ns))")
ax1b.set_ylabel("Voltage (mV)")
ax1b.grid(True)
ax1b.legend()
# if trim_time:
# 	ax1b.set_xlim(trim_times)
	
ax1d.plot(t_univ, norm_delta, linestyle=':', marker='.', color=(0.56, 0, 0.56))
ax1d.set_title("Subtracted, Normalized")
ax1d.set_xlabel("Time (ns))")
ax1d.set_ylabel("Voltage (mV)")
ax1d.grid(True)
# if trim_time:
# 	ax1d.set_xlim(trim_times)
	
ax1c.plot(t_doub, v_doub, linestyle=':', marker='.', color=(0, 0, 0.65), label="Doubler")
ax1c.plot(t_stra, v_stra, linestyle=':', marker='.', color=(0, 0.65, 0), label="No doubler")
ax1c.set_title("Time Domain Comparison - Full")
ax1c.set_xlabel("Time (ns))")
ax1c.set_ylabel("Voltage (mV)")
ax1c.grid(True)

ax1e.plot(tzc_doub, freq_doub, linestyle=':', marker='.', color=(0, 0, 0.65), label="Doubler")
ax1e.plot(tzc_stra, freq_stra, linestyle=':', marker='.', color=(0, 0.65, 0), label="No doubler")
ax1e.set_title("Frequency Analysis")
ax1e.set_xlabel("Time (ns)")
ax1e.set_ylabel("Frequency (GHz)")
ax1e.grid(True)
ax1e.set_ylim([4.75, 4.85])

fig1.tight_layout()

plt.figure(2)
plt.plot(tzc_doub, freq_doub, linestyle=':', marker='.', color=(0, 0, 0.65), label="Doubler")
plt.plot(tzc_stra, freq_stra, linestyle=':', marker='.', color=(0, 0.65, 0), label="No doubler")
plt.title("Frequency Analysis")
# plt.set_xlabel("Time (ns)")
# plt.set_ylabel("Frequency (GHz)")
plt.grid(True)
# plt.ylim([4.75, 4.85])

plt.show()