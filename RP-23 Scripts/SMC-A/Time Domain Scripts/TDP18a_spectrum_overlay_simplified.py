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
import argparse
import mplcursors

#====================== Load data =================

print(f"Loading files...")

# DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "13_Feb_2025")
# DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_Feb_2025")
# DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-19")
DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-25")
# DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-18")
# DATADIR = os.path.join("/Volumes/M7 PhD Data", "18_March_2025 Data", "Time Domain")
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

# #NOTE: COrrected sigmas, trying other cases - known 3 MHz delta at higher spectrum end.
# df_double = pd.read_csv(f"{DATADIR}/C1Medwav_0,075V_-4dBm_2,3679GHz_15Pi_sig35ns_r13_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Medwav_0,0V_-21dBm_4,7758GHz_15Pi_sig25ns_r17_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [-600, 400]
# rescale = True
# offset = 3.63
# scaling = 1.38
# rescale_doubler = True
# offset_doubler = 3
# void_threshold = 0.75

# # NOTE: From 19_March_2025, should have strongest nonlinearity
# df_double_strong = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,070V_-4dBm_2,3679GHz_15Pi_r8_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,0V_-23dBm_4,7758GHz_15Pi_r9_00000.txt", skiprows=4, encoding='utf-8')
# df_double = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,275V_-11,13dBm_2,3679GHz_15Pi_r6a_00000.txt", skiprows=4, encoding='utf-8')
# df_double_strong_f1 = pd.read_csv(f"{DATADIR}/F1Med_waveform_0,070V_-4dBm_2,3679GHz_15Pi_r8_00000.txt", skiprows=4, encoding='utf-8')
# df_straight_f1 = pd.read_csv(f"{DATADIR}/F1Med_waveform_0,0V_-23dBm_4,7758GHz_15Pi_r9_00000.txt", skiprows=4, encoding='utf-8')
# df_double_f1 = pd.read_csv(f"{DATADIR}/F1Med_waveform_0,275V_-11,13dBm_2,3679GHz_15Pi_r6a_00000.txt", skiprows=4, encoding='utf-8')
# # trim_times = [4.5, 5]
# trim_times = [0, 50]
# rescale = True
# offset = 0.8
# scaling = 1.45
void_threshold = 0.75

# # NOTE: From 19_March_2025, Should not have 40 MHz beat (if r9 used as straight)
# df_double = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,275V_-11,13dBm_2,3679GHz_15Pi_r6a_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,0V_-23dBm_4,7758GHz_15Pi_r9_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [-3500, +2900]
# rescale = True
# offset = 0.8
# scaling = 1.48
# void_threshold = 0.75


# # # NOTE: From 18_March_2025, contained 40 MHz beat
# df_double = pd.read_csv(f"{DATADIR}\\C1Long_waveform_0,275V_-11,13dBm_2,3679GHz_100Pi_r2_00000.txt", skiprows=4, encoding='utf-8')
# df_straight = pd.read_csv(f"{DATADIR}\\C1Long_waveform_0,0V_-23dBm_4,7358GHz_100Pi_r3_00000.txt", skiprows=4, encoding='utf-8')
# # trim_times = [-22197, -22185]
# trim_times = [-22000, -21500]
# rescale = True
# offset = -0.65
# scaling = 1.234
# void_threshold = 0.75

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

def run_fft(t_si, v_si):
	
	R = 50
	sample_period = (t_si[-1]-t_si[0])/(len(t_si)-1)
	sample_rate = 1/sample_period
	spectrum_double = np.fft.fft(v_si)
	freq_double = np.fft.fftfreq(len(spectrum_double), d=1/sample_rate)
	
	# Get single-ended results
	freq = freq_double[:len(freq_double)//2]
	spectrum_W = (np.abs(spectrum_double[:len(freq_double)//2])**2) / (R * sample_rate) #len(v_si))
	spectrum = 10 * np.log10(spectrum_W*1e3)

	return freq, spectrum

#============ GEt time and voltage arrays ==================

t_double = np.array(df_double['Time'])
v_double = np.array(df_double['Ampl'])

t_direct = np.array(df_straight['Time'])
v_direct = np.array(df_straight['Ampl'])

#=============== Run FFT and Trim ======================================

f_direct, s_direct = run_fft(t_direct, v_direct)
f_double, s_double = run_fft(t_double, v_double)

f_dir, s_dir = trim_time_series(f_direct/1e9, s_direct, trim_times[0], trim_times[1])
f_doub, s_doub = trim_time_series(f_double/1e9, s_double, trim_times[0], trim_times[1])

#================= Plot results



fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])
# ax1b = fig1.add_subplot(gs1[1, 0])

ax1a.plot(f_dir, s_dir, linestyle=':', marker='.', color=(0.3, 0.3, 0.85), label="Direct Drive")
ax1a.plot(f_doub, s_doub, linestyle=':', marker='.', color=(0, 0.75, 0), label="Doubler, 0.0275 V")

ax1a.set_xlabel(f"Frequency (GHz)")
ax1a.set_ylabel(f"Power spectral density (dBm/Hz)")
ax1a.set_title("Fourier Transform")
ax1a.legend()
ax1a.grid(True)

fig1.tight_layout()

mplcursors.cursor(multiple=True)

plt.show()