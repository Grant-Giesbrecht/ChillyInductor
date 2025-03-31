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
DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-19")
# DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-18")
# DATADIR = os.path.join("/Volumes/M7 PhD Data", "18_March_2025 Data", "Time Domain")
print(f"DATA DIRECTORY: {DATADIR}")

# df_double = []
# df_double.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,

# DATADIR = os.path.join("/Volumes/M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_feb_2025")

# df_double.append(pd.read_csv(f"{DATADIR}/C1 BIAS0,30V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8'))
# trim_times = [-300, -225]
# df_double = pd.read_csv(f"{DATADIR}/C1 BIAS0,15V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8')

f0 = 4.8e9

sample_rate = 40e9
t_series = np.linspace(-250e-9, 250e-9, 1/sample_rate)
sigma = 25e-9

K1 = 1 # L*n^2*I0
Tau = np.sqrt(2)*sigma
chirped_freq = f0 + (4*np.pi*K1)/()

pulse_straight = np.sin(f0*np.pi*2*t_series) * np.exp( -(t_series)**2/2/sigma**2 )
pulse_straight = np.sin(chirped_freq*np.pi*2*t_series) * np.exp( -(t_series)**2/2/sigma**2 )

# NOTE: From 19_March_2025, should have strongest nonlinearity
df_double_strong = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,070V_-4dBm_2,3679GHz_15Pi_r8_00000.txt", skiprows=4, encoding='utf-8')
df_straight = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,0V_-23dBm_4,7758GHz_15Pi_r9_00000.txt", skiprows=4, encoding='utf-8')
df_double = pd.read_csv(f"{DATADIR}/C1Med_waveform_0,275V_-11,13dBm_2,3679GHz_15Pi_r6a_00000.txt", skiprows=4, encoding='utf-8')
df_double_strong_f1 = pd.read_csv(f"{DATADIR}/F1Med_waveform_0,070V_-4dBm_2,3679GHz_15Pi_r8_00000.txt", skiprows=4, encoding='utf-8')
df_straight_f1 = pd.read_csv(f"{DATADIR}/F1Med_waveform_0,0V_-23dBm_4,7758GHz_15Pi_r9_00000.txt", skiprows=4, encoding='utf-8')
df_double_f1 = pd.read_csv(f"{DATADIR}/F1Med_waveform_0,275V_-11,13dBm_2,3679GHz_15Pi_r6a_00000.txt", skiprows=4, encoding='utf-8')
# trim_times = [4.5, 5]
trim_times = [0, 50]
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

t_strong = np.array(df_double_strong['Time'])
v_strong = np.array(df_double_strong['Ampl'])

t_double = np.array(df_double['Time'])
v_double = np.array(df_double['Ampl'])

t_direct = np.array(df_straight['Time'])
v_direct = np.array(df_straight['Ampl'])

#================ Get spectral data from F1 files and trim ================

f_direct_f1 = np.array(df_straight_f1['Time'])
s_direct_f1 = np.array(df_straight_f1['Ampl'])

f_strong_f1 = np.array(df_double_strong_f1['Time'])
s_strong_f1 = np.array(df_double_strong_f1['Ampl'])

f_double_f1 = np.array(df_double_f1['Time'])
s_double_f1 = np.array(df_double_f1['Ampl'])


f_dir_f1, s_dir_f1 = trim_time_series(f_direct_f1/1e9, s_direct_f1, trim_times[0], trim_times[1])
f_doub_f1, s_doub_f1 = trim_time_series(f_double_f1/1e9, s_double_f1, trim_times[0], trim_times[1])
f_str_f1, s_str_f1 = trim_time_series(f_strong_f1/1e9, s_strong_f1, trim_times[0], trim_times[1])

#=============== Run FFT and Trim ======================================

f_direct, s_direct = run_fft(t_direct, v_direct)
f_double, s_double = run_fft(t_double, v_double)
f_strong, s_strong = run_fft(t_strong, v_strong)


f_dir, s_dir = trim_time_series(f_direct/1e9, s_direct, trim_times[0], trim_times[1])
f_doub, s_doub = trim_time_series(f_double/1e9, s_double, trim_times[0], trim_times[1])
f_str, s_str = trim_time_series(f_strong/1e9, s_strong, trim_times[0], trim_times[1])

#================= Plot results



fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])
# ax1b = fig1.add_subplot(gs1[1, 0])

ax1a.plot(f_dir, s_dir, linestyle=':', marker='.', color=(0.3, 0.3, 0.85), label="Direct Drive")
ax1a.plot(f_doub, s_doub, linestyle=':', marker='.', color=(0, 0.75, 0), label="Doubler, 0.0275 V")
ax1a.plot(f_str, s_str, linestyle=':', marker='.', color=(0.65, 0, 0), label="Doubler, 0.07 V")

ax1a.set_xlabel(f"Frequency (GHz)")
ax1a.set_ylabel(f"Power spectral density (dBm/Hz)")
ax1a.set_title("Fourier Transform")
ax1a.legend()
ax1a.grid(True)

fig1.tight_layout()

fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

ax2a.plot(f_dir_f1, s_dir_f1, linestyle=':', marker='o', color=(0.3, 0.3, 0.85), label="Direct Drive")
ax2a.plot(f_doub_f1, s_doub_f1, linestyle=':', marker='o', color=(0, 0.75, 0), label="Doubler, 0.0275 V")
ax2a.plot(f_str_f1, s_str_f1, linestyle=':', marker='o', color=(0.65, 0, 0), label="Doubler, 0.07 V")

ax2a.set_xlabel(f"Frequency (GHz)")
ax2a.set_ylabel(f"Power spectral density (dBm/Hz)")
ax2a.set_title("Oscilloscope FFT")
ax2a.legend()
ax2a.grid(True)

mplcursors.cursor(multiple=True)

plt.show()