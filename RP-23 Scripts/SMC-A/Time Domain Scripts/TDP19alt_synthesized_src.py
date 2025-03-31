''' From my previous TDP scripts I've got some messy signals that look like they
could be due to chirping, but I'm not fully sure. Running a proper simulation by
generating solutions to the nonlinear telegrapher's equations is hard. Making synthetic
waveforms that inject a chirp and analyzing them using the same process is comparatively
easy. 

Runs two forms of analysis:
- Fourier transform: compare spectra
- Normalized-subtraction method

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
import sys
import argparse

#====================== Generate data =================

#----------------- USER CONFIG ----------------
f1 = 4.8e9
f2 = 4.8e9
sigma = 25e-9
offset_rad = 0
ampl = 0.01

t_start = -250e-9
t_end = 250e-9
sample_rate = 40e9

N_avg = 30

add_noise = False
noise_ampl = 0.05*ampl
#------------- END USER CONFIG ---------------

# Calculate time points
n_pts = int((t_end - t_start)*sample_rate+1)
t_series = np.linspace(t_start, t_end, n_pts)

# Calculate gaussian envelope
envelope = np.exp( -(t_series)**2/2/sigma**2 )

K1 = 0.005 # L*n2*I0
tau = np.sqrt(2)*sigma
wavelength = 3e8/f2

# Calculate chirped frequency
fchirp = f2 + (4*np.pi*K1)/(wavelength*tau**2) * t_series * envelope

# Calculate waveforms
v_direct = ampl*np.sin(2*np.pi*f1*t_series) * envelope
v_double = ampl*np.sin(2*np.pi*fchirp*t_series + offset_rad) * envelope

# Add noise
if add_noise:
	
	noise = np.random.normal(0,noise_ampl,len(v_direct))
	
	v_direct = v_direct + noise
	v_double = v_double + noise

#===================== Define functions ===============

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

def find_closest_index(lst, X):
	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
	return closest_index

red = (0.6, 0, 0)
blue = (0, 0, 0.7)
green = (0, 0.5, 0)
orange = (1, 0.5, 0.05)

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

#================= Run normalized-subtraction analysis ===================

# Calculate delta
delta = v_direct - v_double

# Run hilbert on doubled-signal
an_double = hilbert(v_double)
env_double = np.abs(an_double)

# Run hilbert on direct-signal
an_direct = hilbert(v_direct)
env_direct = np.abs(an_direct)

# Normalize delta to doubled-envelope
env_avg = (env_double + env_direct)/2
norm_delta = delta/env_avg

# Run zero-cross analysis
t_zc_direct, f_zc_direct = zero_cross_freq_analysis(t_series, v_direct, N_avg)
t_zc_double, f_zc_double = zero_cross_freq_analysis(t_series, v_double, N_avg)

# #=============== Run FFT and Trim ======================================

# Run FFT on synthetic data
f_direct, s_direct = run_fft(t_series, v_direct)
f_double, s_double = run_fft(t_series, v_double)

#================= Plot results ============================

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(2, 1)
ax1a = fig1.add_subplot(gs1[0, 0])
ax1b = fig1.add_subplot(gs1[1, 0])

ax1a.plot(t_series*1e9, v_direct, linestyle='-', marker='None', color=(0.05, 0.05, 0.6), label="Direct", alpha=0.2)
ax1a.plot(t_series*1e9, v_double, linestyle='-', marker='None', color=(0.05, 0.6, 0.05), label="Doubler", alpha=0.2)
ax1a.set_xlabel(f"Time (ns)")
ax1a.set_ylabel(f"Voltage (mV)")
ax1a.grid(True)
ax1a.legend()

ax1b.plot(t_series*1e9, fchirp/1e9, linestyle='--', marker='None', color=(0.05, 0.6, 0.0), label="Chirp Frequency", alpha=1)
ax1b.axhline(f1/1e9, linestyle='--', color=(0.6, 0, 0.6), label="Frequency 1")
ax1b.set_xlabel(f"Time (ns)")
ax1b.set_ylabel(f"Design Frequency (GHz)")
ax1b.grid(True)
ax1b.legend()

fig1.tight_layout()


fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])
# ax1b = fig2.add_subplot(gs2[1, 0])

ax2a.plot(f_direct/1e9, s_direct, linestyle=':', marker='.', color=(0.3, 0.3, 0.85), label="Direct")
ax2a.plot(f_double/1e9, s_double, linestyle=':', marker='.', color=(0, 0.75, 0), label="Doubler")
ax2a.set_xlabel(f"Frequency (GHz)")
ax2a.set_ylabel(f"Power spectral density (dBm/Hz)")
ax2a.set_title("Fourier Transform")
ax2a.legend()
ax2a.grid(True)
fig2.tight_layout()

fig3 = plt.figure(3)
gs3 = fig3.add_gridspec(3, 1)
ax3a = fig3.add_subplot(gs3[0, 0])
ax3b = fig3.add_subplot(gs3[1, 0])
ax3c = fig3.add_subplot(gs3[2, 0])
# ax3d = fig3.add_subplot(gs3[3, 0])

ax3a.plot(t_series*1e9, delta, linestyle=':', marker='.', color=(0.75, 0.3, 0.3), label="Delta")
ax3a.plot(t_series*1e9, env_double, linestyle='-.', color=(0.5, 0.5, 0.5), label="Envelope: Doubler")
ax3a.plot(t_series*1e9, env_direct, linestyle='-.', color=(0.25, 0.3, 0.3), label="Envelope: Direct")
ax3a.plot(t_series*1e9, env_avg, linestyle='-.', color=(1, 0.7, 0), label="Envelope: Averaged")
ax3a.set_xlabel("Time (ns)")
ax3a.set_ylabel("Voltage (mV)")
ax3b.set_title(f"Delta")
ax3a.grid(True)

ax3b.plot(t_series*1e9, norm_delta, linestyle=':', marker='.', color=(0.7, 0.1, 0.5))
ax3b.set_xlabel("Time (ns)")
ax3b.set_ylabel("Voltage (mV)")
ax3b.set_title(f"Normalized Delta")
ax3b.grid(True)

ax3c.plot(t_zc_direct*1e9, f_zc_direct/1e9, linestyle=':', marker='.', color=(0.3, 0.3, 0.85), label="Direct")
ax3c.plot(t_zc_double*1e9, f_zc_double/1e9, linestyle=':', marker='.', color=(0.3, 0.75, 0.3), label="Doubler")
ax3c.set_xlabel("Time (ns)")
ax3c.set_ylabel("Frequency (GHz)")
ax3c.set_title(f"Zero-Crossing Analysis")
ax3c.grid(True)

fig3.tight_layout()

# fig2 = plt.figure(2)
# gs2 = fig2.add_gridspec(1, 1)
# ax2a = fig2.add_subplot(gs2[0, 0])

# ax2a.plot(f_dir_f1, s_dir_f1, linestyle=':', marker='o', color=(0.3, 0.3, 0.85), label="Direct Drive")
# ax2a.plot(f_doub_f1, s_doub_f1, linestyle=':', marker='o', color=(0, 0.75, 0), label="Doubler, 0.0275 V")
# ax2a.plot(f_str_f1, s_str_f1, linestyle=':', marker='o', color=(0.65, 0, 0), label="Doubler, 0.07 V")

# ax2a.set_xlabel(f"Frequency (GHz)")
# ax2a.set_ylabel(f"Power spectral density (dBm/Hz)")
# ax2a.set_title("Oscilloscope FFT")
# ax2a.legend()
# ax2a.grid(True)

# ax2a.set_xlim([4.7, 4.9])

plt.show()