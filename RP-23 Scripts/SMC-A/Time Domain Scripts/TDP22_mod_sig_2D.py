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
import mplcursors
from scipy.interpolate import interp1d


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pylogfile.base as plf
from graf.base import sample_colormap

log = plf.LogPile()

#====================== Generate data =================

#----------------- USER CONFIG ----------------
f_mod = 40e6
f_rf = 4.8e9
sigma = 25e-9
offset_rad = 0
ampl = 1
ampl_rf = 1
  
t_start_0 = -100e-9
t_end_0 = 100e-9
sample_rate = 40e9

num_pulses = 5  

start_space_ns = 10
end_space_ns = 150
step_space_ns = 2
n_pts_psn = int((end_space_ns - start_space_ns)/step_space_ns)+1
pulse_spacings_ns = np.linspace(start_space_ns, end_space_ns, n_pts_psn)

#------------- END USER CONFIG ---------------

#---------- Generate a single pulse



# Calculate time points
n_pts_0 = int((t_end_0 - t_start_0)*sample_rate+1)
log.debug(f"Making waveform, npts={n_pts_0}")
t_series_0 = np.linspace(t_start_0, t_end_0, n_pts_0)

# Calculate gaussian envelope
envelope_0 = np.exp( -(t_series_0)**2/2/sigma**2 )

# Calculate modf waveform
v_mod_0 = ampl*np.sin(2*np.pi*f_mod*t_series_0) * envelope_0
I_mod_0 = ampl*np.cos(2*np.pi*f_mod*t_series_0) * envelope_0
Q_mod_0 = v_mod_0

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

# #=============== Run FFT and Trim ======================================

#------------ Begin simulation

freq_results = []
spectrum_results = []
timebases = []
rf_signals = []

for psn in pulse_spacings_ns:
	log.info(f"Starting analysis for pulse spacing = >{psn}<")
	
	# Generate spacing time points
	spacer = np.zeros( int(psn*1e-9*sample_rate) )
	
	I_mod = []
	Q_mod = []
	
	# Duplicate pulse N times
	for i in range(num_pulses):
		I_mod.extend(I_mod_0)
		I_mod.extend(spacer)
		
		Q_mod.extend(Q_mod_0)
		Q_mod.extend(spacer)
	
	# Generate new time series
	t_start = 0
	t_end = (len(I_mod)-1)/sample_rate
	t_series = np.linspace(t_start, t_end, len(I_mod))
	
	# Generate RF carriers
	Q_carrier_rf = ampl_rf*np.sin(2*np.pi*f_rf*t_series)
	I_carrier_rf = ampl_rf*np.cos(2*np.pi*f_rf*t_series)
	
	# Modulate RF signal
	v_rf = I_mod * I_carrier_rf - Q_mod*Q_carrier_rf
	
	# Run FFT on synthetic data
	f_mod, s_mod = run_fft(t_series, I_mod)
	freq_rf, s_rf = run_fft(t_series, v_rf)
	
	# Add to data lists
	freq_results.append(freq_rf)
	spectrum_results.append(s_rf)
	timebases.append(t_series)
	rf_signals.append(v_rf)


# #================= Interpolate results over common freqs ==================


f_rf_uni = np.linspace(4.78e9, 4.9e9, 241)
# f_rf_uni = np.linspace(4.82e9, 4.86e9, 81)

# Create the X and Y grids
X, Y = np.meshgrid(f_rf_uni/1e9, pulse_spacings_ns)

# Make blank Z grid
Z = np.zeros([len(pulse_spacings_ns), len(f_rf_uni)])

# Populate Z
for idx, psn in enumerate(pulse_spacings_ns):
	
	# Get interp function
	f = interp1d(freq_results[idx], spectrum_results[idx])
	z_interp = f(f_rf_uni)
	
	# Slice into list
	Z[idx][:] = z_interp


# #==================== Generate Plot ========================

# Create a figure and an axes object for the 3D plot
fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0], projection='3d')

# Plot the surface
surf = ax1a.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Create a figure and an axes object for the 3D plot
fig3 = plt.figure(3)
gs3 = fig3.add_gridspec(1, 1)
ax3a = fig3.add_subplot(gs3[0, 0])

# Plot the surface
surf = ax3a.pcolormesh(X, Y, Z, cmap='viridis')
ax3a.set_xlabel(f"Frequency (GHz)")
ax3a.set_ylabel(f"Pulse Spacings (ns)")
ax3a.set_title("Fourier Transform")

fig3.colorbar(surf, label="Spectral Power (dBm/Hz)")

# Customize the z-axis
# ax1a.zaxis.set_major_locator(LinearLocator(10))
# ax1a.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar
# fig1.colorbar(surf, shrink=0.5, aspect=5)

#================= Generate overlap plot =====================

fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

cmap = sample_colormap(cmap_name='viridis', N=len(pulse_spacings_ns))

for idx, psn in enumerate(pulse_spacings_ns):
	
	ax2a.plot(freq_results[idx]/1e9, spectrum_results[idx], color=cmap[idx], alpha=0.2, label=f"Pulse spacing: {psn}")

ax2a.grid(True)
ax2a.set_xlabel("Frequency (GHz)")
ax2a.set_ylabel("Power (dBm/Hz)")
ax2a.set_xlim([4.78, 4.9])
ax2a.set_title(f"Fourier Transform")

#================= Generate time domain plot =====================

fig4 = plt.figure(4)
gs4 = fig4.add_gridspec(3, 1)
ax4a = fig4.add_subplot(gs4[0, 0])
ax4b = fig4.add_subplot(gs4[1, 0])
ax4c = fig4.add_subplot(gs4[2, 0])

cmap = sample_colormap(cmap_name='viridis', N=len(pulse_spacings_ns))

axes=  [ax4a, ax4b, ax4c]
plots_idx = [0, 5, len(pulse_spacings_ns)-1]

for idx, plot_idx in enumerate(plots_idx):
	psn =  pulse_spacings_ns[plot_idx]
	axes[idx].plot(timebases[plot_idx]*1e9, rf_signals[plot_idx], color=(0, 0, 0.8), label=f"Pulse spacing: {psn} ns")
	axes[idx].set_title(f"Pulse spacing: {psn} ns")

	axes[idx].grid(True)
	axes[idx].set_xlabel("Time (ns)")
	axes[idx].set_ylabel("Voltage (V)")

xl = axes[-1].get_xlim()
for ax in axes:
	ax.set_xlim(xl)

plt.tight_layout()
plt.show()