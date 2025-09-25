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

trim_time = True
use_hilbert_norm = False

# Window fit options
window_size_ns = 3.5
window_step_points = 10



#====================== Load data =================

# DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "13_Feb_2025")
DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_Feb_2025")

# DATADIR = os.path.join("/Volumes/M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "13_Feb_2025")

# df_100mV_3dBm_df = []
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,

# DATADIR = os.path.join("/Volumes/M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "20_feb_2025")

df_100mV_3dBm_df = []
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1 BIAS0,30V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8'))
# trim_times = [-300, -225]
df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1 BIAS0,15V_2,368GHz_HalfPiOut_-4dBm00000.txt", skiprows=4, encoding='utf-8'))
trim_times = [-300, -225]

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
	

def find_closest_index(lst, X):
	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
	return closest_index

red = (0.6, 0, 0)
blue = (0, 0, 0.7)
green = (0, 0.5, 0)
orange = (1, 0.5, 0.05)

def flat_sine (x, ampl, omega, phi):
	return np.sin(omega*x-phi)*ampl

def linear_sine (x, ampl, omega, phi, m, offset):
	return np.sin(omega*x-phi)*(ampl + m*x) + offset

def zero_cross_freq_analysis(dframe, pulse_range_ns = (10, 48)):
	
	print(f"Beginning windowed fit (linear).")
	t0 = time.time()
	
	time_ns_full = list(dframe['Time']*1e9)
	ampl_mV_full = list(dframe['Ampl']*1e3)
	supertitle = "Doubler: 3 dBm"
	
	# Trim timeseries
	if trim_time:
		idx_start = find_closest_index(time_ns_full, pulse_range_ns[0])
		idx_end = find_closest_index(time_ns_full, pulse_range_ns[1])
		time_ns = time_ns_full[idx_start:idx_end+1]
		ampl_mV = ampl_mV_full[idx_start:idx_end+1]
	else:
		time_ns = time_ns_full
		ampl_mV = ampl_mV_full
	
	window_step_ns = (time_ns[1]-time_ns[0]) * window_step_points
	
	#===================== Perform zero crossing analysis ==========================
	
	tzc = find_zero_crossings(time_ns, ampl_mV)
	periods = np.diff(tzc)
	freqs = (1/periods)/2

	t_freqs = tzc[:-1] + periods/2
	
	
	return {'fit_freqs': freqs, 'fit_times':t_freqs, 'time_orig':time_ns, 'ampl_orig':ampl_mV}

def windowed_freq_analysis_linear(dframe, pulse_range_ns = (10, 48)):
	
	print(f"Beginning windowed fit (linear).")
	t0 = time.time()
	
	time_ns_full = list(dframe['Time']*1e9)
	ampl_mV_full = list(dframe['Ampl']*1e3)
	supertitle = "Doubler: 3 dBm"
	
	# Trim timeseries
	if trim_time:
		idx_start = find_closest_index(time_ns_full, pulse_range_ns[0])
		idx_end = find_closest_index(time_ns_full, pulse_range_ns[1])
		time_ns = time_ns_full[idx_start:idx_end+1]
		ampl_mV = ampl_mV_full[idx_start:idx_end+1]
	else:
		time_ns = time_ns_full
		ampl_mV = ampl_mV_full
	
	window_step_ns = (time_ns[1]-time_ns[0]) * window_step_points
	
	#===================== Perform windowed gaussian fit ==========================
	num_fail = 0
	
	# Initial guess
	freq = 4.825
	param = [50, 2*3.14159*freq, 0, 0, 0]
	
	# Set bounds
	lower = (10, 2*pi*2, -pi, -10, -5)
	upper = (220, 2*pi*8, pi, 10, 5)
	bounds = (lower, upper)
	
	# Initialize window
	window = [time_ns[1], window_size_ns+time_ns[0]]
	
	# Prepare output arrays
	fit_times = []
	fit_omegas = []
	fit_covs = []
	
	fit_ampls = []
	fit_phis = []
	fit_ms = []
	fit_offs = []
	
	# Loop over data
	count = 0
	while True:
		
		count += 1
		
		# Find start and end index
		idx0 = find_closest_index(time_ns, window[0])
		idxf = find_closest_index(time_ns, window[1])
		
		# Get fit region
		time_ns_fit = np.array(time_ns[idx0:idxf])
		ampl_mV_fit = np.array(ampl_mV[idx0:idxf])
		
		# Perform fit
		try:
			param, param_cov = curve_fit(linear_sine, time_ns_fit, ampl_mV_fit, p0=param, bounds=bounds)
		except:
			num_fail += 1
			print(param)
			fit_vals = linear_sine(time_ns_fit, param[0], param[1], param[2], param[3], param[4])
			
			if num_fail <= 3:
				fig1 = plt.figure(1)
				gs = fig1.add_gridspec(2, 1)
				ax1a = fig1.add_subplot(gs[0, 0])
				ax1b = fig1.add_subplot(gs[1, 0])
				
				ax1a.plot(time_ns, ampl_mV)
				ax1a.grid(True)
				
				ax1b.plot(time_ns_fit, ampl_mV_fit, linestyle='--', marker='.', color=(0.6, 0, 0), alpha=0.5, label="Measured data")
				ax1b.plot(time_ns_fit, fit_vals, linestyle='--', marker='.', color=(0, 0, 0.6), alpha=0.5, label="Fit")
				ax1b.set_title("Fit Sections")
				ax1b.grid(True)
				ax1b.legend()
				plt.tight_layout()
				plt.show()
			
			# Update window
			window[0] += window_step_ns #window_size_ns*window_step_fraction
			window[1] += window_step_ns #window_size_ns*window_step_fraction
			
			continue
			
		param_err = np.sqrt(np.diag(param_cov))
		
		# Save data
		fit_times.append((time_ns_fit[0]+time_ns_fit[-1])/2)
		fit_omegas.append(param[1])
		fit_covs.append(param_err[1])
		fit_ampls.append(param[0])
		fit_phis.append(param[2])
		fit_ms.append(param[3])
		fit_offs.append(param[4])
		
		# Update window
		window[0] += window_step_ns #window_size_ns*window_step_fraction
		window[1] += window_step_ns #window_size_ns*window_step_fraction
		
		# Break condition
		if window[1] > time_ns[-1]:
			break
	
	# convert from angular frequency
	fit_freqs = np.array(fit_omegas)/2/np.pi
	fit_err = np.array(fit_covs)/2/np.pi
	
	print(f"    Completed in {time.time()-t0} s")
	
	return {'fit_freqs': fit_freqs, 'fit_err':fit_err, 'fit_times':fit_times, 'raw_times':time_ns, 'raw_ampl':ampl_mV, 'fit_offs':fit_offs, 'fit_ampls':fit_ampls, 'fit_phis':fit_phis, 'fit_ms':fit_ms, 'bounds':bounds}

def add_freqfit_data(dpacket, ax_freq, ax_ampl, color=blue):
	
	fit_times = dpacket['fit_times']
	fit_freqs = dpacket['fit_freqs']
	fit_err = dpacket['fit_err']
	
	ax_freq.plot(fit_times, fit_freqs, marker='.', linestyle=':', alpha=0.8, color=color)
	ax_freq.fill_between(fit_times, fit_freqs-fit_err, fit_freqs+fit_err, color=color, alpha=0.1)
	
	ax_ampl.plot(dpacket['raw_times'], dpacket['raw_ampl'], linestyle=':', marker='.', color=color, alpha=0.5)

def add_fitparam_data(dpacket, ax2a, ax2b, ax2c, ax2d, color=(0.6, 0, 0.2)):
	ax2a.plot(dpacket['fit_ampls'], linestyle='--', marker='.', color=color)
	ax2d.plot(dpacket['fit_phis'], linestyle='--', marker='.', color=color)
	
	try:
		ax2c.plot(dpacket['fit_ms'], linestyle='--', marker='.', color=color)
		ax2b.plot(dpacket['fit_offs'], linestyle='--', marker='.', color=color)
	except:
		pass

# #====================== Crunch numbers ========================

df_analyzed = []
for df in df_100mV_3dBm_df:
	# df_analyzed.append(windowed_freq_analysis_linear(df))
	df_analyzed.append(zero_cross_freq_analysis(df, pulse_range_ns=trim_times) )

# #====================== Perform plotting ========================


t = df_analyzed[0]['time_orig']
y = df_analyzed[0]['ampl_orig']
freqs = df_analyzed[0]['fit_freqs']
t_freqs = df_analyzed[0]['fit_times']

fig1 = plt.figure(1, figsize=(14, 6.5))
gs = fig1.add_gridspec(2, 1)
ax1a = fig1.add_subplot(gs[0, 0])
ax1b = fig1.add_subplot(gs[1, 0])

ax1a.plot(t, y, linestyle=':', marker='.', color=(0, 0, 0.65))
ax1a.set_title("Full Waveform")
ax1a.set_xlabel("Time")
ax1a.set_ylabel("Amplitude")
ax1a.grid(True)
xl = ax1a.get_xlim()

ax1b.plot(t_freqs, freqs, linestyle=':', marker='.', color=(0, 0.6, 0), alpha=1)
ax1b.set_title("Full Waveform")
ax1b.set_xlabel("Time")
ax1b.set_ylabel("Frequency")
ax1b.grid(True)
ax1b.set_xlim(xl)
ax1b.set_ylim([4.5, 5.5])

fig1.tight_layout()

plt.show()