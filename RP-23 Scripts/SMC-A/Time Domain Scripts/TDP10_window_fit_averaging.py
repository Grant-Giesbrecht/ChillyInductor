import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz
import time
import os
from pylogfile.base import markdown

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

HILBERT_NORMALIZED = 0
LINEAR_SINE_FIT = 1
GUIDED_SINE_FIT = 2

#====================== Program options =================

pi = 3.1415926535
merge = True
include_3rd = False

trim_time = True
fit_method = HILBERT_NORMALIZED

# Window fit options
window_size_ns = 2
window_step_points = 10


#====================== Load data =================

# DATADIR = "/Volumes/M4 PHD/13FebTimeDomain"
DATADIR = os.path.join("G:", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "13_Feb_2025")

df_100mV_3dBm_df = []
df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,4GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8'))
df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,4GHZ_3dBm00001.txt", skiprows=4, encoding='utf-8'))
df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,4GHZ_3dBm00002.txt", skiprows=4, encoding='utf-8'))
df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,4GHZ_3dBm00003.txt", skiprows=4, encoding='utf-8'))
df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,4GHZ_3dBm00004.txt", skiprows=4, encoding='utf-8'))
shifts_ns = [0.147, 0.089, 0.008, -0.009]
shifts_ns = [0.163, 0.135, 0.048, -0.020]

# df_100mV_3dBm_df = []
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_3dBm00001.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_3dBm00002.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_3dBm00003.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_3dBm00004.txt", skiprows=4, encoding='utf-8'))

# df_100mV_3dBm_df = []
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_0dBm00000.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_0dBm00001.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_0dBm00002.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_0dBm00003.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,30V-F2,4GHZ_0dBm00004.txt", skiprows=4, encoding='utf-8'))

# df_100mV_3dBm_df = []
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,10V-F2,4GHZ_0dBm00000.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,10V-F2,4GHZ_0dBm00001.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,10V-F2,4GHZ_0dBm00002.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,10V-F2,4GHZ_0dBm00003.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,10V-F2,4GHZ_0dBm00004.txt", skiprows=4, encoding='utf-8'))

# df_100mV_3dBm_df = []
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,20V-F2,4GHZ_0dBm00000.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,20V-F2,4GHZ_0dBm00001.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,20V-F2,4GHZ_0dBm00002.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,20V-F2,4GHZ_0dBm00003.txt", skiprows=4, encoding='utf-8'))
# df_100mV_3dBm_df.append(pd.read_csv(f"{DATADIR}/C1BIAS0,20V-F2,4GHZ_0dBm00004.txt", skiprows=4, encoding='utf-8'))
# shifts_ns = [0.026, 0.055, 0.0565, 0.032]

master_df = pd.concat(df_100mV_3dBm_df)
master_df.sort_values(by='Time')
master_df.reset_index(drop=True, inplace=True)

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
	return np.sin(omega*x-phi)*(ampl + m*(x-x[0])) + offset

def windowed_freq_analysis_norm(dframe, pulse_range_ns = (10, 48)):
	
	print(f"Beginning windowed fit (normalized).")
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
	
	#===================== Perform Hilbert transform ==========================
	# Goal is to normalize sine wave amplitude
	
	# Perform hilbert transform and calculate derivative parameters
	a_signal = hilbert(ampl_mV)
	ampl_env = np.abs(a_signal)
	ampl_norm = ampl_mV/ampl_env
	
	#===================== Perform windowed gaussian fit ==========================
	
	# Initial guess
	freq = 4.825
	param = [1, 2*3.14159*freq, 0]
	# param = [1, 2*3.14159*freq, 0, 0, 0]
	
	# Set bounds
	lower = (0.9, 2*pi*2, -pi)
	upper = (1.1, 2*pi*8, pi)
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
		
		idx0 = find_closest_index(time_ns, window[0])
		idxf = find_closest_index(time_ns, window[1])
		
		time_ns_fit = np.array(time_ns[idx0:idxf])
		ampl_mV_fit = np.array(ampl_norm[idx0:idxf])
		
		# Perform fit
		param, param_cov = curve_fit(flat_sine, time_ns_fit, ampl_mV_fit, p0=param, bounds=bounds)
		param_err = np.sqrt(np.diag(param_cov))
		
		# Save data
		fit_times.append((time_ns_fit[0]+time_ns_fit[-1])/2)
		fit_omegas.append(param[1])
		fit_covs.append(param_err[1])
		fit_ampls.append(param[0])
		fit_phis.append(param[2])
		
		# fit_ms.append(param[3])
		# fit_offs.append(param[4])
		
		# Update window
		window[0] += window_step_ns #window_size_ns*window_step_fraction
		window[1] += window_step_ns #window_size_ns*window_step_fraction
		
		if window[1] > time_ns[-1]:
			break
		
	fit_freqs = np.array(fit_omegas)/2/np.pi
	fit_err = np.array(fit_covs)/2/np.pi
	
	print(f"    Completed in {time.time()-t0} s")
	
	return {'fit_freqs': fit_freqs, 'fit_err':fit_err, 'fit_times':fit_times, 'raw_times':time_ns, 'raw_ampl':ampl_mV, 'fit_ampls':fit_ampls, 'fit_phis':fit_phis, 'fit_ms':fit_ms, 'bounds':bounds}

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

def windowed_freq_analysis_linear_guided(dframe, pulse_range_ns = (10, 48)):
	
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
	
	#===================== Perform Hilbert transform ==========================
	# Goal is to provide tighter bounds on amplitude and slope
	
	# Perform hilbert transform and calculate derivative parameters
	a_signal = hilbert(ampl_mV)
	ampl_env = np.abs(a_signal)
	
	#===================== Perform windowed gaussian fit ==========================
	num_fail = 0
	
	# Initial guess
	freq = 4.825
	param = [50, 2*3.14159*freq, 0, 0, 0]
	
	# Set bounds
	lower = [10, 2*pi*2, -pi, -10, -5]
	upper = [220, 2*pi*8, pi, 10, 5]
	bounds = [lower, upper]
	
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
	ampl_bounds_hi = []
	ampl_bounds_low = []
	slope_bounds_low = []
	slope_bounds_hi = []
	
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
		envl_mV_fit = np.array(ampl_env[idx0:idxf])
		
		# Adjust bounds
		bounds[0][0] = np.min(envl_mV_fit) # Adjust lower bounds
		bounds[1][0] = np.max(envl_mV_fit) # Adjust upper bounds
		
		# Adjust slope
		est_slope_abs = (np.max(envl_mV_fit)-np.min(envl_mV_fit))/(time_ns_fit[-1]-time_ns_fit[0])
		est_delta = np.mean(envl_mV_fit[-5:]) - np.mean(envl_mV_fit[:5])
		slope_sign = (est_delta)/np.abs(est_delta)
		est_slope = est_slope_abs * slope_sign
		slope_bounds = [est_slope * 0.25, est_slope * 1.25]
		
		bounds[0][3] = np.min(slope_bounds)
		bounds[1][3] = np.max(slope_bounds)
		
		# Set initial guess
		param[0] = np.mean([np.max(envl_mV_fit), np.min(envl_mV_fit)])
		param[3] = est_slope
		
		# Perform fit
		try:
			param, param_cov = curve_fit(linear_sine, time_ns_fit, ampl_mV_fit, p0=param, bounds=bounds)
		except Exception as e:
			num_fail += 1
			print(markdown(f"Failed to converge: (>:q{e}<)"))
			
			def print_range(index, name):
				lb=bounds[0][index]
				ub=bounds[1][index]
				val=param[index]
				print(markdown(f"    {name}: >:q{lb}< \< >{val}< \< >{ub}<"))
			
			fit_vals = linear_sine(time_ns_fit, param[0], param[1], param[2], param[3], param[4])
			print_range(0, "Amplitude")
			print_range(1, "Omega")
			print_range(2, "Phi")
			print_range(3, "Slope")
			print_range(4, "Offset")
			
			if num_fail <= 3:
				fig1 = plt.figure(1)
				gs = fig1.add_gridspec(2, 1)
				ax1a = fig1.add_subplot(gs[0, 0])
				ax1b = fig1.add_subplot(gs[1, 0])
				
				ax1a.plot(time_ns, ampl_mV)
				ax1a.grid(True)
				
				ax1b.plot(time_ns_fit, ampl_mV_fit, linestyle='--', marker='.', color=(0.6, 0, 0), alpha=0.5, label="Measured data")
				ax1b.plot(time_ns_fit, envl_mV_fit, linestyle='--', marker='.', color=(1, 0.5, 0.05), alpha=0.5, label="Envelope")
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
		
		# Record bounds
		ampl_bounds_low.append(bounds[0][0])
		ampl_bounds_hi.append(bounds[1][0])
		slope_bounds_low.append(bounds[0][3])
		slope_bounds_hi.append(bounds[1][3])
		
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
	
	return {'fit_freqs': fit_freqs, 'fit_err':fit_err, 'fit_times':fit_times, 'raw_times':time_ns, 'raw_ampl':ampl_mV, 'fit_offs':fit_offs, 'fit_ampls':fit_ampls, 'fit_phis':fit_phis, 'fit_ms':fit_ms, 'bounds':bounds, 'ampl_bounds_low':ampl_bounds_low, 'ampl_bounds_hi':ampl_bounds_hi, 'slope_bounds_hi':slope_bounds_hi, 'slope_bounds_low':slope_bounds_low}

def add_freqfit_data(dpacket, ax_freq, ax_ampl, color=blue, label=""):
	
	fit_times = dpacket['fit_times']
	fit_freqs = dpacket['fit_freqs']
	fit_err = dpacket['fit_err']
	
	ax_freq.plot(fit_times, fit_freqs, marker='.', linestyle=':', alpha=0.8, color=color, label=label)
	ax_freq.fill_between(fit_times, fit_freqs-fit_err, fit_freqs+fit_err, color=color, alpha=0.1)
	
	ax_ampl.plot(dpacket['raw_times'], dpacket['raw_ampl'], linestyle=':', marker='.', color=color, alpha=0.5, label=label)

def add_fitparam_data(dpacket, ax2a, ax2b, ax2c, ax2d, color=(0.6, 0, 0.2)):
	ax2a.plot(dpacket['fit_ampls'], linestyle='--', marker='.', color=color)
	ax2d.plot(dpacket['fit_phis'], linestyle='--', marker='.', color=color)
	
	try:
		ax2c.plot(dpacket['fit_ms'], linestyle='--', marker='.', color=color)
		ax2b.plot(dpacket['fit_offs'], linestyle='--', marker='.', color=color)
	except:
		pass
	
	# Check if guided bounds present
	if 'slope_bounds_low' in dpacket:
		print(f"Found guided bounds")
		tempx = np.linspace(1, len(dpacket['slope_bounds_low'])-1, len(dpacket['slope_bounds_low']) )
		ax2c.fill_between(tempx, dpacket['slope_bounds_low'], dpacket['slope_bounds_hi'], color=color, alpha=0.1)
		ax2a.fill_between(tempx, dpacket['ampl_bounds_low'], dpacket['ampl_bounds_hi'], color=color, alpha=0.1)


# Apply test time-shift
# shifts_ns = [0.026, 0.055, 0.0565, 0.032]
df_100mV_3dBm_trim = []
pulse_range_ns = (10, 48)
for idx, frame in enumerate(df_100mV_3dBm_df):
	
	if idx < len(shifts_ns):
		frame['Time'] += shifts_ns[idx]*1e-9
	
	# Trim timeseries
	if trim_time:
		idx_start = find_closest_index(frame['Time'], pulse_range_ns[0]*1e-9)
		idx_end = find_closest_index(frame['Time'], pulse_range_ns[1]*1e-9)
		time_ns = frame['Time'][idx_start:idx_end+1]
		ampl_mV = frame['Ampl'][idx_start:idx_end+1]
	else:
		time_ns = frame['Time']
		ampl_mV = frame['Ampl']
	df_100mV_3dBm_trim.append(pd.DataFrame({'Time':time_ns, 'Ampl':ampl_mV}))

# #====================== Crunch numbers ========================


if fit_method == HILBERT_NORMALIZED:
	master_analyzed = windowed_freq_analysis_norm(master_df)
elif fit_method == LINEAR_SINE_FIT:
	master_analyzed = windowed_freq_analysis_linear(master_df)
elif fit_method == GUIDED_SINE_FIT:
	master_analyzed = windowed_freq_analysis_linear_guided(master_df)

# #====================== Perform plotting ========================

fig1 = plt.figure(figsize=(8,6))
gs = fig1.add_gridspec(2, 1)
ax1a = fig1.add_subplot(gs[0, 0])
ax1b = fig1.add_subplot(gs[1, 0])

add_freqfit_data(master_analyzed, ax1a, ax1b, color=(0, 0, 0.7))

ax1a.scatter([15, 20, 25, 30, 35, 40], [4.8515, 4.8821, 4.8742, 4.8933, 4.8931, 4.8816], marker='o', color=(0, 0.65, 0))

ax1a.set_xlabel("Time (ns)")
ax1a.set_ylabel("Frequency (GHz)")
ax1a.set_title("Fit Frequency")
ax1a.grid(True)

ax1b.set_xlabel("Time (ns)")
ax1b.set_ylabel("Amplitude (mV)")
ax1b.set_title("Time Domain Data")
ax1b.grid(True)

if fit_method == HILBERT_NORMALIZED:
	fig2 = plt.figure(figsize=(5, 6))
	gs2 = fig2.add_gridspec(2, 1)
	ax2a = fig2.add_subplot(gs2[0, 0])
	ax2b = None
	ax2c = None
	ax2d = fig2.add_subplot(gs2[1, 0])
else:
	fig2 = plt.figure(figsize=(5, 6))
	gs2 = fig2.add_gridspec(4, 1)
	ax2a = fig2.add_subplot(gs2[0, 0])
	ax2b = fig2.add_subplot(gs2[1, 0])
	ax2c = fig2.add_subplot(gs2[2, 0])
	ax2d = fig2.add_subplot(gs2[3, 0])



add_fitparam_data(master_analyzed, ax2a, ax2b, ax2c, ax2d, color=(0, 0, 0.7))

ax2a.set_title("Amplitude")
ax2d.set_title("phis")
if fit_method == LINEAR_SINE_FIT or fit_method == GUIDED_SINE_FIT:
	ax2b.set_title("Offset")
	ax2c.set_title("Slope")


ax2a.set_ylabel("Amplitude (mV)")
ax2a.grid(True)
ax2d.set_ylabel("Phase shift (rad)")
ax2d.grid(True)
ax2d.set_xlabel("Fit Index")
if fit_method == LINEAR_SINE_FIT or fit_method == GUIDED_SINE_FIT:
	ax2b.set_ylabel("Offset (mV)")
	ax2b.grid(True)
	ax2c.set_ylabel("Slope (mV/ns)")
	ax2c.grid(True)

if fit_method == HILBERT_NORMALIZED or fit_method == LINEAR_SINE_FIT:
	ax2a.axhline(y=master_analyzed['bounds'][0][0], color=(0,0,0), linestyle='--')
	ax2a.axhline(y=master_analyzed['bounds'][1][0], color=(0,0,0), linestyle='--')
ax2d.axhline(y=master_analyzed['bounds'][0][2], color=(0,0,0), linestyle='--')
ax2d.axhline(y=master_analyzed['bounds'][1][2], color=(0,0,0), linestyle='--')
if fit_method == LINEAR_SINE_FIT or fit_method == GUIDED_SINE_FIT:
	ax2b.axhline(y=master_analyzed['bounds'][0][4], color=(0,0,0), linestyle='--')
	ax2b.axhline(y=master_analyzed['bounds'][1][4], color=(0,0,0), linestyle='--')
if fit_method == LINEAR_SINE_FIT:
	ax2c.axhline(y=master_analyzed['bounds'][0][3], color=(0,0,0), linestyle='--')
	ax2c.axhline(y=master_analyzed['bounds'][1][3], color=(0,0,0), linestyle='--')

fig3 = plt.figure(figsize=(10,4.5))
gs = fig3.add_gridspec(1, 1)
ax3a = fig3.add_subplot(gs[0, 0])	

# Add to plot
cmap = get_colormap_colors('plasma', len(df_100mV_3dBm_trim))
for idx, frame in enumerate(df_100mV_3dBm_trim):
	ax3a.plot(frame['Time']*1e9, frame['Ampl'], linestyle=':', marker='.', color=cmap[idx], label=f"idx={idx}")

ax3a.set_xlabel("Time (ns)")
ax3a.set_ylabel("Amplitude (mV)")
ax3a.set_title("Waveform Alignment")
ax3a.grid(True)
ax3a.legend()

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
plt.show()

