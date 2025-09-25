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

window_size_ns = 3.5
window_step_points = 10

# parser = argparse.ArgumentParser()
# parser.add_argument('filename')
# parser.add_argument('-f', '--fft', help='show fft of data', action='store_true')
# parser.add_argument('--trimdc', help='Removes DC from FFT data', action='store_true')
# parser.add_argument('--f1', help='Plot F1 file over FFT', action='store_true')
# args = parser.parse_args()

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

def linear_sine (x, ampl, omega, phi, m, offset):
	return np.sin(omega*x-phi)*(ampl + m*x) + offset

def zero_cross_freq_analysis(dframe, pulse_range_ns = (10, 48)):
	
	print(f"Beginning windowed fit (linear).")
	t0 = time.time()
	
	time_ns_full = list(dframe['Time']*1e9)
	ampl_mV_full = list(dframe['Ampl']*1e3)
	supertitle = "Doubler: 3 dBm"
	
	# Trim timeseries
	if pulse_range_ns is not None:
		idx_start = find_closest_index(time_ns_full, pulse_range_ns[0])
		idx_end = find_closest_index(time_ns_full, pulse_range_ns[1])
		time_ns = time_ns_full[idx_start:idx_end+1]
		ampl_mV = ampl_mV_full[idx_start:idx_end+1]
	else:
		time_ns = time_ns_full
		ampl_mV = ampl_mV_full
	
	# Get the size of window update
	window_step_ns = (time_ns[1]-time_ns[0]) * window_step_points
	
	#===================== Perform zero crossing analysis ==========================
	
	tzc = find_zero_crossings(time_ns, ampl_mV)
	periods = np.diff(tzc)
	freqs = (1/periods)/2

	t_freqs = tzc[:-1] + periods/2
	
	# Return 
	return {'fit_freqs': freqs, 'fit_times':t_freqs}

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

def process_file(filename:str, pulse_range_ns = (10, 48)):
	''' 
	'''
	
	# Read file
	try:
		df = pd.read_csv(filename, skiprows=4, encoding='utf-8')
	except:
		print(f"Failed to find file {filename}. Aborting.")
		return None
	
	# Create local variables
	time_ns_full = np.array(df['Time'])
	ampl_mV_full = np.array(df['Ampl'])
	
	# Trim timeseries
	if pulse_range_ns is not None:
		idx_start = find_closest_index(time_ns_full, pulse_range_ns[0])
		idx_end = find_closest_index(time_ns_full, pulse_range_ns[1])
		t_si = time_ns_full[idx_start:idx_end+1]
		v_si = ampl_mV_full[idx_start:idx_end+1]
	else:
		t_si = time_ns_full
		v_si = ampl_mV_full
	
	# Get converted unit version
	time_ns = t_si*1e9
	ampl_mV = v_si*1e3
	
	#===================== Fourier Transform ==========================
	
	# Fourier Transform
	R = 50
	sample_period = (t_si[-1]-t_si[0])/(len(t_si)-1)
	sample_rate = 1/sample_period
	spectrum_double = np.fft.fft(v_si)
	freq_double = np.fft.fftfreq(len(spectrum_double), d=1/sample_rate)
	
	# Get single-ended results
	freq = freq_double[:len(freq_double)//2]
	spectrum_W = (np.abs(spectrum_double[:len(freq_double)//2])**2) / (R * sample_rate) #len(v_si))
	spectrum = 10 * np.log10(spectrum_W*1e3)
	
	#===================== Perform zero crossing analysis ==========================
	
	# Run zero-crossing 
	tzc = find_zero_crossings(time_ns, ampl_mV)
	periods = np.diff(tzc)
	freqs = (1/periods)/2
	t_freqs = tzc[:-1] + periods/2
	
	#===================== Fourier Transform ==========================
	
	# Return parameters
	return (freq, spectrum, t_si, v_si)

# Create figure
fig1 = plt.figure(1, figsize=(8, 8))


gs = fig1.add_gridspec(4, 1)
ax1a = fig1.add_subplot(gs[0:2, 0])
ax1b = fig1.add_subplot(gs[2, 0])
ax1c = fig1.add_subplot(gs[3, 0])

ax1b.plot(freq/1e9, spectrum, linestyle=':', marker='.', color=(0, 0.6, 0), label='FFT')
ax1b.plot(df1_freq, df1_spec, linestyle=':', marker='.', color=(0.56, 0., 0.56), label='Scope spectrum')
ax1b.set_xlabel(f"Frequency (GHz)")
ax1b.set_ylabel(f"Power")
ax1b.set_title(f"Fourier Transform")
ax1b.set_xlim([4.8, 4.85])
ax1b.grid(True)

ax1c.plot(df1_freq, df1_spec, linestyle=':', marker='.', color=(0.56, 0., 0.56), label='Scope spectrum')
ax1c.set_xlabel(f"Frequency (GHz)")
ax1c.set_ylabel(f"Power")
ax1c.set_title(f"Scope Fourier Transform")
ax1c.set_xlim([4.8, 4.85])
ax1c.grid(True)

ax1a.set_title(f"Time Domain Signal")

	
ax1a.plot(t, v, linestyle=':', marker='.', color=(0, 0, 0.65))
ax1a.set_xlabel("Time (ns))")
ax1a.set_ylabel("Voltage (mV)")
ax1a.grid(True)

mplcursors.cursor(multiple=True)

fig1.tight_layout()

plt.show()