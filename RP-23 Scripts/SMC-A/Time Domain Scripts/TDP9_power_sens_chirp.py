import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz

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
use_lpfilter = False

# Window fit options
window_size_ns = 3.5
# window_step_fraction = 0.25
window_step_points = 2

#====================== Load data =================

DATADIR = "/Volumes/M4 PHD/13FebTimeDomain"

df_100mV_3dBm_collection = []
df_100mV_3dBm_r0 = pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,4GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')
df_100mV_3dBm_r1 = pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,4GHZ_3dBm00001.txt", skiprows=4, encoding='utf-8')
df_100mV_3dBm_r2 = pd.read_csv(f"{DATADIR}/C1BIAS0,1V-F2,4GHZ_3dBm00002.txt", skiprows=4, encoding='utf-8')

def find_closest_index(lst, X):
	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
	return closest_index

red = (0.6, 0, 0)
blue = (0, 0, 0.7)
green = (0, 0.5, 0)
orange = (1, 0.5, 0.05)

cmap = get_colormap_colors('plasma', 3)




def flat_sine (x, ampl, omega, phi):
	return np.sin(omega*x-phi)*ampl

def linear_sine (x, ampl, omega, phi, m, offset):
	return np.sin(omega*x-phi)*(ampl + m*x) + offset

def windowed_freq_analysis(dframe, pulse_range_ns = (10, 48)):
	
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
	# lower = (0.9, 2*pi*2, -pi, -0.2, -0.1)
	# upper = (1.1, 2*pi*8, pi, 0.2, 0.1)
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
	
	return {'fit_freqs': fit_freqs, 'fit_err':fit_err, 'fit_times':fit_times, 'raw_times':time_ns, 'raw_ampl':ampl_mV}

def add_data(dpacket, ax_freq, ax_ampl, color=blue):
	
	fit_times = dpacket['fit_times']
	fit_freqs = dpacket['fit_freqs']
	fit_err = dpacket['fit_err']
	
	ax_freq.plot(fit_times, fit_freqs, marker='.', linestyle=':', alpha=0.2, color=color)
	ax_freq.fill_between(fit_times, fit_freqs-fit_err, fit_freqs+fit_err, color=color, alpha=0.3)
	
	ax_ampl.plot(dpacket['raw_times'], dpacket['raw_ampl'], linestyle=':', marker='.', color=color, alpha=0.2)

# #====================== Crunch numbers ========================

frame0 = windowed_freq_analysis(df_100mV_3dBm_r0)
frame1 = windowed_freq_analysis(df_100mV_3dBm_r1)
frame2 = windowed_freq_analysis(df_100mV_3dBm_r2)


# #====================== Perform plotting ========================

fig1 = plt.figure(figsize=(8,6))
gs = fig1.add_gridspec(2, 1)

ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[1, 0])
# ax2 = fig1.add_subplot(gs[1, 0])
# ax1 = fig1.add_subplot(gs[2:4, 0])

add_data(frame0, ax0, ax1, color=cmap[0])
add_data(frame1, ax0, ax1, color=cmap[1])
add_data(frame2, ax0, ax1, color=cmap[2])

# ax0.plot()
# ax0.fill_between(fit_times, fit_freqs-fit_err, fit_freqs+fit_err, color=blue, alpha=0.3)
# ax1.plot(time_ns[1:], inst_freq/1e9, linestyle=':', marker='.', alpha=0.2, color=blue)
# ax1.plot(time_ns[1:], inst_norm_freq/1e9, linestyle=':', marker='.', alpha=0.2, color=green, label='Hilbert 2')

ax0.set_xlabel("Time (ns)")
ax0.set_ylabel("Frequency (GHz)")
ax0.set_title("Fit Frequency")
ax0.grid(True)
# ax0.legend()

# ax2.set_xlabel("Time (ns)")
# ax2.set_ylabel("Amplitude (mV)")
# ax2.set_title("Normalized Signal")
# ax2.grid(True)
# ax2.set_ylim([-1.5, 1.5])

ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Amplitude (mV)")
ax1.set_title("Time Domain Data")
ax1.grid(True)

plt.tight_layout()

# fig2 = plt.figure(figsize=(8,6))
# gs = fig2.add_gridspec(4, 1)

# ax0 = fig2.add_subplot(gs[0, 0])
# ax1 = fig2.add_subplot(gs[1, 0])
# ax2 = fig2.add_subplot(gs[2, 0])
# ax3 = fig2.add_subplot(gs[3, 0])


# # ax0.plot(fit_times, fit_ms)
# # ax0.set_title("M")

# # ax1.plot(fit_times, fit_offs)
# # ax1.set_title("Offset")

# ax2.plot(fit_times, fit_ampls)
# ax2.set_title("Amplitude")

# ax3.plot(fit_times, fit_phis)
# ax3.set_title("Phi")

# plt.tight_layout()

plt.show()





