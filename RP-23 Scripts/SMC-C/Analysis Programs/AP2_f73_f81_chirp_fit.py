#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from colorama import Fore, Style
import time
import os
import sys
import argparse
import mplcursors
from graf.base import sample_colormap
from scipy.signal import hilbert
import pylogfile.base as plf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

log = plf.LogPile()

path = "/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-C July Campaign/1-July-2025 TD Measurements/"

trim_time = True
time_bounds = [[-1300, -800]]

def find_closest_index(lst, X):
	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
	return closest_index

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

class AP2Analyzer:
	
	def __init__(self, df:pd.DataFrame, power_dBm:float):
		
		self.time_ns = np.array(df['Time']*1e9)
		self.wave_mV = np.array(df['Ampl']*1e3)
		
		if trim_time:
			new_time_ns = []
			new_wave_mV = []
			
			for prange in time_bounds:
				idx_start = find_closest_index(self.time_ns, prange[0])
				idx_end = find_closest_index(self.time_ns, prange[1])
				new_time_ns = np.concatenate([new_time_ns, self.time_ns[idx_start:idx_end+1]])
				new_wave_mV = np.concatenate([new_wave_mV, self.wave_mV[idx_start:idx_end+1]])
			
			self.time_ns = new_time_ns
			self.wave_mV = new_wave_mV
			
			log.info(f"Trimmed time for dataset with P=>{power_dBm}< dBm.")
		
		# Calculate envelope
		hilbert_temp = hilbert(self.wave_mV)
		self.envelope = np.abs(hilbert_temp)
		
		self.calc_zero_cross()
		
		self.z_time_trim = []
		self.z_freq_trim = []
		self.fit_results = []
		self.fit_curve = []
		self.fit_data()
		
		self.power_dBm = power_dBm
		
		log.info(f"Analysis complete for dataset with P=>{power_dBm}< dBm.")
	
	def calc_zero_cross(self):
		''' Calculate frequency using zero crossing analysis.'''
		
		tzc = find_zero_crossings(self.time_ns, self.wave_mV)
		N_avg = 2
		
		# Select every other zero crossing to see full periods and become insensitive to y-offsets.
		tzc_fullperiod = tzc[::2*N_avg]
		periods = np.diff(tzc_fullperiod)
		self.zcf_freq = (1/periods)*N_avg

		self.zcf_time = tzc_fullperiod[:-1] + periods/2
	
	def fit_data(self):
		''' Fit an SPM model to the experiment data. '''
		
		# Re-trim the data so that anything outside of an acceptable frequency range is removed.
		f_max = 0.115
		f_min = 0.085

		# Find valid indices
		valid_mask = (self.zcf_freq >= f_min) & (self.zcf_freq <= f_max)
		if not np.any(valid_mask):
			raise ValueError("No values found in wave_mV between wave_min and wave_max.")

		start_index = np.argmax(valid_mask)  # first True index
		end_index = len(valid_mask) - np.argmax(valid_mask[::-1])  # last True index

		# Trim arrays
		self.z_time_trim = self.zcf_time[start_index:end_index]*1e-9
		self.z_freq_trim = self.zcf_freq[start_index:end_index]*1e9
		
		
		omega_0 = 100e6
		omega_tolerance = 5e6
		
		t_center = np.mean(self.z_time_trim)
		t_center_tol = 35e-9
		
		
		# --- Define the SPM-based model function ---
		def spm_model(t, w0, t0, K1, K2, K3):
			t_convert = (t-t0)
			return w0 + (4 * np.pi * K1) / (K2 * K3**2) * (t_convert) * np.exp(-((t_convert)**2) / K3**2)
		
		# Interpolate amplitude_2 onto the time_ns grid
		interp_func = interp1d(self.time_ns*1e-9, self.envelope, kind='linear', bounds_error=False, fill_value=0)
		weights = interp_func(self.z_time_trim)
		
		# Avoid division by zero — add a small constant
		weights += 1e-6
		
		# Convert amplitude weights to standard deviations (smaller sigma => higher weight)
		sigma = 1.0 / weights
		
		# Set curve bounds
		lower_bounds = [omega_0-omega_tolerance,  t_center-t_center_tol,    0.05e-5,    0.05,    10e-9]
		# lower_bounds = [omega_0-omega_tolerance,  t_center-t_center_tol,    0.5e-5,    0.05,    30e-9]
		# upper_bounds = [ omega_0+omega_tolerance, t_center+t_center_tol, 5e-5, 0.1, 70e-9]
		upper_bounds = [ omega_0+omega_tolerance, t_center+t_center_tol, 5e-3, 0.1, 120e-9]
		
		# Set initial guess
		initial_guess = [omega_0, t_center, 1.2e-5, 0.0625, 50e-9]
		
		# Fit the curve
		popt, pcov = curve_fit(spm_model, self.z_time_trim, self.z_freq_trim, sigma=sigma, absolute_sigma=True, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
		
		print(f"Sigma = {sigma}")
		print(f"popt = {popt}")
		
		self.fit_results = popt
		
		# Plot results
		self.fit_curve = spm_model(self.z_time_trim, *popt)
		
		# plt.plot(self.z_time_trim, self.z_freq_trim, label='Original Data', linestyle=':', marker='.', color=(0, 0.6, 0.6))
		# plt.plot(self.z_time_trim, self.fit_curve, label='Fit', linestyle='--', linewidth=0.75, color=(0.8, 0, 0))
		# plt.legend()
		# plt.xlabel("Time (ns)")
		# plt.ylabel("Amplitude (mV)")
		# plt.title("Weighted Curve Fit")
		# plt.show()


#=====================================================================#
#           Load Doubler Data

try:
	fn = "C1RP23C_f73_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_73 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data73 = AP2Analyzer(df_73, -13.52)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f74_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_74 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data74 = AP2Analyzer(df_74, -6)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f75_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_75 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data75 = AP2Analyzer(df_75, 0)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f76_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_76 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data76 = AP2Analyzer(df_76, 6)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f77_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_77 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data77 = AP2Analyzer(df_77, 12)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f78_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_78 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data78 = AP2Analyzer(df_78, -3)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f79_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_79 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data79 = AP2Analyzer(df_79, 3)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f80_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_80 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data80 = AP2Analyzer(df_80, 9)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f81_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_81 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data81 = AP2Analyzer(df_81, 15)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

#=====================================================================#
#           Load Traditional Data

try:
	fn = "C1RP23C_f82_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_82 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data82 = AP2Analyzer(df_82, -24.74)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f83_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_83 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data83 = AP2Analyzer(df_83, -18)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f84_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_84 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data84 = AP2Analyzer(df_84, -12)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f85_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_85 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data85 = AP2Analyzer(df_85, -6)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f86_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_86 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data86 = AP2Analyzer(df_86, 0)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f87_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_87 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data87 = AP2Analyzer(df_87, 6)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f88_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_88 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data88 = AP2Analyzer(df_88, 12)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

class AutoColorMap:
	
	def __init__(self, color_data:list, N, log:plf.LogPile):
		
		self.N = N
		self.color_data = color_data
		
		self.idx = 0
		
		self.log = log
	
	def __call__(self):
		try:
			cd = self.color_data[self.idx]
			self.idx += 1
			if self.idx >= self.N:
				self.idx = 0
				self.log.debug(f"Colormap auto-reset.")
		except:
			self.idx = 0
			cd = (0, 0, 0)
		
		return cd
	
	def reset(self):
		self.idx = 0

# Prepare data list
data_list = [data73, data74, data78, data75, data79, data76, data80, data77, data81]
data_list_trad = [data82, data83, data84, data85, data86, data87, data88]

#=====================================================================#
#           Plot Doubler Data

# Prepare color map
N = 9
cmdata = sample_colormap('viridis', N=N)
cmap = AutoColorMap(cmdata, N, log)

# Prepare alpha setting
ALP = 0.4

#------------------------------
# Figure 1

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

for dat in data_list:
	c = cmap()
	ax1a.plot(dat.time_ns, dat.wave_mV, linestyle=':', marker='.', label=f"P = {dat.power_dBm} dBm", color=c, alpha=ALP)
	ax1a.plot(dat.time_ns, dat.envelope, linestyle='-', label=f"Envelope, P = {dat.power_dBm} dBm", color=c, alpha=ALP, linewidth=0.5)
cmap.reset()
ax1a.legend()
ax1a.grid(True)


#------------------------------
# Figure 2

fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

ALP=0.8
for dat in data_list:
	c = cmap()
	ax2a.plot(dat.zcf_time, dat.zcf_freq, linestyle=':', marker='.', label=f"Waveform, P = {dat.power_dBm} dBm", color=c, alpha=ALP, markersize=10)
cmap.reset()
# ax2a.legend()
ax2a.grid(True)
ax2a.set_ylim([0.085, 0.115])

#------------------------------
# Figure 5

fig5 = plt.figure(5)
gs5 = fig5.add_gridspec(2, 2)
ax5a = fig5.add_subplot(gs5[0, 0])
ax5b = fig5.add_subplot(gs5[1, 0])
ax5c = fig5.add_subplot(gs5[0, 1])
ax5d = fig5.add_subplot(gs5[1, 1])

ALP=0.8
powers = []
K1s = []
lambdas = []
sigmas = []
for dat in data_list:
	powers.append(dat.power_dBm)
	K1s.append(dat.fit_results[2])
	lambdas.append(dat.fit_results[3])
	sigmas.append(dat.fit_results[4])

ax5a.plot(powers, K1s, linestyle=':', marker='+', color=(0, 0.6, 0.2))
ax5a.grid(True)
ax5a.set_xlabel("Power (dBm)")
ax5a.set_ylabel(f"L n2 I0")

ax5b.plot(powers, lambdas, linestyle=':', marker='+', color=(0, 0.6, 0.2))
ax5b.grid(True)
ax5b.set_xlabel("Power (dBm)")
ax5b.set_ylabel(f"Lambda")

ax5c.plot(powers, sigmas, linestyle=':', marker='+', color=(0, 0.6, 0.2))
ax5c.grid(True)
ax5c.set_xlabel("Power (dBm)")
ax5c.set_ylabel(f"Sigma")

fig5.tight_layout()

#=====================================================================#
#           Plot Traditional Data

# Prepare color map
N = 7
cmdata = sample_colormap('magma', N=N)
cmap = AutoColorMap(cmdata, N, log)

# Prepare alpha setting
ALP = 0.4

#------------------------------
# Figure 3

fig3 = plt.figure(3)
gs3 = fig3.add_gridspec(1, 1)
ax3a = fig3.add_subplot(gs3[0, 0])

for dat in data_list_trad:
	ax3a.plot(dat.time_ns, dat.wave_mV, linestyle=':', marker='x', label=f"P = {dat.power_dBm} dBm", color=cmap(), alpha=ALP)
cmap.reset()
ax3a.legend()
ax3a.grid(True)


#------------------------------
# Figure 4

fig4 = plt.figure(4)
gs4 = fig4.add_gridspec(1, 1)
ax4a = fig4.add_subplot(gs4[0, 0])

ALP=0.8
for dat in data_list_trad:
	ax4a.plot(dat.zcf_time, dat.zcf_freq, linestyle=':', marker='.', label=f"P = {dat.power_dBm} dBm", color=cmap(), alpha=ALP, markersize=10)
cmap.reset()
# ax2a.legend()
ax4a.grid(True)
ax4a.set_ylim([0.085, 0.115])







mplcursors.cursor(multiple=True)

plt.show()
