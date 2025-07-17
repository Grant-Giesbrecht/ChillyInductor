import numpy as np
from scipy.signal import hilbert
import pylogfile.base as plf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

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
	
	def __init__(self, df:pd.DataFrame, power_dBm:float, log:plf.LogPile, trim_time:bool=False, time_bounds:list=[], IF_GHz:float=0.1, IF_chirp_tol_GHz:float=0.015, do_val_trim:bool=True, do_fit:bool=True):
		
		self.IF_GHz = IF_GHz
		self.IF_chirp_tol_GHz = IF_chirp_tol_GHz
		self.do_val_trim = do_val_trim
		
		self.time_ns = np.array(df['Time']*1e9)
		self.wave_mV = np.array(df['Ampl']*1e3)
		
		self.log = log
		
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
			
			self.log.info(f"Trimmed time for dataset with P=>{power_dBm}< dBm.")
		
		# Calculate envelope
		hilbert_temp = hilbert(self.wave_mV)
		self.envelope = np.abs(hilbert_temp)
		
		self.calc_zero_cross()
		
		self.z_time_trim = []
		self.z_freq_trim = []
		self.fit_results = []
		self.fit_curve = []
		if do_fit:
			self.fit_data()
		
		self.power_dBm = power_dBm
		
		self.log.info(f"Analysis complete for dataset with P=>{power_dBm}< dBm.")
	
	def calc_zero_cross(self):
		''' Calculate frequency using zero crossing analysis.'''
		
		tzc = find_zero_crossings(self.time_ns, self.wave_mV)
		N_avg = 3
		
		# Select every other zero crossing to see full periods and become insensitive to y-offsets.
		tzc_fullperiod = tzc[::2*N_avg]
		periods = np.diff(tzc_fullperiod)
		self.zcf_freq = (1/periods)*N_avg

		self.zcf_time = tzc_fullperiod[:-1] + periods/2
	
	def fit_data(self):
		''' Fit an SPM model to the experiment data. '''
		
		if self.do_val_trim:
			# Re-trim the data so that anything outside of an acceptable frequency range is removed.
			f_max = self.IF_GHz + self.IF_chirp_tol_GHz
			f_min = self.IF_GHz - self.IF_chirp_tol_GHz

			# Find valid indices
			valid_mask = (self.zcf_freq >= f_min) & (self.zcf_freq <= f_max)
			if not np.any(valid_mask):
				raise ValueError(f"No values found in wave_mV between f_min ({f_min}) and f_max ({f_max}).")

			start_index = np.argmax(valid_mask)  # first True index
			end_index = len(valid_mask) - np.argmax(valid_mask[::-1])  # last True index

			# Trim arrays
			self.z_time_trim = self.zcf_time[start_index:end_index]*1e-9
			self.z_freq_trim = self.zcf_freq[start_index:end_index]*1e9
		else:
			self.z_time_trim = self.zcf_time
			self.z_freq_trim = self.zcf_freq
		
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