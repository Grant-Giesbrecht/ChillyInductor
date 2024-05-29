import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from dataclasses import dataclass
import sys

analysis_file = "E:\ARC0 PhD Data\RP-22 Lk Dil Fridge 2024\Data\SMC-A Downconversion v1\dMS1_28May2024_DC1V0_r1.hdf"

@dataclass
class PowerSpectrum:
	
	f_rf:float = None
	f_lo:float = None
	
	p_rf:float = None
	p_lo:float = None
	
	p_lo2:float = None
	p_lo3:float = None
	
	p_rf2:float = None
	p_rf3:float = None
	
	p_mx1L:float = None
	p_mx1H:float = None
	p_mx2L:float = None
	p_mx2H:float = None

##--------------------------------------------
# Read HDF5 File

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	source_script = fh['conditions']['source_script'][()]
	conf_json = fh['conditions']['conf_json'][()]
	
	f_rf = fh['dataset']['freq_rf_GHz'][()]
	f_lo = fh['dataset']['freq_lo_GHz'][()]
	c = fh['dataset']['power_RF_dBm'][()]
	d = fh['dataset']['power_LO_dBm'][()]
	waveform_f_Hz = fh['dataset']['waveform_f_Hz'][()]
	waveform_s_dBm = fh['dataset']['waveform_s_dBm'][()]
	waveform_rbw_Hz = fh['dataset']['waveform_rbw_Hz'][()]

t_hdfr = time.time()-t_hdfr_0
print(f"Read HDF file in {t_hdfr} sec.")

##---------------------------------------------------
# Make basic plot

# Define model function to be used to fit to the data above:
def gauss(x, *p):
	A, mu, sigma = p
	return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def get_gaussian_fit(x_data, y_data, x_fit) -> list:
	''' Fits a gaussian to the provided data and returns the interpolated Y points'''
	
	# Normalize Y data
	norm_fact = min(y_data)
	y_data_norm = y_data - norm_fact
	
	# Coefficient Guess
	p0 = [1, 0, 1]
	
	# Run Fit
	coeff, var_matrix = curve_fit(gauss, x_data, y_data_norm, p0=p0, maxfev=10000)
	
	# Return calculated Y values
	return gauss(x_fit, *coeff) + norm_fact

def power_at(waveform_f_Hz, waveform_s_dBm, f_target, bandwidth):
	''' Returns the power at a given frequency '''
	
	if bandwidth is None:
		bandwidth = 11*(waveform_f_Hz[1] - waveform_f_Hz[0])
		print(f"Setting peak-detect bandwidth to {bandwidth} Hz")
		
	# Get target indecies
	try:
		targ_range = np.logical_and(waveform_f_Hz>=(f_target-bandwidth/2), waveform_f_Hz<=(f_target+bandwidth/2))
		# targ_range = waveform_f_Hz>=(f_target-bandwidth/2)
		n_match = np.count_nonzero(targ_range)
		print(f"Found {n_match} points")
	except:
		print(waveform_f_Hz>=(f_target-bandwidth/2))
		print(waveform_f_Hz<=(f_target+bandwidth/2))
		print("Failed to find overlap points")
		return None
	
	if n_match == 0:
		return None
	
	#TODO: Include gaussian fit
	
	# Return Max Power
	return max(waveform_s_dBm[targ_range])

def bin_sweep(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, f_rf, f_lo, rbw_threshold_Hz=5e3, linestyle=':', marker='.', fig_no=1, peak_calc_bw=None):
	''' Takes a mixing sweep and returns the binned data, ie power at each harmonic level. '''
	
	# Handle list
	if type(waveform_f_Hz[0]) == list or type(waveform_f_Hz[0]) == np.ndarray:
		
		print("LIST MODE")
		
		# Run each one
		psl = []
		for f_Hz, s_dBm, rbw_Hz, f_rf_, f_lo_ in zip(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, f_rf, f_lo):
			psl.append(bin_sweep(f_Hz, s_dBm, rbw_Hz, f_rf_, f_lo_))
			
		return psl
	
	else:
		
		print("SINGLE MODE")
		print(type(waveform_f_Hz))
		print(type(waveform_f_Hz[0]))
		
		# Remove coarse sweep points
		idx_fine = waveform_rbw_Hz <= rbw_threshold_Hz
		waveform_f_Hz = waveform_f_Hz[idx_fine]
		waveform_s_dBm = waveform_s_dBm[idx_fine]
		
		ps = PowerSpectrum()
		
		# Calculate center frequencies
		ps.f_rf = f_rf
		ps.f_lo = f_lo
		
		ps.p_rf = power_at(waveform_f_Hz, waveform_s_dBm, f_rf, peak_calc_bw)
		ps.p_lo = power_at(waveform_f_Hz, waveform_s_dBm, f_lo, peak_calc_bw)
		
		ps.p_lo2 = power_at(waveform_f_Hz, waveform_s_dBm, f_lo*2, peak_calc_bw)
		ps.p_lo3 = power_at(waveform_f_Hz, waveform_s_dBm, f_lo*3, peak_calc_bw)
		
		ps.p_rf2 = power_at(waveform_f_Hz, waveform_s_dBm, f_rf*2, peak_calc_bw)
		ps.p_rf3 = power_at(waveform_f_Hz, waveform_s_dBm, f_rf*3, peak_calc_bw)
		
		ps.p_mx1L = power_at(waveform_f_Hz, waveform_s_dBm, f_rf-f_lo, peak_calc_bw)
		ps.p_mx1H = power_at(waveform_f_Hz, waveform_s_dBm, f_rf+f_lo, peak_calc_bw)
		ps.p_mx2L = power_at(waveform_f_Hz, waveform_s_dBm, f_rf-f_lo*2, peak_calc_bw)
		ps.p_mx2H = power_at(waveform_f_Hz, waveform_s_dBm, f_rf+f_lo*2, peak_calc_bw)
		
		return ps

spec_list = bin_sweep(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, f_rf*1e9, f_lo*1e9)

p_rf = [ps.p_rf for ps in spec_list]
p_lo = [ps.p_lo for ps in spec_list]

p_lo2 = [ps.p_lo2 for ps in spec_list]
p_lo3 = [ps.p_lo3 for ps in spec_list]

p_rf2 = [ps.p_rf2 for ps in spec_list]
p_rf3 = [ps.p_rf3 for ps in spec_list]

p_mx1L = [ps.p_mx1L for ps in spec_list]
p_mx1H = [ps.p_mx1H for ps in spec_list]
p_mx2L = [ps.p_mx2L for ps in spec_list]
p_mx2H = [ps.p_mx2H for ps in spec_list]

plt.figure(1)
plt.plot(p_rf, label='RF')
plt.plot(p_lo, label="LO")
plt.plot(p_lo2, label='LO2')
plt.plot(p_lo3, label="LO3")
plt.ylabel("Power (dBm)")
plt.xlabel("Sweep Index")
plt.grid()
plt.legend()

plt.figure(2)
# plt.plot(p_mx1H, label='MP-1 High')
# plt.plot(p_mx1L, label='MP-1 Low')
# plt.plot(p_mx2H, label='MP-2 High')
# plt.plot(p_mx2L, label='MP-2 Low')
plt.plot(p_mx1H, label='RF + LO')
plt.plot(p_mx1L, label='RF - LO')
plt.plot(p_mx2H, label='RF + 2LO')
plt.plot(p_mx2L, label='RF - 2LO')
plt.ylabel("Power (dBm)")
plt.xlabel("Sweep Index")
plt.grid()
plt.legend()

plt.figure(3)
plt.plot(f_rf, label='RF')
plt.plot(f_lo, label='LO')
plt.ylabel("Frequency (GHz)")
plt.xlabel("Sweep Index")
plt.grid()
plt.legend()

plt.show()


