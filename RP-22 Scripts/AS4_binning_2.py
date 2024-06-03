import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from dataclasses import dataclass
from rp22_helper import *

datapath = get_datadir_path(22, 'A')
filename = "dMS1_28May2024_DC1V0_r1.hdf"

analysis_file = os.path.join(datapath, filename)

USE_CONST_SCALE = True

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
	p_rf = fh['dataset']['power_RF_dBm'][()]
	p_lo = fh['dataset']['power_LO_dBm'][()]
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
		# print(f"Setting peak-detect bandwidth to {bandwidth} Hz")
		
	# Get target indecies
	try:
		targ_range = np.logical_and(waveform_f_Hz>=(f_target-bandwidth/2), waveform_f_Hz<=(f_target+bandwidth/2))
		# targ_range = waveform_f_Hz>=(f_target-bandwidth/2)
		n_match = np.count_nonzero(targ_range)
		# print(f"Found {n_match} points")
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
		
		# print("LIST MODE")
		
		# Run each one
		psl = []
		for f_Hz, s_dBm, rbw_Hz, f_rf_, f_lo_ in zip(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, f_rf, f_lo):
			psl.append(bin_sweep(f_Hz, s_dBm, rbw_Hz, f_rf_, f_lo_))
			
		return psl
	
	else:
		
		# print("SINGLE MODE")
		# print(type(waveform_f_Hz))
		# print(type(waveform_f_Hz[0]))
		
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

ff_rf = np.array([ps.f_rf for ps in spec_list])
ff_lo = np.array([ps.f_lo for ps in spec_list])

p_rf = np.array([ps.p_rf for ps in spec_list])
p_lo = np.array([ps.p_lo for ps in spec_list])

p_lo2 = np.array([ps.p_lo2 for ps in spec_list])
p_lo3 = np.array([ps.p_lo3 for ps in spec_list])

p_rf2 = np.array([ps.p_rf2 for ps in spec_list])
p_rf3 = np.array([ps.p_rf3 for ps in spec_list])

p_mx1L = np.array([ps.p_mx1L for ps in spec_list])
p_mx1H = np.array([ps.p_mx1H for ps in spec_list])
p_mx2L = np.array([ps.p_mx2L for ps in spec_list])
p_mx2H = np.array([ps.p_mx2H for ps in spec_list])

# Get unique values
unique_f_rf = sorted(list(set(ff_rf)))
unique_f_lo = sorted(list(set(ff_lo)))

# Initialize array
X, Y = np.meshgrid(unique_f_lo, unique_f_rf)

Z_rf0 = np.zeros([len(unique_f_rf), len(unique_f_lo)])
Z_lo0 = np.zeros([len(unique_f_rf), len(unique_f_lo)])

Z_lo2 = np.zeros([len(unique_f_rf), len(unique_f_lo)])
Z_lo3 = np.zeros([len(unique_f_rf), len(unique_f_lo)])
Z_rf2 = np.zeros([len(unique_f_rf), len(unique_f_lo)])
Z_rf3 = np.zeros([len(unique_f_rf), len(unique_f_lo)])

Z_mx1H = np.zeros([len(unique_f_rf), len(unique_f_lo)])
Z_mx1L = np.zeros([len(unique_f_rf), len(unique_f_lo)])
Z_mx2H = np.zeros([len(unique_f_rf), len(unique_f_lo)])
Z_mx2L = np.zeros([len(unique_f_rf), len(unique_f_lo)])

for irf, ufrf in enumerate(unique_f_rf):
	
	# Get matching indecies
	idxs_rf = (ff_rf == ufrf)
	
	for ilo, uflo in enumerate(unique_f_lo):
		
		# Get matching indecies
		idxs_lo = (ff_lo == uflo)
		
		# Check correct number found
		num_lo = np.count_nonzero(idxs_lo)
		num_rf = np.count_nonzero(idxs_rf)
		idx_master = np.bitwise_and(idxs_rf, idxs_lo)
		num_master = np.count_nonzero(idx_master)
		if (num_master != 1):
			print(f"Error, found wrong number of points ({num_master}| {num_rf}, {num_lo}).")
			break
		
		idx_master = np.bitwise_and(idxs_rf, idxs_lo)
		
		Z_rf0[irf][ilo] = p_rf[idx_master][0]
		Z_lo0[irf][ilo] = p_lo[idx_master][0]
		
		Z_lo2[irf][ilo] = p_lo2[idx_master][0]
		Z_lo3[irf][ilo] = p_lo3[idx_master][0]
		Z_rf2[irf][ilo] = p_rf2[idx_master][0]
		Z_rf3[irf][ilo] = p_rf3[idx_master][0]
		
		Z_mx1L[irf][ilo] = p_mx1L[idx_master][0]
		Z_mx1H[irf][ilo] = p_mx1H[idx_master][0]
		Z_mx2L[irf][ilo] = p_mx2L[idx_master][0]
		Z_mx2H[irf][ilo] = p_mx2H[idx_master][0]
		

c_min = None
c_max = None
n_levels = 400
if USE_CONST_SCALE:
	c_min = np.nanmin([np.nanmin(Z_lo2), np.nanmin(Z_lo3), np.nanmin(Z_rf2), np.nanmin(Z_rf3)])
	c_max = np.nanmax([np.nanmax(Z_lo2), np.nanmax(Z_lo3), np.nanmax(Z_rf2), np.nanmax(Z_rf3)])
	
	print(f"c_min={c_min}, c_max={c_max}")

plt.figure(1)
plt.subplot(2, 2, 1)
plt.contourf(X/1e9, Y/1e9, Z_lo2, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("2*f_LO")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.contourf(X/1e9, Y/1e9, Z_lo3, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("3*f_LO")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.contourf(X/1e9, Y/1e9, Z_rf2, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("2*f_RF")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.contourf(X/1e9, Y/1e9, Z_rf3, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("3*f_RF")
plt.colorbar()

if USE_CONST_SCALE:
	c_min = np.nanmin([np.nanmin(Z_mx1H), np.nanmin(Z_mx1L), np.nanmin(Z_mx2L), np.nanmin(Z_mx2H)])
	c_max = np.nanmax([np.nanmax(Z_mx1H), np.nanmax(Z_mx1L), np.nanmax(Z_mx2L), np.nanmax(Z_mx2H)])
	
	print(f"c_min={c_min}, c_max={c_max}")
	
plt.figure(2)
plt.subplot(2, 2, 1)
plt.contourf(X/1e9, Y/1e9, Z_mx1L, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("RF - LO")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.contourf(X/1e9, Y/1e9, Z_mx1H, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("RF + LO")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.contourf(X/1e9, Y/1e9, Z_mx2L, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("RF - 2LO")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.contourf(X/1e9, Y/1e9, Z_mx2H, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("RF + 2LO")
plt.colorbar()

plt.figure(3)
plt.subplot(2, 2, 1)
plt.contourf(X/1e9, Y/1e9, Z_mx1L-Z_rf0, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("P(RF - LO)-P(RF)")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.contourf(X/1e9, Y/1e9, Z_mx1H-Z_rf0, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("P(RF + LO)-P(RF)")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.contourf(X/1e9, Y/1e9, Z_mx2L-Z_rf0, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("P(RF - 2LO)-P(RF)")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.contourf(X/1e9, Y/1e9, Z_mx2H-Z_rf0, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("P(RF + 2LO)-P(RF)")
plt.colorbar()

plt.figure(4)
plt.subplot(2, 2, 1)
plt.contourf(X/1e9, Y/1e9, Z_mx1L-Z_lo0, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("P(RF - LO)-P(LO)")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.contourf(X/1e9, Y/1e9, Z_mx1H-Z_lo0, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("P(RF + LO)-P(LO)")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.contourf(X/1e9, Y/1e9, Z_mx2L-Z_lo0, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("P(RF - 2LO)-P(LO)")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.contourf(X/1e9, Y/1e9, Z_mx2H-Z_lo0, levels=np.linspace(c_min,c_max,n_levels))
plt.xlabel("LO Frequency (GHz)")
plt.ylabel("RF Frequency (GHz)")
plt.title("P(RF + 2LO)-P(LO)")
plt.colorbar()

plt.show()