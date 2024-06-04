import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from dataclasses import dataclass
from rp22_helper import *
from colorama import Fore, Style
import sys

datapath = get_datadir_path(rp=22, smc='A')
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
filename = "MS3_02June2024_r1.hdf"

analysis_file = os.path.join(datapath, filename)

USE_CONST_SCALE = True

@dataclass
class PowerSpectrum:
	
	p_fund:float = None
	p_2h:float = None
	p_3h:float = None
	p_pm:float = None

##--------------------------------------------
# Read HDF5 File

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	source_script = fh['conditions']['source_script'][()]
	conf_json = fh['conditions']['conf_json'][()]
	
	coupled_power_dBm = fh['dataset']['coupled_power_dBm'][()]
	rf_enabled = fh['dataset']['rf_enabled'][()]
	lo_enabled = fh['dataset']['lo_enabled'][()]
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

def dBm2W(x:float):
	mw = 10**(x/10)
	w = mw/1000
	return w

def W2dBm(x:float):
	mw = x*1000
	dBm = 10*np.log10(mw)
	return dBm

def power_total(waveform_f_Hz, waveform_s_dBm):
	''' Returns the total power in a spectrum. '''
	
	all_bws = []
	
	pwr = 0
	
	for idx, s_pt in enumerate(waveform_s_dBm):
		
		bw_list = []
		try:
			bw_new = waveform_f_Hz[idx]-waveform_f_Hz[idx-1]
			if bw_new == 0:
				bw_new = waveform_f_Hz[idx-1]-waveform_f_Hz[idx-2]
			bw_list.append(bw_new)
		except:
			pass
		try:
			bw_new = waveform_f_Hz[idx+1]-waveform_f_Hz[idx]
			if bw_new == 0:
				bw_new = waveform_f_Hz[idx+2]-waveform_f_Hz[idx+1]
			bw_list.append(bw_new)
		except:
			pass
		bw = min(bw_list)
		
		if bw > 10e3:
			continue
		
		all_bws.append(bw)
		
		pwr += dBm2W(s_pt)*bw
	
	return W2dBm(pwr)

def lin2dB(x:float, use_10:bool=False):
	if use_10:
		return 10*np.log10(x)
	else:
		return 20*np.log10(x)

def cal_bin_sweep(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, f_rf, f_lo, p_rf, p_lo, rf_enable, lo_enable, pwr_meter, rbw_threshold_Hz=5e3, linestyle=':', marker='.', fig_no=1, peak_calc_bw=None):
	''' Looks at power loss via the sweep data '''
	
	# Handle list
	if type(waveform_f_Hz[0]) == list or type(waveform_f_Hz[0]) == np.ndarray:
		
		# Run each one
		psl = []
		for f_Hz, s_dBm, rbw_Hz, f_rf_, f_lo_, p_rf_, p_lo_, rf_en_, lo_en_, pwr_meter_ in zip(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, f_rf, f_lo, p_rf, p_lo, rf_enable, lo_enable, pwr_meter):
			new_point = cal_bin_sweep(f_Hz, s_dBm, rbw_Hz, f_rf_, f_lo_, p_rf_, p_lo_, rf_en_, lo_en_, pwr_meter_)
			if new_point is not None:
				psl.append(new_point)
			
		return psl
	
	else:
		
		# Remove coarse sweep points
		idx_fine = waveform_rbw_Hz <= rbw_threshold_Hz
		waveform_f_Hz = waveform_f_Hz[idx_fine]
		waveform_s_dBm = waveform_s_dBm[idx_fine]
		
		ps = PowerSpectrum()
		
		# Look at relevant signal generator
		if rf_enable:
			p_sg = p_rf
			f_sg = f_rf
			is_rf = True
		else:
			p_sg = p_lo
			f_sg = f_lo
			is_rf = False
		
		# Calculate power at each harmonic
		# ps.p_fund = power_at(waveform_f_Hz, waveform_s_dBm, fcl_sg, peak_calc_bw)
		ps.p_fund = power_total(waveform_f_Hz, waveform_s_dBm)
		ps.p_2h = power_at(waveform_f_Hz, waveform_s_dBm, f_sg*2, peak_calc_bw)
		ps.p_3h = power_at(waveform_f_Hz, waveform_s_dBm, f_sg*3, peak_calc_bw)
		ps.p_pm = pwr_meter
		ps.p_sg = p_sg
		ps.freq = f_sg
		ps.is_rf = is_rf
		
		if any(np.array([ps.p_fund, ps.p_2h, ps.p_3h, ps.p_pm, ps.p_sg, ps.freq])==None):
			print(f"Skipping point: f={f_sg/1e9} GHz, P={p_sg} dBm, RF={is_rf}")
			return None
		
		return ps

spec_list = cal_bin_sweep(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, f_rf*1e9, f_lo*1e9, p_rf, p_lo, rf_enabled, lo_enabled, coupled_power_dBm)

Pfund = np.array([ps.p_fund for ps in spec_list])
P2h = np.array([ps.p_2h for ps in spec_list])
P3h = np.array([ps.p_3h for ps in spec_list])
Ppm = np.array([ps.p_pm for ps in spec_list])
P0 = np.array([ps.p_sg for ps in spec_list])
freqs = np.array([ps.freq for ps in spec_list])
is_rf = np.array([ps.is_rf for ps in spec_list])
is_lo = np.logical_not(is_rf)

# Get unique values
unique_f = sorted(list(set(freqs)))
unique_p0 = sorted(list(set(P0)))


plt.figure(1)
plt.subplot(1, 2, 1)
plt.scatter(freqs/1e9, P0-Pfund, s=10, marker='.')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Loss (dB)")
plt.title("Loss to S.A. Fundamental")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.scatter(freqs/1e9, P0, s=10, marker='.', label='P_sig_gen')
plt.scatter(freqs/1e9, Pfund, s=10, marker='.', label='P_meas_fund')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Spectral Power (dBm/Hz)")
plt.title("Power Levels")
plt.grid(True)
plt.legend()

c_rf = (0, 0, .7)
c_lo = (0.7, 0.3, 0)

plt.figure(2)
plt.subplot(1, 2, 1)
plt.scatter(freqs[is_rf]/1e9, P0[is_rf]-Ppm[is_rf]-10-3, s=10, marker='.', label='RF', color=(0, 0, 0.7))
plt.scatter(freqs[is_lo]/1e9, P0[is_lo]-Ppm[is_lo]-10-3, s=10, marker='.', label='LO', color=(.7, 0.3, 0))
plt.xlabel("Frequency (GHz)")
plt.ylabel("Loss (dB)")
plt.title("Loss to Power Meter")
plt.ylim([0, 10])
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(freqs[is_rf]/1e9, P0[is_rf], s=10, marker='.', label='P_sg RF', color=(0, 0, .7))
plt.scatter(freqs[is_lo]/1e9, P0[is_lo], s=10, marker='.', label='P_sg LO', color=(0.7, .3, 0))
plt.scatter(freqs[is_rf]/1e9, Ppm[is_rf], s=10, marker='x', label='P_meter RF', color=c_rf)
plt.scatter(freqs[is_lo]/1e9, Ppm[is_lo], s=10, marker='x', label='P_meter LO', color=c_lo)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (dBm)")
plt.title("Power Levels")
plt.grid(True)
plt.legend()

plt.tight_layout()




plt.show()


















# # Initialize array
# X, Y = np.meshgrid(unique_f_lo, unique_f_rf)

# Z_rf0 = np.zeros([len(unique_f_rf), len(unique_f_lo)])
# Z_lo0 = np.zeros([len(unique_f_rf), len(unique_f_lo)])

# Z_lo2 = np.zeros([len(unique_f_rf), len(unique_f_lo)])
# Z_lo3 = np.zeros([len(unique_f_rf), len(unique_f_lo)])
# Z_rf2 = np.zeros([len(unique_f_rf), len(unique_f_lo)])
# Z_rf3 = np.zeros([len(unique_f_rf), len(unique_f_lo)])

# Z_mx1H = np.zeros([len(unique_f_rf), len(unique_f_lo)])
# Z_mx1L = np.zeros([len(unique_f_rf), len(unique_f_lo)])
# Z_mx2H = np.zeros([len(unique_f_rf), len(unique_f_lo)])
# Z_mx2L = np.zeros([len(unique_f_rf), len(unique_f_lo)])

# for irf, ufrf in enumerate(unique_f_rf):
	
# 	# Get matching indecies
# 	idxs_rf = (ff_rf == ufrf)
	
# 	for ilo, uflo in enumerate(unique_f_lo):
		
# 		# Get matching indecies
# 		idxs_lo = (ff_lo == uflo)
		
# 		# Check correct number found
# 		num_lo = np.count_nonzero(idxs_lo)
# 		num_rf = np.count_nonzero(idxs_rf)
# 		idx_master = np.bitwise_and(idxs_rf, idxs_lo)
# 		num_master = np.count_nonzero(idx_master)
# 		if (num_master != 1):
# 			print(f"Error, found wrong number of points ({num_master}| {num_rf}, {num_lo}).")
# 			break
		
# 		idx_master = np.bitwise_and(idxs_rf, idxs_lo)
		
# 		Z_rf0[irf][ilo] = p_rf[idx_master][0]
# 		Z_lo0[irf][ilo] = p_lo[idx_master][0]
		
# 		Z_lo2[irf][ilo] = p_lo2[idx_master][0]
# 		Z_lo3[irf][ilo] = p_lo3[idx_master][0]
# 		Z_rf2[irf][ilo] = p_rf2[idx_master][0]
# 		Z_rf3[irf][ilo] = p_rf3[idx_master][0]
		
# 		Z_mx1L[irf][ilo] = p_mx1L[idx_master][0]
# 		Z_mx1H[irf][ilo] = p_mx1H[idx_master][0]
# 		Z_mx2L[irf][ilo] = p_mx2L[idx_master][0]
# 		Z_mx2H[irf][ilo] = p_mx2H[idx_master][0]
		

# c_min = None
# c_max = None
# n_levels = 400
# if USE_CONST_SCALE:
# 	c_min = np.nanmin([np.nanmin(Z_lo2), np.nanmin(Z_lo3), np.nanmin(Z_rf2), np.nanmin(Z_rf3)])
# 	c_max = np.nanmax([np.nanmax(Z_lo2), np.nanmax(Z_lo3), np.nanmax(Z_rf2), np.nanmax(Z_rf3)])
	
# 	print(f"c_min={c_min}, c_max={c_max}")

# plt.figure(1)
# plt.subplot(2, 2, 1)
# plt.contourf(X/1e9, Y/1e9, Z_lo2, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("2*f_LO")
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.contourf(X/1e9, Y/1e9, Z_lo3, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("3*f_LO")
# plt.colorbar()

# plt.subplot(2, 2, 2)
# plt.contourf(X/1e9, Y/1e9, Z_rf2, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("2*f_RF")
# plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.contourf(X/1e9, Y/1e9, Z_rf3, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("3*f_RF")
# plt.colorbar()

# if USE_CONST_SCALE:
# 	c_min = np.nanmin([np.nanmin(Z_mx1H), np.nanmin(Z_mx1L), np.nanmin(Z_mx2L), np.nanmin(Z_mx2H)])
# 	c_max = np.nanmax([np.nanmax(Z_mx1H), np.nanmax(Z_mx1L), np.nanmax(Z_mx2L), np.nanmax(Z_mx2H)])
	
# 	print(f"c_min={c_min}, c_max={c_max}")
	
# plt.figure(2)
# plt.subplot(2, 2, 1)
# plt.contourf(X/1e9, Y/1e9, Z_mx1L, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("RF - LO")
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.contourf(X/1e9, Y/1e9, Z_mx1H, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("RF + LO")
# plt.colorbar()

# plt.subplot(2, 2, 2)
# plt.contourf(X/1e9, Y/1e9, Z_mx2L, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("RF - 2LO")
# plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.contourf(X/1e9, Y/1e9, Z_mx2H, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("RF + 2LO")
# plt.colorbar()

# plt.figure(3)
# plt.subplot(2, 2, 1)
# plt.contourf(X/1e9, Y/1e9, Z_mx1L-Z_rf0, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("P(RF - LO)-P(RF)")
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.contourf(X/1e9, Y/1e9, Z_mx1H-Z_rf0, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("P(RF + LO)-P(RF)")
# plt.colorbar()

# plt.subplot(2, 2, 2)
# plt.contourf(X/1e9, Y/1e9, Z_mx2L-Z_rf0, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("P(RF - 2LO)-P(RF)")
# plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.contourf(X/1e9, Y/1e9, Z_mx2H-Z_rf0, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("P(RF + 2LO)-P(RF)")
# plt.colorbar()

# plt.figure(4)
# plt.subplot(2, 2, 1)
# plt.contourf(X/1e9, Y/1e9, Z_mx1L-Z_lo0, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("P(RF - LO)-P(LO)")
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.contourf(X/1e9, Y/1e9, Z_mx1H-Z_lo0, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("P(RF + LO)-P(LO)")
# plt.colorbar()

# plt.subplot(2, 2, 2)
# plt.contourf(X/1e9, Y/1e9, Z_mx2L-Z_lo0, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("P(RF - 2LO)-P(LO)")
# plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.contourf(X/1e9, Y/1e9, Z_mx2H-Z_lo0, levels=np.linspace(c_min,c_max,n_levels))
# plt.xlabel("LO Frequency (GHz)")
# plt.ylabel("RF Frequency (GHz)")
# plt.title("P(RF + 2LO)-P(LO)")
# plt.colorbar()

# plt.show()