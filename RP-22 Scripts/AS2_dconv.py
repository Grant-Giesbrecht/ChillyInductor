import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from colorama import Fore, Style

##--------------------------------------------
# Read HDF5 File

analysis_file = "E:\ARC0 PhD Data\RP-22 Lk Dil Fridge 2024\Data\SMC-A Downconversion v1\dMS1_28May2024_DC1V0_r1.hdf"

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

def plot_sweep(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, rbw_threshold_Hz=5e3, linestyle=':', marker='.', fig_no=1, f_rf=None, f_lo=None):
	
	# Calcuate indecies
	fine_idx = waveform_rbw_Hz<=rbw_threshold_Hz
	coarse_idx = waveform_rbw_Hz>rbw_threshold_Hz
	
	xlow = min(waveform_f_Hz/1e9)
	xhigh = max(waveform_f_Hz/1e9)
	
	plt.figure(fig_no)
	plt.subplot(2, 1, 1)
	plt.scatter(waveform_f_Hz[fine_idx]/1e9, waveform_s_dBm[fine_idx], marker='.')
	plt.grid(True)
	plt.xlabel("Frequency (GHz)")
	plt.ylabel("Power (dBm)")
	plt.title("Fine Sweep")
	plt.xlim(xlow, xhigh)
	
	plt.subplot(2, 1, 2)
	plt.scatter(waveform_f_Hz[coarse_idx]/1e9, waveform_s_dBm[coarse_idx], marker='.')
	plt.grid(True)
	plt.xlabel("Frequency (GHz)")
	plt.ylabel("Power (dBm)")
	plt.title("Coarse Sweep")
	plt.xlim(xlow, xhigh)
	
	box_width = (xhigh-xlow)/100
	rf_c = (0, 0.7, 0)
	mix_c = (0.7, 0, 0.7)
	lo_c = (0.7, 0, 0)
	alp = [0.25, 0.15, 0.075]
	
	
	# Draw harmonics
	if (f_rf is not None) and (f_lo is not None):
		
		def add_box(box_width, center, alp, color):
			ybounds = plt.ylim()
			
			r = Rectangle((center-box_width/2, ybounds[0]), box_width, ybounds[1]-ybounds[0], facecolor=color, alpha=alp)
			
			ax = plt.gca()
			ax.add_patch(r)
		
		for pltN in [1, 2]:
			plt.subplot(2, 1, pltN)
			add_box(box_width, f_rf, alp[0], rf_c)
			add_box(box_width, 2*f_rf, alp[1], rf_c)
			add_box(box_width, f_rf-f_lo, alp[1], mix_c)
			add_box(box_width, f_rf+f_lo, alp[1], mix_c)
			add_box(box_width, f_rf-2*f_lo, alp[2], mix_c)
			add_box(box_width, f_rf+2*f_lo, alp[2], mix_c)
			add_box(box_width, f_lo, alp[0], lo_c)
			add_box(box_width, 2*f_lo, alp[1], lo_c)
			add_box(box_width, 3*f_lo, alp[2], lo_c)

##-------------------------------------------------------------------------
# To plot a range of sweeps

for idx in range(50, 70):
	
	print(f"{Fore.CYAN}Displaying info for sweep:{Style.RESET_ALL}")
	print(f"\tf_rf = {Fore.YELLOW}{f_rf[idx]} GHz{Style.RESET_ALL}")
	print(f"\tf_lo = {Fore.YELLOW}{f_lo[idx]} GHz{Style.RESET_ALL}")
	print(f"\tP_rf = {Fore.YELLOW}{p_rf[idx]} dBm{Style.RESET_ALL}")
	print(f"\tP_lo = {Fore.YELLOW}{p_lo[idx]} dBm{Style.RESET_ALL}")
	
	plot_sweep(waveform_f_Hz[idx], waveform_s_dBm[idx], waveform_rbw_Hz[idx], fig_no=1, f_rf=f_rf[idx], f_lo=f_lo[idx])
	plt.show()


##-------------------------------------------------------------------------
# To plot specific conditions

f_rf_target = 6
f_lo_target = 1.5

# f_rf