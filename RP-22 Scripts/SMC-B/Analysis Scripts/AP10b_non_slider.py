'''
Reads harmonic generation data from MP3.
'''


import matplotlib.pyplot as plt
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
import os
from ganymede import *
from pylogfile.base import *
import sys
import numpy as np
import pickle
from matplotlib.widgets import Slider, Button

plt.rcParams['font.family'] = 'Aptos'

#------------------------------------------------------------
# Import Data

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C1*E', 'Track 2*'])

if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

# filename = "RP22B_MP3_t1_31July2024_R4C4T1_r1_autosave.hdf"
filename = "RP22B_MP5a_t3_19Sept2024_R4C1T2_r1.hdf"

analysis_file = os.path.join(datapath, filename)

log = LogPile()

##--------------------------------------------
# Read HDF5 File

print("Loading file contents into memory")
# log.info("Loading file contents into memory")

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	# Read primary dataset
	GROUP = 'dataset'
	freq_rf_GHz = fh[GROUP]['freq_rf_GHz'][()]
	power_rf_dBm = fh[GROUP]['power_rf_dBm'][()]
	
	waveform_f_Hz = fh[GROUP]['waveform_f_Hz'][()]
	waveform_s_dBm = fh[GROUP]['waveform_s_dBm'][()]
	waveform_rbw_Hz = fh[GROUP]['waveform_rbw_Hz'][()]
	
	MFLI_V_offset_V = fh[GROUP]['MFLI_V_offset_V'][()]
	requested_Idc_mA = fh[GROUP]['requested_Idc_mA'][()]
	raw_meas_Vdc_V = fh[GROUP]['raw_meas_Vdc_V'][()]
	Idc_mA = fh[GROUP]['Idc_mA'][()]
	detect_normal = fh[GROUP]['detect_normal'][()]
	
	temperature_K = fh[GROUP]['temperature_K'][()]
	
	temp_sp_K = fh[GROUP]['temp_setpoint_K'][()]

##--------------------------------------------
# Generate Mixing Products lists

rf1 = spectrum_peak_list(waveform_f_Hz, waveform_s_dBm, freq_rf_GHz*1e9)
rf2 = spectrum_peak_list(waveform_f_Hz, waveform_s_dBm, freq_rf_GHz*2e9)
rf3 = spectrum_peak_list(waveform_f_Hz, waveform_s_dBm, freq_rf_GHz*3e9)


##--------------------------------------------
# Plot Results
#
# At a given power level (lets start with -40 dBm), can
# I see any harmonic content, and how does it change with bias?
#
# Slider to change frequency and power


# Init figure
fig1, ax_list = plt.subplots(1, 3, figsize=(16, 4))
ax1 = ax_list[0]
ax2 = ax_list[1]
ax3 = ax_list[2]

freq_list = np.unique(freq_rf_GHz)
pwr_list = np.unique(power_rf_dBm)
temp_list = np.unique(temp_sp_K)
req_bias_list = np.unique(requested_Idc_mA)

SELECTED_FREQUENCY = freq_list[0]
SELECTED_POWER = pwr_list[0]

# for bv in req_bias_list:
for bv in [0.03]:
	
	if bv > 0.04:
		break
	
	# Filter relevant data
	mask_bias = (requested_Idc_mA == bv)
	mask_freq = (freq_rf_GHz == SELECTED_FREQUENCY)
	mask_pwr = (power_rf_dBm == SELECTED_POWER)
	mask_temps = (temp_sp_K < 8)
	mask = (mask_freq & mask_pwr & mask_bias & mask_temps)
	
	masked_temps = np.unique(temp_sp_K[mask])
	
	mean_temp = []
	std_temp = []
	mean_rf1 = []
	mean_rf2 = []
	mean_rf3 = []
	std_rf1 = []
	std_rf2 = []
	std_rf3 = []
	for mb in masked_temps:
		locmask = (temp_sp_K == mb) & mask
		
		mean_temp.append(np.mean(temperature_K[locmask]))
		std_temp.append(np.std(temperature_K[locmask]))
		
		mean_rf1.append(np.mean(rf1[locmask]))
		std_rf1.append(np.std(rf1[locmask]))
		
		mean_rf2.append(np.mean(rf2[locmask]))
		std_rf2.append(np.std(rf2[locmask]))
		
		mean_rf3.append(np.mean(rf3[locmask]))
		std_rf3.append(np.std(rf3[locmask]))

	mean_rf1 = np.array(mean_rf1)
	mean_rf2 = np.array(mean_rf2)
	mean_rf3 = np.array(mean_rf3)

	std_rf1 = np.array(std_rf1)
	std_rf2 = np.array(std_rf2)
	std_rf3 = np.array(std_rf3)

	ax1.errorbar(mean_temp, mean_rf1, xerr=std_temp, yerr=std_rf1, linestyle=':', marker='o', markersize=1.5, color=(0, 0.7, 0))
	ax2.errorbar(mean_temp, mean_rf2, xerr=std_temp, yerr=std_rf2, linestyle=':', marker='o', markersize=1.5, color=(0, 0, 0.7))
	ax3.errorbar(mean_temp, mean_rf3, xerr=std_temp, yerr=std_rf3, linestyle=':', marker='o', markersize=1.5, color=(0.7, 0, 0))

	ax1.fill_between(mean_temp, mean_rf1 - std_rf1, mean_rf1 + std_rf1, color=(0, 0.7, 0), alpha=0.25)
	ax2.fill_between(mean_temp, mean_rf2 - std_rf2, mean_rf2 + std_rf2, color=(0, 0, 0.7), alpha=0.25)
	ax3.fill_between(mean_temp, mean_rf3 - std_rf3, mean_rf3 + std_rf3, color=(0.7, 0, 0), alpha=0.25)

	ax1.set_xlabel("Temperature (K)")
	ax1.set_ylabel("Power (dBm)")
	ax1.set_title("Fundamental Tone")
	ax1.grid(True)
	
	ax2.set_xlabel("Temperature (K)")
	ax2.set_ylabel("Power (dBm)")
	ax2.set_title("2nd Harmonic Tone")
	ax2.grid(True)
	
	ax3.set_xlabel("Temperature (K)")
	ax3.set_ylabel("Power (dBm)")
	ax3.set_title("3rd Harmonic Tone")
	ax3.grid(True)
	
fig1.tight_layout()


##------------------------------------------ Make normalized figure

# Init figure
fig2, ax_list = plt.subplots(1, 3, figsize=(16, 4))
ax1_2 = ax_list[0]
ax2_2 = ax_list[1]
ax3_2 = ax_list[2]

freq_list = np.unique(freq_rf_GHz)
pwr_list = np.unique(power_rf_dBm)
temp_list = np.unique(temp_sp_K)
req_bias_list = np.unique(requested_Idc_mA)

SELECTED_FREQUENCY = freq_list[0]
SELECTED_POWER = pwr_list[0]

for bv in req_bias_list:
# for bv in [0.03]:
	
	if bv > 0.04:
		break
	
	# Filter relevant data
	mask_bias = (requested_Idc_mA == bv)
	mask_freq = (freq_rf_GHz == SELECTED_FREQUENCY)
	mask_pwr = (power_rf_dBm == SELECTED_POWER)
	mask_temps = (temp_sp_K < 8)
	mask = (mask_freq & mask_pwr & mask_bias & mask_temps)
	
	masked_temps = np.unique(temp_sp_K[mask])
	
	mean_temp = []
	std_temp = []
	mean_rf1 = []
	mean_rf2 = []
	mean_rf3 = []
	std_rf1 = []
	std_rf2 = []
	std_rf3 = []
	for mb in masked_temps:
		locmask = (temp_sp_K == mb) & mask
		
		mean_temp.append(np.mean(temperature_K[locmask]))
		std_temp.append(np.std(temperature_K[locmask]))
		
		mean_rf1.append(np.mean(rf1[locmask]))
		std_rf1.append(np.std(rf1[locmask]))
		
		mean_rf2.append(np.mean(rf2[locmask]))
		std_rf2.append(np.std(rf2[locmask]))
		
		mean_rf3.append(np.mean(rf3[locmask]))
		std_rf3.append(np.std(rf3[locmask]))

	mean_rf1 = np.array(mean_rf1)
	mean_rf2 = np.array(mean_rf2)
	mean_rf3 = np.array(mean_rf3)

	std_rf1 = np.array(std_rf1)
	std_rf2 = np.array(std_rf2)
	std_rf3 = np.array(std_rf3)

	ax1_2.errorbar(mean_temp, mean_rf1-mean_rf1[0], xerr=std_temp, yerr=std_rf1, linestyle=':', marker='o', markersize=1.5, color=(0, 0.7, 0), elinewidth=0.2)
	ax2_2.errorbar(mean_temp, mean_rf2-mean_rf2[0], xerr=std_temp, yerr=std_rf2, linestyle=':', marker='o', markersize=1.5, color=(0, 0, 0.7), elinewidth=0.2)
	ax3_2.errorbar(mean_temp, mean_rf3-mean_rf3[0], xerr=std_temp, yerr=std_rf3, linestyle=':', marker='o', markersize=1.5, color=(0.7, 0, 0), elinewidth=0.2)
	
	ALPHA = 0.1
	
	ax1_2.fill_between(mean_temp, mean_rf1-mean_rf1[0] - std_rf1, mean_rf1-mean_rf1[0] + std_rf1, color=(0, 0.7, 0), alpha=ALPHA)
	ax2_2.fill_between(mean_temp, mean_rf2-mean_rf2[0] - std_rf2, mean_rf2-mean_rf2[0] + std_rf2, color=(0, 0, 0.7), alpha=ALPHA)
	ax3_2.fill_between(mean_temp, mean_rf3-mean_rf3[0] - std_rf3, mean_rf3-mean_rf3[0] + std_rf3, color=(0.7, 0, 0), alpha=ALPHA)

	ax1_2.set_xlabel("Temperature (K)")
	ax1_2.set_ylabel("Power (Normalized to 7 K) (dB)")
	ax1_2.set_title("Fundamental Tone")
	ax1_2.grid(True)
	
	ax2_2.set_xlabel("Temperature (K)")
	ax2_2.set_ylabel("Power (Normalized to 7 K) (dB)")
	ax2_2.set_title("2nd Harmonic Tone")
	ax2_2.grid(True)
	
	ax3_2.set_xlabel("Temperature (K)")
	ax3_2.set_ylabel("Power (Normalized to 7 K) (dB)")
	ax3_2.set_title("3rd Harmonic Tone")
	ax3_2.grid(True)
	
fig2.tight_layout()

plt.show()




