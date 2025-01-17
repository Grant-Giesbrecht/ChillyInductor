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
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))
fig1.subplots_adjust(left=0.065, bottom=0.065, top=0.95, right=0.7)

freq_list = np.unique(freq_rf_GHz)
pwr_list = np.unique(power_rf_dBm)
temp_list = np.unique(temp_sp_K)
req_bias_list = np.unique(requested_Idc_mA)

conditions = {'sel_freq_GHz': freq_list[len(freq_list)//2], 'sel_power_dBm': pwr_list[len(pwr_list)//2], 'temp_K':temp_list[0]}

def plot_selcted_data():
	
	f = conditions['sel_freq_GHz']
	p = conditions['sel_power_dBm']
	t = conditions['temp_K']
	
	# Filter relevant data
	mask_freq = (freq_rf_GHz == conditions['sel_freq_GHz'])
	mask_pwr = (power_rf_dBm == conditions['sel_power_dBm'])
	mask_temp = (temp_sp_K == conditions['temp_K'])
	mask = (mask_freq & mask_pwr & mask_temp)
	
	# Plot results
	ax1.cla()
	
	# Average out repeated sweeps
	masked_biases = np.unique(requested_Idc_mA[mask])
	mean_bias = []
	std_bias = []
	mean_rf1 = []
	mean_rf2 = []
	mean_rf3 = []
	std_rf1 = []
	std_rf2 = []
	std_rf3 = []
	for mb in masked_biases:
		
		locmask = (requested_Idc_mA == mb) & mask
		
		mean_bias.append(np.mean(Idc_mA[locmask]))
		std_bias.append(np.std(Idc_mA[locmask]))
		
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
	
	ax1.errorbar(mean_bias, mean_rf1, xerr=std_bias, yerr=std_rf1, linestyle=':', marker='o', markersize=1.5, color=(0, 0.7, 0))
	ax1.errorbar(mean_bias, mean_rf2, xerr=std_bias, yerr=std_rf2, linestyle=':', marker='o', markersize=1.5, color=(0, 0, 0.7))
	ax1.errorbar(mean_bias, mean_rf3, xerr=std_bias, yerr=std_rf3, linestyle=':', marker='o', markersize=1.5, color=(0.7, 0, 0))
	
	ax1.fill_between(mean_bias, mean_rf1 - std_rf1, mean_rf1 + std_rf1, color=(0, 0.7, 0), alpha=0.25)
	ax1.fill_between(mean_bias, mean_rf2 - std_rf2, mean_rf2 + std_rf2, color=(0, 0, 0.7), alpha=0.25)
	ax1.fill_between(mean_bias, mean_rf3 - std_rf3, mean_rf3 + std_rf3, color=(0.7, 0, 0), alpha=0.25)
	
	
	ax1.set_title(f"f = {f} GHz, p = {p} dBm")
	ax1.set_xlabel("DC Bias (mA)")
	ax1.set_ylabel("Power (dBm)")
	ax1.legend(["Fundamental", "2nd Harm.", "3rd Harm."])
	ax1.grid(True)
	ax1.set_ylim([-100, -20])
	ax1.set_xlim([0, 0.15])
	
	fig1.canvas.draw_idle()
	
	

def update_pwr(x):
	conditions['sel_power_dBm'] = x
	plot_selcted_data()
	
def update_freq(x):
	conditions['sel_freq_GHz'] = x
	plot_selcted_data()
	
def update_temp(x):
	conditions['temp_K'] = x
	plot_selcted_data()

# Down-sample freq list
if len(freq_list) > 15:
	freq_list_downsampled = downsample_labels(freq_list, 11)
else:
	freq_list_downsampled = freq_list

# Down-sample power list
if len(pwr_list) > 15:
	pwr_list_downsampled = downsample_labels(pwr_list, 11)
else:
	pwr_list_downsampled = pwr_list

# Down-sample temp list
if len(temp_list) > 15:
	temp_list_downsampled = downsample_labels(temp_list, 11)
else:
	temp_list_downsampled = temp_list

# Frequency Slider
ax_freq = fig1.add_axes([0.75, 0.1, 0.03, 0.8])
slider_freq = Slider(ax_freq, 'Freq (GHz)', np.min(freq_list), np.max(freq_list), initcolor='none', valstep=freq_list, color='green', orientation='vertical', valinit=conditions['sel_freq_GHz'])
slider_freq.on_changed(update_freq)
ax_freq.add_artist(ax_freq.yaxis)
ax_freq.set_yticks(freq_list, labels=freq_list_downsampled)

# Power Slider
ax_pwr = fig1.add_axes([0.84, 0.1, 0.03, 0.8])
slider_pwr = Slider(ax_pwr, 'Power (dBm)', np.min(pwr_list), np.max(pwr_list), initcolor='none', valstep=pwr_list, color='red', orientation='vertical', valinit=conditions['sel_power_dBm'])
slider_pwr.on_changed(update_pwr)
ax_pwr.add_artist(ax_pwr.yaxis)
ax_pwr.set_yticks(pwr_list, labels=pwr_list_downsampled)

# Temp Slider
ax_temp = fig1.add_axes([0.93, 0.1, 0.03, 0.8])
slider_temp = Slider(ax_temp, 'Temp (K)', np.min(temp_list), np.max(temp_list), initcolor='none', valstep=temp_list, color='blue', orientation='vertical', valinit=conditions['temp_K'])
slider_temp.on_changed(update_temp)
ax_temp.add_artist(ax_temp.yaxis)
ax_temp.set_yticks(temp_list, labels=temp_list_downsampled)


plot_selcted_data()

# END_IDX = -1
# ax1.plot(timestamps[:END_IDX], temperatures[:END_IDX], linestyle=':', label="Temp (K)", color=(0, 0, 0.7), marker='.')
# ax2.plot(timestamps[:END_IDX], resistances[:END_IDX]-sc_res, linestyle='--', label="Resistance (Ohms)", color=(0.7, 0, 0), marker='.')
# plt.grid(True)
# plt.legend()
# plt.xlabel("Time")
# ax1.set_ylabel("Temperature (K)")
# ax2.set_ylabel("Resistance (Ohms)")
# ax1.set_title("Cooldown over Time")

# fig2 = plt.figure(2)
# ax1_2 = fig2.gca()
# ax1_2.plot(temperatures, resistances-sc_res, linestyle=':', marker='.', color=(0.7, 0, 0))
# ax1_2.set_ylabel("Resistance (Ohms)")
# ax1_2.set_xlabel("Temperature (K)")
# ax1_2.set_title("Resistance over Temperature")
# # ax1_2.set_xlim([2.8, 4.4])
# plt.grid(True)

# # Save pickled-figs
# pickle.dump(fig1, open(os.path.join("..", "Figures", f"AS3-fig1-{filename}.pklfig"), 'wb'))
# pickle.dump(fig2, open(os.path.join("..", "Figures", f"AS3-fig2-{filename}"), 'wb'))

# # Save pickled-figs
# pickle.dump(fig1, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig1.pklfig"), 'wb'))
# pickle.dump(fig2, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig2.pklfig"), 'wb'))












plt.show()




