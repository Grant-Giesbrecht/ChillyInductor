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

#------------------------------------------------------------
# Import Data

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm'])

if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

filename = "RP22B_MP3_t1_31July2024_R4C4T1_r1_autosave.hdf"

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
fig1.subplots_adjust(left=0.065, bottom=0.065, top=0.95, right=0.8)

freq_list = np.unique(freq_rf_GHz)
pwr_list = np.unique(power_rf_dBm)
req_bias_list = np.unique(requested_Idc_mA)

conditions = {'sel_freq_GHz': freq_list[len(freq_list)//2], 'sel_power_dBm': pwr_list[len(pwr_list)//2]}

print(freq_list)
print(pwr_list)

def plot_selcted_data():
	
	f = conditions['sel_freq_GHz']
	p = conditions['sel_power_dBm']
	
	# Filter relevant data
	mask_freq = (freq_rf_GHz == conditions['sel_freq_GHz'])
	mask_pwr = (power_rf_dBm == conditions['sel_power_dBm'])
	mask = (mask_freq & mask_pwr)
	
	# Plot results
	ax1.cla()
	
	# Check correct number of points
	mask_len = np.sum(mask)
	if len(req_bias_list) != mask_len:
		log.warning(f"Cannot display data: Mismatched number of points (mask: {mask_len}, bias: {len(req_bias_list)})")
		fig1.canvas.draw_idle()
		return
	
	
	ax1.plot(Idc_mA[mask], rf1[mask], linestyle=':', marker='o', markersize=1.5, color=(0, 0.7, 0))
	ax1.plot(Idc_mA[mask], rf2[mask], linestyle=':', marker='o', markersize=1.5, color=(0, 0, 0.7))
	ax1.plot(Idc_mA[mask], rf3[mask], linestyle=':', marker='o', markersize=1.5, color=(0.7, 0, 0))
	ax1.set_title(f"f = {f} GHz, p = {p} dBm")
	ax1.set_xlabel("DC Bias (mA)")
	ax1.set_ylabel("Power (dBm)")
	ax1.legend(["Fundamental", "2nd Harm.", "3rd Harm."])
	ax1.grid(True)
	
	fig1.canvas.draw_idle()
	
	

def update_pwr(x):
	conditions['sel_power_dBm'] = x
	plot_selcted_data()
	
def update_freq(x):
	conditions['sel_freq_GHz'] = x
	plot_selcted_data()

# Frequency Slider
ax_freq = fig1.add_axes([0.84, 0.1, 0.03, 0.8])
slider_freq = Slider(ax_freq, 'Freq (GHz)', np.min(freq_list), np.max(freq_list), initcolor='none', valstep=freq_list, color='green', orientation='vertical', valinit=conditions['sel_freq_GHz'])
slider_freq.on_changed(update_freq)

# Power Slider
ax_pwr = fig1.add_axes([0.93, 0.1, 0.03, 0.8])
slider_pwr = Slider(ax_pwr, 'Power (dBm)', np.min(pwr_list), np.max(pwr_list), initcolor='none', valstep=pwr_list, color='red', orientation='vertical', valinit=conditions['sel_power_dBm'])
slider_pwr.on_changed(update_pwr)

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
# pickle.dump(fig2, open(os.path.join("..", "Figures", f"AS3-fig2-{filename}.pklfig"), 'wb'))

# # Save pickled-figs
# pickle.dump(fig1, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig1.pklfig"), 'wb'))
# pickle.dump(fig2, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig2.pklfig"), 'wb'))












plt.show()



