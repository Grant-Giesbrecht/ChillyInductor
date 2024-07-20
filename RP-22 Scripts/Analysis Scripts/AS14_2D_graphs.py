''' Built from AS12a. Intended to filter measurements s.t. they can be compared against simulated
or theoretical values. Will filter the measured data down to 2D plots by selecting a singular
fRF and fLO and scanning across Ibias.
'''

##===========================================================================
## Load everything!

import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
import sys
import os
import pandas as pd
import argparse
from ganymede import *

parser = argparse.ArgumentParser()
# parser.add_argument('-h', '--help')
parser.add_argument('-nc', '--nocal', help="Run without calibration analysis.", action='store_true')
args = parser.parse_args()

#------------------------------------------------------------
# Import Data
datapath = get_datadir_path(rp=22, smc='A')
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
filename = "dMS4_t3a_14June2024_r1.hdf"

analysis_file = os.path.join(datapath, filename)

USE_CONST_SCALE = True

##--------------------------------------------
# Read HDF5 File

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	READ_MODE = 'dataset'
	df_cond_dict = {'powermeter_dBm': fh[READ_MODE]['coupled_power_meas_dBm'],
			 'rf_enabled': fh[READ_MODE]['rf_enabled'],
			 'lo_enabled': fh[READ_MODE]['lo_enabled'],
			 'freq_rf_GHz': fh[READ_MODE]['freq_rf_GHz'],
			 'freq_lo_GHz': fh[READ_MODE]['freq_lo_GHz'],
			 'power_rf_dBm': fh[READ_MODE]['power_RF_dBm'],
			 'power_lo_dBm': fh[READ_MODE]['power_LO_dBm'],
			 'DC_current_est_V': fh[READ_MODE]['DC_current_est_V'],
			 'output_set_voltage_V': fh[READ_MODE]['output_set_voltage_V'],
			 'target_DC_current_mA': fh[READ_MODE]['target_DC_current_mA'],
			 }
	df_sa_dict = {'wav_f_Hz': list(fh[READ_MODE]['waveform_f_Hz']),
			 'wav_s_dBm': list(fh[READ_MODE]['waveform_s_dBm']),
			 'wav_rbw_Hz': list(fh[READ_MODE]['waveform_rbw_Hz']),
			 }
	
	READ_MODE = 'calibration_data'
	cf_cond_dict = {'powermeter_dBm': fh[READ_MODE]['coupled_power_meas_dBm'],
			 'rf_enabled': fh[READ_MODE]['rf_enabled'],
			 'lo_enabled': fh[READ_MODE]['lo_enabled'],
			 'freq_rf_GHz': fh[READ_MODE]['freq_rf_GHz'],
			 'freq_lo_GHz': fh[READ_MODE]['freq_lo_GHz'],
			 'power_rf_dBm': fh[READ_MODE]['power_RF_dBm'],
			 'power_lo_dBm': fh[READ_MODE]['power_LO_dBm']
			 }
	cf_sa_dict = {'wav_f_Hz': list(fh[READ_MODE]['waveform_f_Hz']),
			 'wav_s_dBm': list(fh[READ_MODE]['waveform_s_dBm']),
			 'wav_rbw_Hz': list(fh[READ_MODE]['waveform_rbw_Hz']),
			 }
	
	# Load configuration data
	dataset_config = json.loads(fh['info']['configuration'][()].decode())
	
	# Create calibration data dataframes
	cf_cond = pd.DataFrame(cf_cond_dict)
	cf_sa = pd.DataFrame(cf_sa_dict)
	
	# Create dataset dataframes
	df_cond = pd.DataFrame(df_cond_dict)
	df_sa = pd.DataFrame(df_sa_dict)
	
	# Trim values greater than 10e3
	X = 10e3
	I_match = (df_cond <= X).all(axis=1)
	df_cond = df_cond[I_match]
	df_sa = df_sa[I_match]
	I_match = (cf_cond <= X).all(axis=1)
	cf_cond = cf_cond[I_match]
	cf_sa = cf_sa[I_match]

R_sense = dataset_config['curr_sense_ohms']

# Create new column for estimated bias current
Ibias_raw =  df_cond.DC_current_est_V/R_sense*1e3
Ibias_bin = bin_empirical(Ibias_raw)
df_cond.insert(len(df_cond.columns), 'Ibias_mA', Ibias_raw)
df_cond.insert(len(df_cond.columns), 'Ibias_mA_bin', Ibias_bin)

# Calculate mixing data DF
df_mix = calc_mixing_data(df_cond, df_sa)

# Print some stats
print(f"{Fore.YELLOW}DF_COND:{Style.RESET_ALL}")
print(df_cond)
print(f"{Fore.YELLOW}DF_SA:{Style.RESET_ALL}")
print(df_sa)




# Select frequencies to plot
f_RF_GHz = 3.5
f_LO_GHz = 0.2

##===========================================================================
# Make 2D graphs of measured data

x,y_peak_mx1l, _ = dfplot2d(df_mix, xparam='Ibias_mA', yparam='peak_mx1l', fixedparam={'freq_rf_GHz':(f_RF_GHz, 0, 0.005), 'freq_lo_GHz': (f_LO_GHz, 0, 0.005)}, skip_plot=True, fig_no=1, subplot_no=None, show_markers=True, hovertips=False)
x,y_peak_mx1h, _ = dfplot2d(df_mix, xparam='Ibias_mA', yparam='peak_mx1h', fixedparam={'freq_rf_GHz':(f_RF_GHz, 0, 0.005), 'freq_lo_GHz': (f_LO_GHz, 0, 0.005)}, skip_plot=True, fig_no=1, subplot_no=None, show_markers=True, hovertips=False)

x,y_peak_mx2l, _ = dfplot2d(df_mix, xparam='Ibias_mA', yparam='peak_mx2l', fixedparam={'freq_rf_GHz':(f_RF_GHz, 0, 0.005), 'freq_lo_GHz': (f_LO_GHz, 0, 0.005)}, skip_plot=True, fig_no=1, subplot_no=None, show_markers=True, hovertips=False)
x,y_peak_mx2h, _ = dfplot2d(df_mix, xparam='Ibias_mA', yparam='peak_mx2h', fixedparam={'freq_rf_GHz':(f_RF_GHz, 0, 0.005), 'freq_lo_GHz': (f_LO_GHz, 0, 0.005)}, skip_plot=True, fig_no=1, subplot_no=None, show_markers=True, hovertips=False)

x,y_peak_rf1, _ = dfplot2d(df_mix, xparam='Ibias_mA', yparam='peak_rf1', fixedparam={'freq_rf_GHz':(f_RF_GHz, 0, 0.005), 'freq_lo_GHz': (f_LO_GHz, 0, 0.005)}, skip_plot=True, fig_no=1, subplot_no=None, show_markers=True, hovertips=False)
x,y_peak_rf2, _ = dfplot2d(df_mix, xparam='Ibias_mA', yparam='peak_rf2', fixedparam={'freq_rf_GHz':(f_RF_GHz, 0, 0.005), 'freq_lo_GHz': (f_LO_GHz, 0, 0.005)}, skip_plot=True, fig_no=1, subplot_no=None, show_markers=True, hovertips=False)
x,y_peak_rf3, _ = dfplot2d(df_mix, xparam='Ibias_mA', yparam='peak_rf3', fixedparam={'freq_rf_GHz':(f_RF_GHz, 0, 0.005), 'freq_lo_GHz': (f_LO_GHz, 0, 0.005)}, skip_plot=True, fig_no=1, subplot_no=None, show_markers=True, hovertips=False)


# fixedparam={'freq_rf_GHz':(f_RF_GHz, 0, 0.01), 'freq_lo_GHz': (f_LO_GHz, 0, 0.005), 'Ibias_mA': (0.64, 0, 0.1)}

plt.style.use("C:\\Users\\grant\\Documents\\GitHub\\ChillyInductor\\src\\custom.mplstyle.py")
plt.style.use('classic')
# plt.rcParams.update(plt.rcParamsDefault)
# plt.style.use('ggplot')

plt.figure(1)
plt.plot(x, y_peak_mx1l, linestyle=':', linewidth=1, marker='v', markersize=4, color=(0.4, 0, 0.6), label='MX1l')
plt.plot(x, y_peak_mx1h, linestyle=':', linewidth=1, marker='^', markersize=4, color=(0.4, 0, 0.6), label='MX1h')
plt.plot(x, y_peak_mx2l, linestyle=':', linewidth=1, marker='v', markersize=4, color=(0.88, 0.44, 0), label='MX2l')
plt.plot(x, y_peak_mx2h, linestyle=':', linewidth=1, marker='^', markersize=4, color=(0.88, 0.44, 0), label='MX2h')
plt.plot(x, y_peak_rf1, linestyle=':', linewidth=1, marker='.', markersize=4, color=(0, 0.44, 0.01), label='RF1')
plt.plot(x, y_peak_rf2, linestyle=':', linewidth=1, marker='+', markersize=4, color=(0, 0.44, 0.01), label='RF2')
plt.plot(x, y_peak_rf3, linestyle=':', linewidth=1, marker='x', markersize=4, color=(0, 0.44, 0.01), label='RF3')
plt.xlabel("Ibias mA")
plt.ylabel("Peak Height (dBm)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(x, dBm2W(y_peak_mx1l), linestyle=':', linewidth=1, marker='v', markersize=4, color=(0.4, 0, 0.6), label='MX1l')
plt.plot(x, dBm2W(y_peak_mx1h), linestyle=':', linewidth=1, marker='^', markersize=4, color=(0.4, 0, 0.6), label='MX1h')
plt.xlabel("Ibias mA")
plt.ylabel("Peak Height (W)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, dBm2W(y_peak_mx2l), linestyle=':', linewidth=1, marker='v', markersize=4, color=(0.88, 0.44, 0), label='MX2l')
plt.plot(x, dBm2W(y_peak_mx2h), linestyle=':', linewidth=1, marker='^', markersize=4, color=(0.88, 0.44, 0), label='MX2h')
plt.plot(x, dBm2W(y_peak_rf2), linestyle=':', linewidth=1, marker='+', markersize=4, color=(0, 0.44, 0.01), label='RF2')
plt.plot(x, dBm2W(y_peak_rf3), linestyle=':', linewidth=1, marker='x', markersize=4, color=(0, 0.44, 0.01), label='RF3')
plt.xlabel("Ibias mA")
plt.ylabel("Peak Height (W)")
plt.grid(True)
plt.legend()
plt.show()
	