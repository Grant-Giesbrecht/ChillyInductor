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
import matplotlib.font_manager as fm

parser = argparse.ArgumentParser()
parser.add_argument('--font', help='Use custom fonts', action='store_true')
parser.add_argument('--twilight', help='Use twighlight_shifted cmap', action='store_true')
args = parser.parse_args()

if args.font:
	
	# fontfam = fm.FontProperties(fname="C:\\Users\\grant\\Downloads\\Microsoft Aptos Fonts\\Aptos.ttf")
	# print(f"Changing font to: {fontfam.get_name()}")
	# plt.rcParams['font.family'] = fontfam.get_name()
	
	plt.rcParams['font.family'] = 'Aptos'
	
#------------------------------------------------------------
# Import Data
datapath = get_datadir_path(rp=22, smc='A')
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
filename = "dMS4_t2_10June2024_r1.hdf"

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


##-----------------

## Plot bias currents

if not args.font:
	plt.figure(1)
	plt.plot(df_cond.Ibias_mA, marker='.', markersize=5, linestyle=':', linewidth=0.2, color=(0, 0, 0.8))
	plt.plot(Ibias_bin, marker='.', markersize=5, linestyle=':', linewidth=0.2, color=(0, 0.7, 0))
	plt.grid(True)
	plt.xlabel("Sweep Index")
	plt.ylabel("Bias Current (mA)")

	plt.figure(2)
	plt.plot(np.diff(df_cond.Ibias_mA), marker='.', markersize=5, linestyle=':', linewidth=0.2)
	plt.grid(True)

	print(f"mean diff = {np.mean(np.diff(df_cond.Ibias_mA))}")
	print(f"max diff = {np.max(np.diff(df_cond.Ibias_mA))}")

	plt.show()


##------------------------

color_map = 'magma'
# color_map = 'Blues'
# color_map = 'bone'
if args.twilight:
	color_map = 'twilight_shifted'

## Make graphs of each power level

# Deploy ipython magic to get the graphs windowed rather than inline

X,Y,Zmx1l = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_mx1l', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=1, show_markers=True, hovertips=False, cmap=color_map)
X,Y,Zrf1 = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_rf1', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=2, show_markers=True, hovertips=False, cmap=color_map)
X,Y,Zlo1 = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_lo1', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=3, show_markers=True, hovertips=False, cmap=color_map)

X,Y,Zmx1h = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_mx1h', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=4, show_markers=True, hovertips=False, cmap=color_map)
X,Y,Zmx2l = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_mx2l', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=5, show_markers=True, hovertips=False, cmap=color_map)
X,Y,Zmx2h = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_mx2h', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=6, show_markers=True, hovertips=False, cmap=color_map)

## Make graphs of mixing loss (X=f_rf, Y=f_lo)
lplot3d(X, Y, Zmx1l-Zrf1, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='mx1l-rf1', fig_no=7, show_markers=True, hovertips=False, cmap=color_map)

plt.show()


##---------------------------------------


## Make a graph of mixing loss (Fix: f_rf, X=bias, Y=f_lo)

X,Y,Zmx1l = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=1, show_markers=True, hovertips=False, cmap=color_map)
X,Y,Zmx1h = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=2, show_markers=True, hovertips=False, cmap=color_map)
X,Y,Zmx2l = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx2l', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=3, show_markers=True, hovertips=False, cmap=color_map)
X,Y,Zmx2h = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx2h', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=4, show_markers=True, hovertips=False, cmap=color_map)
X,Y,Zrf1 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf1', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=5, show_markers=True, hovertips=False, cmap=color_map)

lplot3d(X, Y, Zmx1l-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx1l-rf1', fig_no=7, show_markers=True, hovertips=False, cmap=color_map)
lplot3d(X, Y, Zmx1h-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx1h-rf1', xlabel="Bias Current (mA)", ylabel="Local Oscillator Frequency (GHz)", zlabel="(RF+LO) Mixing Gain (dB)", fig_no=8, show_markers=True, hovertips=False, cmap=color_map, markersize=10)

lplot3d(X, Y, Zmx2l-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx2l-rf1', xlabel="Bias Current (mA)", ylabel="Local Oscillator Frequency (GHz)", zlabel="MX2L Gain", fig_no=9, show_markers=True, hovertips=False, cmap=color_map)
lplot3d(X, Y, Zmx2h-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx2h-rf1', fig_no=10, show_markers=True, hovertips=False, cmap=color_map)

plt.show()