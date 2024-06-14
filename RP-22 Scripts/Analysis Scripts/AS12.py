''' This takes the functionality of AS11.ipynb and puts it into a script form. 

Adds in calibration abilities

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

##===========================================================================
# Plot bias currents

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(df_cond.Ibias_mA, marker='.', markersize=5, linestyle=':', linewidth=0.2, color=(0, 0, 0.8))
plt.plot(Ibias_bin, marker='.', markersize=5, linestyle=':', linewidth=0.2, color=(0, 0.7, 0))
plt.grid(True)
plt.xlabel("Sweep Index")
plt.ylabel("Bias Current (mA)")

plt.subplot(1, 2, 2)
plt.plot(np.diff(df_cond.Ibias_mA), marker='.', markersize=5, linestyle=':', linewidth=0.2)
plt.grid(True)

plt.suptitle("Bias Current Analysis")


print(f"mean diff = {np.mean(np.diff(df_cond.Ibias_mA))}")
print(f"max diff = {np.max(np.diff(df_cond.Ibias_mA))}")

##===========================================================================
# Plot graphs of power levels

X,Y,Zmx1l = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_mx1l', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=2, subplot_no=(2, 3, 1), show_markers=True, hovertips=False)
X,Y,Zrf1 = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_rf1', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=2, subplot_no=(2, 3, 2), show_markers=True, hovertips=False)
X,Y,Zlo1 = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_lo1', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=2, subplot_no=(2, 3, 3), show_markers=True, hovertips=False)

X,Y,Zmx1h = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_mx1h', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=2, subplot_no=(2, 3, 4), show_markers=True, hovertips=False)
X,Y,Zmx2l = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_mx2l', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=2, subplot_no=(2, 3, 5), show_markers=True, hovertips=False)
X,Y,Zmx2h = dfplot3d(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_mx2h', fixedparam={'Ibias_mA':(0, 0, 0.2)}, skip_plot=False, fig_no=2, subplot_no=(2, 3, 6), show_markers=True, hovertips=False)
plt.suptitle("Peaks (fRF-fLO Space)")

##===========================================================================
# Make a graph of mixing loss (Fix: f_rf, X=bias, Y=f_lo)

X,Y,Zmx1l = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 1), show_markers=True, hovertips=False)
X,Y,Zmx1h = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 2), show_markers=True, hovertips=False)
X,Y,Zmx2l = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx2l', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 3), show_markers=True, hovertips=False)
X,Y,Zmx2h = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx2h', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 4), show_markers=True, hovertips=False)
X,Y,Zrf1 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf1', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 5), show_markers=True, hovertips=False)
plt.suptitle("Peaks (Bias-fLO Space)")

COLOR_MAP = 'plasma'

lplot3d(X, Y, Zmx1l-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx1l-rf1', fig_no=4, subplot_no=(2, 2, 1), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(X, Y, Zmx1h-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx1h-rf1', fig_no=4, subplot_no=(2, 2, 2), show_markers=True, hovertips=False, cmap=COLOR_MAP)

lplot3d(X, Y, Zmx2l-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx2l-rf1', fig_no=4, subplot_no=(2, 2, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(X, Y, Zmx2h-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx2h-rf1', fig_no=4, subplot_no=(2, 2, 4), show_markers=True, hovertips=False, cmap=COLOR_MAP)
plt.suptitle("Mixing Gain (Bias-fLO Space)")

##===========================================================================
# Quit if no calibration requested

if args.nocal:
	plt.show()
	sys.exit()

##===========================================================================
# Begin calibration step

##--------------------------------------------
# Read HDF5 File - input loss

# filename = "dMS3_t2_11June2024.hdf"
filename = "dMS3_t3_11June2024_r1.hdf"

analysis_file = os.path.join(datapath, filename)

USE_CONST_SCALE = True

@dataclass
class PowerSpectrum:
	
	p_fund:float = None
	p_2h:float = None
	p_3h:float = None
	p_pm:float = None

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	READ_MODE = 'calibration_data'
	cal_dict = {'powermeter_dBm': fh[READ_MODE]['coupled_power_meas_dBm'],
			 'rf_enabled': fh[READ_MODE]['rf_enabled'],
			 'lo_enabled': fh[READ_MODE]['lo_enabled'],
			 'freq_rf_GHz': fh[READ_MODE]['freq_rf_GHz'],
			 'freq_lo_GHz': fh[READ_MODE]['freq_lo_GHz'],
			 'power_rf_dBm': fh[READ_MODE]['power_RF_dBm'],
			 'power_lo_dBm': fh[READ_MODE]['power_LO_dBm'],
			 'wav_f_Hz': list(fh[READ_MODE]['waveform_f_Hz']),
			 'wav_s_dBm': list(fh[READ_MODE]['waveform_s_dBm']),
			 'wav_rbw_Hz': list(fh[READ_MODE]['waveform_rbw_Hz']),
			 }
	
	source_script = fh['info']['source_script'][()]
	conf_json = fh['info']['configuration'][()]
	operator_notes = fh['info']['operator_notes'][()]
	
	# Create calibration data dataframes
	df_cal = pd.DataFrame(cal_dict)

print(f"{Fore.YELLOW}DF_CAL:{Style.RESET_ALL}")
print(df_cal)

# Add columns to DF for power at each harmonic and total power

# Assign row one at a time
data_block = []
for index, row in df_cal.iterrows():
	
	# Get mixing products for this row
	nps = calc_harm_data_single(df_cal.iloc[index, :], df_cal.iloc[index, :])
	
	nps_list = [nps.rf1, nps.rf2, nps.rf3, nps.lo1, nps.lo2, nps.lo3, nps.total]
	
	# Append to data block
	data_block.append(nps_list)

df_temp = pd.DataFrame(data_block, columns=['peak_rf1', 'peak_rf2', 'peak_rf3', 'peak_lo1', 'peak_lo2', 'peak_lo3', 'spectrum_total'])
df_cal = pd.merge(df_cal, df_temp, left_index=True, right_index=True, how='outer')

##--------------------------------------------
# Plot loss in (Freq-Power space)

X,Y,Z_SaTotRf = dfplot3d(df_cal, xparam='power_rf_dBm', yparam='freq_rf_GHz', zparam='spectrum_total', fixedparam={'rf_enabled':1}, skip_plot=False, fig_no=5, subplot_no=(1, 2, 1), show_markers=True, hovertips=False)
X,Y,Z_PmRf = dfplot3d(df_cal, xparam='power_rf_dBm', yparam='freq_rf_GHz', zparam='powermeter_dBm', fixedparam={'rf_enabled':1}, skip_plot=False, fig_no=5, subplot_no=(1, 2, 2), show_markers=True, hovertips=False)
plt.suptitle("Calibration: Input Measurements")

lplot3d(X, Y, Z_SaTotRf-Z_PmRf, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='SaTotRf-PmRf', fig_no=6, show_markers=True, hovertips=False)
plt.suptitle("Calibration: Input Loss")

##--------------------------------------------
# Read HDF5 File - output loss

filename = "OutputCal_12June2024.hdf"

analysis_file = os.path.join(datapath, filename)

USE_CONST_SCALE = True

@dataclass
class PowerSpectrum:
	
	p_fund:float = None
	p_2h:float = None
	p_3h:float = None
	p_pm:float = None

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	READ_MODE = 'output_cal'
	sp_dict = {'S11_dB': fh[READ_MODE]['S11_dB'],
			 'S22_dB': fh[READ_MODE]['S22_dB'],
			 'S21_dB': fh[READ_MODE]['S21_dB'],
			 'S12_dB': fh[READ_MODE]['S12_dB'],
			 'S11_rad': fh[READ_MODE]['S11_rad'],
			 'S22_rad': fh[READ_MODE]['S22_rad'],
			 'S21_rad': fh[READ_MODE]['S21_rad'],
			 'S12_rad': fh[READ_MODE]['S12_rad'],
			 'freq_Hz': fh[READ_MODE]['freq_Hz'],
			 }
	
	# Create calibration data dataframes
	df_sparam = pd.DataFrame(sp_dict)

print(f"{Fore.YELLOW}DF_SPARAM:{Style.RESET_ALL}")
print(df_sparam)

##--------------------------------------------
# Plot output loss s-parameters

mksz = 6

plt.figure(7)
plt.plot(df_sparam.freq_Hz/1e9, df_sparam.S11_dB, linestyle=':', color="#fcbe03", marker='.', markersize=mksz, label="S11")
plt.plot(df_sparam.freq_Hz/1e9, df_sparam.S22_dB, linestyle=':', color="#eb0f0c", marker='.', markersize=mksz, label="S22")
plt.plot(df_sparam.freq_Hz/1e9, df_sparam.S21_dB, linestyle=':', color="#03bafc", marker='.', markersize=mksz, label="S21")
plt.plot(df_sparam.freq_Hz/1e9, df_sparam.S12_dB, linestyle=':', color="#03a32b", marker='.', markersize=mksz, label="S12")

plt.xlabel("Frequency (GHz)")
plt.ylabel("(dB)")
plt.legend()
plt.title("Calibration: Output S-Parameters")

plt.grid(True)

##--------------------------------------------
# Define calibrated loss functions

delta = make_loss_lookup_fn(df_sparam.freq_Hz, df_sparam.S21_dB)

##--------------------------------------------
# Apply loss compensation to DF

def make_cal_fn(peak_name, freq_name:str, freq_b:str='freq_lo_GHz', rf_mult:int=1, lo_mult:int=0):
	
	def fn(row):
		return row[peak_name] - delta(row[freq_name]*1e9*rf_mult + row[freq_b]*1e9*lo_mult)
	
	return fn

df_mix['peak_rf1@2'] = df_mix.apply(make_cal_fn('peak_rf1', 'freq_rf_GHz'), axis=1)
df_mix['peak_mx1l@2'] = df_mix.apply(make_cal_fn('peak_mx1l', 'freq_rf_GHz', lo_mult=-1), axis=1)


##--------------------------------------------
# Make a graph of calibrated mixing loss (Fix: f_rf, X=bias, Y=f_lo)

COLOR_MAP = 'plasma'

X,Y,Zmx1l_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l@2', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=8, subplot_no=(1, 3, 1), show_markers=True, hovertips=False)
X,Y,Zrf1_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf1@2', fixedparam={'freq_rf_GHz':(4, 0, 0.1)}, skip_plot=False, fig_no=8, subplot_no=(1, 3, 2), show_markers=True, hovertips=False)
lplot3d(X, Y, Zmx1l_PL2-Zrf1_PL2, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='Plane 2: mx1l-rf1', fig_no=8, subplot_no=(1, 3, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)

X,Y,Zmx1l = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_mx1l', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=9, subplot_no=(1, 3, 1), show_markers=True, hovertips=False)
X,Y,Zrf1 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_rf1', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=9, subplot_no=(1, 3, 2), show_markers=True, hovertips=False)
lplot3d(X, Y, Zmx1l-Zrf1, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='Plane 1: mx1l-rf1', fig_no=9, subplot_no=(1, 3, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)

X,Y,Zmx1l_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_mx1l@2', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=10, subplot_no=(1, 3, 1), show_markers=True, hovertips=False)
X,Y,Zrf1_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_rf1@2', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=10, subplot_no=(1, 3, 2), show_markers=True, hovertips=False)
lplot3d(X, Y, Zmx1l_PL2-Zrf1_PL2, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='Plane 2: mx1l-rf1', fig_no=10, subplot_no=(1, 3, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)

plt.show()