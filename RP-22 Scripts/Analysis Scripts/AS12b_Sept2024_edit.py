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
parser.add_argument('--t3a', help="Use t3a datafile", action='store_true')
parser.add_argument('--font', help='Use custom fonts', action='store_true')
parser.add_argument('--twilight', help='Use twighlight_shifted cmap', action='store_true')
args = parser.parse_args()

if args.font:
	plt.rcParams['font.family'] = 'Aptos'

color_map = 'magma'
if args.twilight:
	color_map = 'twilight_shifted'

#------------------------------------------------------------
# Import Data
datapath = get_datadir_path(rp=22, smc='A')
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

if args.t3a:
	filename = "dMS4_t3a_14June2024_r1.hdf"
else:
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

COLOR_MAPNEW = color_map

freq_filt = 4
if args.t3a:
	freq_filt = 3.5

X,Y,Zmx1l = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 1), show_markers=True, hovertips=False)
X,Y,Zmx1h = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 2), show_markers=True, hovertips=False)
X,Y,Zmx2l = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx2l', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 3), show_markers=True, hovertips=False)
X,Y,Zmx2h = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx2h', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 4), show_markers=True, hovertips=False)
X,Y,Zrf1 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf1', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 5), show_markers=True, hovertips=False)
X,Y,Zlo1 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_lo1', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=3, subplot_no=(2, 3, 6), show_markers=True, hovertips=False, cmap=COLOR_MAPNEW)
plt.suptitle("Peaks (Bias-fLO Space)")

COLOR_MAP = 'plasma'

Zrflo = W2dBm(dBm2W(Zrf1)+dBm2W(Zlo1))

lplot3d(X, Y, Zmx1l-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx1l-rf1', fig_no=4, subplot_no=(2, 2, 1), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(X, Y, Zmx1h-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx1h-rf1', fig_no=4, subplot_no=(2, 2, 2), show_markers=True, hovertips=False, cmap=COLOR_MAP)

lplot3d(X, Y, Zmx2l-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx2l-rf1', fig_no=4, subplot_no=(2, 2, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(X, Y, Zmx2h-Zrf1, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx2h-rf1', fig_no=4, subplot_no=(2, 2, 4), show_markers=True, hovertips=False, cmap=COLOR_MAP)
plt.suptitle("Mixing Gain (Bias-fLO Space)")

lplot3d(X, Y, Zmx1l-Zrflo, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx1l-rf1-lo1', fig_no=11, subplot_no=(2, 2, 1), show_markers=True, hovertips=False, cmap=COLOR_MAPNEW)
lplot3d(X, Y, Zmx1h-Zrflo, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx1h-rf1-lo1', fig_no=11, subplot_no=(2, 2, 2), show_markers=True, hovertips=False, cmap=COLOR_MAPNEW)

lplot3d(X, Y, Zmx2l-Zrflo, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx2l-rf1-lo1', fig_no=11, subplot_no=(2, 2, 3), show_markers=True, hovertips=False, cmap=COLOR_MAPNEW)
lplot3d(X, Y, Zmx2h-Zrflo, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='mx2h-rf1-lo1', fig_no=11, subplot_no=(2, 2, 4), show_markers=True, hovertips=False, cmap=COLOR_MAPNEW)
plt.suptitle("Total Mixing Gain (Bias-fLO Space)")

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

print(f"Analyzing calibration data...")
t0 = time.time()

data_block = []
for index, row in df_cal.iterrows():
	
	# Get mixing products for this row
	nps = calc_harm_data_single(df_cal.iloc[index, :], df_cal.iloc[index, :])
	
	nps_list = [nps.rf1, nps.rf2, nps.rf3, nps.lo1, nps.lo2, nps.lo3, nps.total]
	
	# Append to data block
	data_block.append(nps_list)

df_temp = pd.DataFrame(data_block, columns=['peak_rf1', 'peak_rf2', 'peak_rf3', 'peak_lo1', 'peak_lo2', 'peak_lo3', 'spectrum_total'])
df_cal = pd.merge(df_cal, df_temp, left_index=True, right_index=True, how='outer')

print(f"Finsihed analyzing calibration data in {rd(time.time()-t0)} seconds.")
##--------------------------------------------
# Plot loss in (Freq-Power space)

X,Y,Z_SaTotRf = dfplot3d(df_cal, xparam='power_rf_dBm', yparam='freq_rf_GHz', zparam='spectrum_total', fixedparam={'rf_enabled':1}, skip_plot=False, fig_no=5, subplot_no=(2, 3, 1), show_markers=True, hovertips=False, cmap=color_map)
X,Y,Z_PmRf = dfplot3d(df_cal, xparam='power_rf_dBm', yparam='freq_rf_GHz', zparam='powermeter_dBm', fixedparam={'rf_enabled':1}, skip_plot=False, fig_no=5, subplot_no=(2, 3, 2), show_markers=True, hovertips=False, cmap=color_map)
X,Y,Z_SARf1 = dfplot3d(df_cal, xparam='power_rf_dBm', yparam='freq_rf_GHz', zparam='peak_rf1', fixedparam={'rf_enabled':1}, skip_plot=False, fig_no=5, subplot_no=(2, 3, 3), show_markers=True, hovertips=False, cmap=color_map)
X,Y,Z_SARf2 = dfplot3d(df_cal, xparam='power_rf_dBm', yparam='freq_rf_GHz', zparam='peak_rf2', fixedparam={'rf_enabled':1}, skip_plot=False, fig_no=5, subplot_no=(2, 3, 4), show_markers=True, hovertips=False, cmap=color_map)
X,Y,Z_SARf3 = dfplot3d(df_cal, xparam='power_rf_dBm', yparam='freq_rf_GHz', zparam='peak_rf3', fixedparam={'rf_enabled':1}, skip_plot=False, fig_no=5, subplot_no=(2, 3, 5), show_markers=True, hovertips=False, cmap=color_map)
plt.suptitle("Calibration: Input Measurements")

lplot3d(X, Y, Z_SaTotRf-Z_PmRf, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='SaTotRf-PmRf', fig_no=6, show_markers=True, hovertips=False)
plt.suptitle("Calibration: Input Loss")

tot_rf_power_W = dBm2W(Z_SARf1) + dBm2W(Z_SARf2) + dBm2W(Z_SARf3)
powermeter_gain = Z_PmRf - W2dBm(tot_rf_power_W)

X_ = np.array([xx[1:] for xx in X])
Y_ = np.array([yy[1:] for yy in Y])
pmg_ = np.array([pmg[1:] for pmg in powermeter_gain])

lplot3d(X, Y, powermeter_gain, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='Power Meter Gain', fig_no=12, show_markers=True, hovertips=False, cmap=color_map)
plt.suptitle("Power Meter Gain")

lplot3d(X_, Y_, pmg_, xparam='Bias Current (mA)', yparam='Local Oscillator Frequency (GHz)', zparam='Gain (dB)', fig_no=21, show_markers=True, hovertips=False, cmap=color_map)
plt.suptitle("Input Coupler Gain")

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

plt.figure(13)
plt.plot(Y[X==-10], powermeter_gain[X==-10], marker='.', linestyle=':')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power Meter Gain (dB)")
plt.grid(True)

out_delta = make_loss_lookup_fn(df_sparam.freq_Hz, df_sparam.S21_dB)
in_delta = make_loss_lookup_fn(Y[X==-10]*1e9, powermeter_gain[X==-10])

##--------------------------------------------
# Apply loss compensation to DF

def make_outcal_fn(peak_name, freq_name:str, freq_b:str='freq_lo_GHz', rf_mult:int=1, lo_mult:int=0):
	
	def fn(row):
		return row[peak_name] - out_delta(row[freq_name]*1e9*rf_mult + row[freq_b]*1e9*lo_mult)
	
	return fn

def make_incal_fn(peak_name, freq_name:str, freq_b:str='freq_lo_GHz', rf_mult:int=1, lo_mult:int=0):
	
	def fn(row):
		return row[peak_name] - in_delta(row[freq_name]*1e9*rf_mult + row[freq_b]*1e9*lo_mult)
	
	return fn

# Calculate parameters at "plane 2", the output of the cryostat
df_mix['peak_rf1@2'] = df_mix.apply(make_outcal_fn('peak_rf1', 'freq_rf_GHz'), axis=1)
df_mix['peak_rf2@2'] = df_mix.apply(make_outcal_fn('peak_rf2', 'freq_rf_GHz'), axis=1)
df_mix['peak_rf3@2'] = df_mix.apply(make_outcal_fn('peak_rf3', 'freq_rf_GHz'), axis=1)
df_mix['peak_mx1l@2'] = df_mix.apply(make_outcal_fn('peak_mx1l', 'freq_rf_GHz', lo_mult=-1), axis=1)
df_mix['peak_mx1h@2'] = df_mix.apply(make_outcal_fn('peak_mx1h', 'freq_rf_GHz', lo_mult=1), axis=1)
df_mix['peak_mx2l@3'] = df_mix.apply(make_incal_fn('peak_mx2l', 'freq_rf_GHz', lo_mult=-2), axis=1)
df_mix['peak_mx2h@3'] = df_mix.apply(make_incal_fn('peak_mx2h', 'freq_rf_GHz', lo_mult=2), axis=1)
df_mix['peak_lo1@2'] = df_mix.apply(make_outcal_fn('peak_lo1', 'freq_lo_GHz'), axis=1)

##--------------------------------------------
# Make a graph of calibrated mixing loss (Fix: f_rf, X=bias, Y=f_lo)

COLOR_MAP = 'plasma'

X,Y,Zmx1l_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l@2', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=8, subplot_no=(2, 3, 1), show_markers=True, hovertips=False)
X,Y,Zmx1h_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1h@2', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=8, subplot_no=(2, 3, 2), show_markers=True, hovertips=False)
X,Y,Zrf1_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf1@2', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=8, subplot_no=(2, 3, 3), show_markers=True, hovertips=False)
X,Y,Zrf2_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf2@2', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=8, subplot_no=(2, 3, 4), show_markers=True, hovertips=False)
X,Y,Zrf3_PL2 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf3@2', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=8, subplot_no=(2, 3, 5), show_markers=True, hovertips=False)
lplot3d(X, Y, Zmx1l_PL2-Zrf1_PL2, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='Plane 2: mx1l-rf1', fig_no=8, subplot_no=(2, 3, 6), show_markers=True, hovertips=False, cmap=COLOR_MAP)
plt.suptitle("Plane 2 (bias-flo space)")
plt.suptitle("Spectral Components at Plane 2, Cryostat Output")

X,Y,Zmx1l = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_mx1l', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=9, subplot_no=(1, 3, 1), show_markers=True, hovertips=False)
X,Y,Zrf1 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_rf1', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=9, subplot_no=(1, 3, 2), show_markers=True, hovertips=False)
lplot3d(X, Y, Zmx1l-Zrf1, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='Plane 1: mx1l-rf1', fig_no=9, subplot_no=(1, 3, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)
plt.suptitle("Plane 1 (bias-fRF space)")

X,Y,Zmx1l_PL2rf = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_mx1l@2', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=10, subplot_no=(1, 3, 1), show_markers=True, hovertips=False)
Xrf,Yrf,Zrf1_PL2rf = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_rf1@2', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=10, subplot_no=(1, 3, 2), show_markers=True, hovertips=False)
lplot3d(X, Y, Zmx1l_PL2rf-Zrf1_PL2rf, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='Plane 2: mx1l-rf1', fig_no=10, subplot_no=(1, 3, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)
plt.suptitle("Plane 2 (bias-fRF space)")

# Calculate parameters at "plane 3", the input of the cryostat
df_mix['peak_rf1@3'] = df_mix.apply(make_incal_fn('peak_rf1', 'freq_rf_GHz'), axis=1)
df_mix['peak_rf2@3'] = df_mix.apply(make_incal_fn('peak_rf2', 'freq_rf_GHz'), axis=1)
df_mix['peak_rf3@3'] = df_mix.apply(make_incal_fn('peak_rf3', 'freq_rf_GHz'), axis=1)
df_mix['peak_mx1l@3'] = df_mix.apply(make_incal_fn('peak_mx1l', 'freq_rf_GHz', lo_mult=-1), axis=1)
df_mix['peak_mx1h@3'] = df_mix.apply(make_incal_fn('peak_mx1h', 'freq_rf_GHz', lo_mult=-1), axis=1)

df_mix['peak_lo1@3'] = df_mix.apply(make_incal_fn('peak_lo1', 'freq_lo_GHz'), axis=1)

X,Y,Zmx1l_PL3 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1l@3', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=13, subplot_no=(2, 3, 1), show_markers=True, hovertips=False)
X,Y,Zmx1h_PL3 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1h@3', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=13, subplot_no=(2, 3, 2), show_markers=True, hovertips=False)
X,Y,Zrf1_PL3 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf1@3', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=13, subplot_no=(2, 3, 3), show_markers=True, hovertips=False)
X,Y,Zrf2_PL3 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf2@3', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=13, subplot_no=(2, 3, 4), show_markers=True, hovertips=False)
X,Y,Zrf3_PL3 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_rf3@3', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=13, subplot_no=(2, 3, 5), show_markers=True, hovertips=False)
Xrf,Yrf,Zrf1_PL3rf = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_rf1@3', fixedparam={'freq_lo_GHz':(0.4, 0, 0.01)}, skip_plot=False, fig_no=13, subplot_no=(2, 3, 6), show_markers=True, hovertips=False)
plt.suptitle("Spectral Components at Plane 3, Cryostat Input")

lplot3d(X, Y, Zmx1l_PL3-Zrf1_PL3, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='MX1L@3 - RF1@3', fig_no=14, subplot_no=(2, 2, 1), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(X, Y, Zmx1l_PL2-Zrf1_PL3, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='MX1L@2 - RF1@3', fig_no=14, subplot_no=(2, 2, 2), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(X, Y, Zrf1_PL2-Zrf1_PL3, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='RF1@2 - RF1@3', fig_no=14, subplot_no=(2, 2, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(Xrf, Yrf, Zrf1_PL2rf-Zrf1_PL3rf, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='RF1@2 - RF1@3', fig_no=14, subplot_no=(2, 2, 4), show_markers=True, hovertips=False, cmap=COLOR_MAP)
# lplot3d(Xff, Yff, Zrf1_PL2ff-Zrf1_PL3ff, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='RF1@2 - RF1@3', fig_no=14, subplot_no=(2, 3, 4), show_markers=True, hovertips=False, cmap=COLOR_MAP)

## ------------------------------------
# Show lots of detail on loss through cryostat to make sure this methodology is somewhat sound

print(np.unique(df_mix['Ibias_mA_bin']))

bias_val = 0
bias_thresh = 0.05

bias_val = 7.12
bias_thresh = 0.05

Xff,Yff, Zrf1_PL3ff = dfplot3d(df_mix, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='peak_rf1@3', fixedparam={'Ibias_mA_bin':(bias_val, 0, bias_thresh)}, skip_plot=False, fig_no=15, subplot_no=(2, 3, 1), show_markers=True, hovertips=False)
_,_, Zrf1_PL2ff = dfplot3d(df_mix, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='peak_rf1@2', fixedparam={'Ibias_mA_bin':(bias_val, 0, bias_thresh)}, skip_plot=False, fig_no=15, subplot_no=(2, 3, 2), show_markers=True, hovertips=False)
_,_, Zrf2_PL3ff = dfplot3d(df_mix, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='peak_rf2@3', fixedparam={'Ibias_mA_bin':(bias_val, 0, bias_thresh)}, skip_plot=False, fig_no=15, subplot_no=(2, 3, 3), show_markers=True, hovertips=False)
_,_, Zrf2_PL2ff = dfplot3d(df_mix, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='peak_rf2@2', fixedparam={'Ibias_mA_bin':(bias_val, 0, bias_thresh)}, skip_plot=False, fig_no=15, subplot_no=(2, 3, 4), show_markers=True, hovertips=False)
_,_, Zlo1_PL3ff = dfplot3d(df_mix, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='peak_lo1@3', fixedparam={'Ibias_mA_bin':(bias_val, 0, bias_thresh)}, skip_plot=False, fig_no=15, subplot_no=(2, 3, 5), show_markers=True, hovertips=False)
_,_, Zlo1_PL2ff = dfplot3d(df_mix, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='peak_lo1@2', fixedparam={'Ibias_mA_bin':(bias_val, 0, bias_thresh)}, skip_plot=False, fig_no=15, subplot_no=(2, 3, 6), show_markers=True, hovertips=False)

lplot3d(Xff, Yff, Zrf1_PL2ff-Zrf1_PL3ff, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='RF1@2 - RF1@3', fig_no=16, subplot_no=(1, 3, 1), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(Xff, Yff, Zrf2_PL2ff-Zrf2_PL3ff, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='RF2@2 - RF2@3', fig_no=16, subplot_no=(1, 3, 2), show_markers=True, hovertips=False, cmap=COLOR_MAP)
lplot3d(Xff, Yff, Zlo1_PL2ff-Zlo1_PL3ff, xparam='freq_lo_GHz', yparam='freq_rf_GHz', zparam='LO1@2 - LO1@3', fig_no=16, subplot_no=(1, 3, 3), show_markers=True, hovertips=False, cmap=COLOR_MAP)

##--------------------------------------------------
# This graph is the big kahuna: Mixing gain looking at power in vs
# harmonic power out.

# Calculate remaining parameters
X,Y,Zmx1h_PL3 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx1h@3', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=18, subplot_no=(1, 3, 1), show_markers=True, hovertips=False)
_,_,Zmx2l_PL3 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx2l@3', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=18, subplot_no=(1, 3, 2), show_markers=True, hovertips=False)
_,_,Zmx2h_PL3 = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_lo_GHz', zparam='peak_mx2h@3', fixedparam={'freq_rf_GHz':(freq_filt, 0, 0.1)}, skip_plot=False, fig_no=18, subplot_no=(1, 3, 3), show_markers=True, hovertips=False)

lplot3d(X, Y, Zmx1l_PL3-Zrf1_PL2, xparam='Bias Current (mA)', yparam='Local Oscillator Frequency (GHz)', zparam='Gain (dB)', fig_no=17, subplot_no=(2, 2, 1), show_markers=True, hovertips=False, cmap=color_map, title="$P(f_{RF}-f_{LO})-P(f_{RF})$")
lplot3d(X, Y, Zmx1h_PL3-Zrf1_PL2, xparam='Bias Current (mA)', yparam='Local Oscillator Frequency (GHz)', zparam='Gain (dB)', fig_no=17, subplot_no=(2, 2, 2), show_markers=True, hovertips=False, cmap=color_map, title="$P(f_{RF}+f_{LO})-P(f_{RF})$")
lplot3d(X, Y, Zmx2l_PL3-Zrf1_PL2, xparam='Bias Current (mA)', yparam='Local Oscillator Frequency (GHz)', zparam='Gain (dB)', fig_no=17, subplot_no=(2, 2, 3), show_markers=True, hovertips=False, cmap=color_map, title="$P(f_{RF}-2f_{LO})-P(f_{RF})$")
lplot3d(X, Y, Zmx2h_PL3-Zrf1_PL2, xparam='Bias Current (mA)', yparam='Local Oscillator Frequency (GHz)', zparam='Gain (dB)', fig_no=17, subplot_no=(2, 2, 4), show_markers=True, hovertips=False, cmap=color_map, title="$P(f_{RF}+2f_{LO})-P(f_{RF})$")
plt.suptitle("Calibrated Mixing Gain")

# Calculate remaining parameters - same thing across f_rf
freq_filt = 0.3
X,Y,Zmx1h_PL3rf = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_mx1h@3', fixedparam={'freq_lo_GHz':(freq_filt, 0, 0.01)}, skip_plot=False, fig_no=19, subplot_no=(2, 2, 1), show_markers=True, hovertips=False)
_,_,Zmx1l_PL3rf = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_mx1l@3', fixedparam={'freq_lo_GHz':(freq_filt, 0, 0.01)}, skip_plot=False, fig_no=19, subplot_no=(2, 2, 2), show_markers=True, hovertips=False)
_,_,Zmx2l_PL3rf = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_mx2l@3', fixedparam={'freq_lo_GHz':(freq_filt, 0, 0.01)}, skip_plot=False, fig_no=19, subplot_no=(2, 2, 3), show_markers=True, hovertips=False)
_,_,Zmx2h_PL3rf = dfplot3d(df_mix, xparam='Ibias_mA_bin', yparam='freq_rf_GHz', zparam='peak_mx2h@3', fixedparam={'freq_lo_GHz':(freq_filt, 0, 0.01)}, skip_plot=False, fig_no=19, subplot_no=(2, 2, 4), show_markers=True, hovertips=False)

lplot3d(X, Y, Zmx1l_PL3rf-Zrf1_PL2rf, xparam='Bias Current (mA)', yparam='RF Frequency (GHz)', zparam='Gain (dB)', fig_no=20, subplot_no=(2, 2, 1), show_markers=True, hovertips=False, cmap=color_map, title="$P(f_{RF}-f_{LO})-P(f_{RF})$")
lplot3d(X, Y, Zmx1h_PL3rf-Zrf1_PL2rf, xparam='Bias Current (mA)', yparam='RF Frequency (GHz)', zparam='Gain (dB)', fig_no=20, subplot_no=(2, 2, 2), show_markers=True, hovertips=False, cmap=color_map, title="$P(f_{RF}+f_{LO})-P(f_{RF})$")
lplot3d(X, Y, Zmx2l_PL3rf-Zrf1_PL2rf, xparam='Bias Current (mA)', yparam='RF Frequency (GHz)', zparam='Gain (dB)', fig_no=20, subplot_no=(2, 2, 3), show_markers=True, hovertips=False, cmap=color_map, title="$P(f_{RF}-2f_{LO})-P(f_{RF})$")
lplot3d(X, Y, Zmx2h_PL3rf-Zrf1_PL2rf, xparam='Bias Current (mA)', yparam='RF Frequency (GHz)', zparam='Gain (dB)', fig_no=20, subplot_no=(2, 2, 4), show_markers=True, hovertips=False, cmap=color_map, title="$P(f_{RF}+2f_{LO})-P(f_{RF})$")
plt.suptitle("Calibrated Mixing Gain")


plt.show()