import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from dataclasses import dataclass
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
import sys
import os
import pandas as pd

#------------------------------------------------------------
# Import Data
datapath = get_datadir_path(rp=22, smc='A')
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
# filename = "dMS2_t3_04June2024_r1.hdf"
# filename = "dMS2_t4_05June2024_r1.hdf"
filename = "dMS2_t3_05June2024_r2.hdf"

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
			 'power_lo_dBm': fh[READ_MODE]['power_LO_dBm']
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
	
	cf_cond = pd.DataFrame(cf_cond_dict)
	cf_sa = pd.DataFrame(cf_sa_dict)
	
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
	
	print(f"{Fore.YELLOW}DF_COND:{Style.RESET_ALL}")
	print(df_cond)
	print(f"{Fore.YELLOW}DF_SA:{Style.RESET_ALL}")
	print(df_sa)

# Plot drive conditions
plot_drive_conditions(df_cond, fig_no=1)

# Plot spectrum points
plot_spectrum_df(df_sa, df_cond, index=5, fig_no=2, print_conditions=True)
plot_spectrum_df(df_sa, df_cond, index=70, autoshow=False, fig_no=3, print_conditions=True)

##===========================================================================
# Whats the best way to calculate power at the RF and LO tones? 
#
# Let's start by looking just at the RF tone...

##.......................................
# First we'll see how much power is measured at the SA at each tone

idx = 10
p1 = spectrum_peak(df_sa.wav_f_Hz.iloc[idx], df_sa.wav_s_dBm.iloc[idx], df_cond.freq_rf_GHz.iloc[idx]*1e9)
p2 = spectrum_peak(df_sa.wav_f_Hz.iloc[idx], df_sa.wav_s_dBm.iloc[idx], df_cond.freq_rf_GHz.iloc[idx]*1e9*2)
p3 = spectrum_peak(df_sa.wav_f_Hz.iloc[idx], df_sa.wav_s_dBm.iloc[idx], df_cond.freq_rf_GHz.iloc[idx]*1e9*3)
plot_spectrum_df(df_sa, df_cond, index=idx, autoshow=False, fig_no=4, print_conditions=True)

print(f"Spectrum analyzer RF harmonic powers:")
print(f"p1={p1}, p2={p2}, p3={p3}")

##.......................................
# How about the power meter?

print(f"Power meter reading: {df_cond.powermeter_dBm.iloc[idx]}")
print(f"Set powers: RF={df_cond.power_rf_dBm.iloc[0]} dBm, LO={df_cond.power_lo_dBm.iloc[0]} dBm")

pwr_sum = spectrum_sum(df_sa.wav_f_Hz.iloc[idx], df_sa.wav_s_dBm.iloc[idx])
print(f"Total spectral power: {pwr_sum} dBm")



#TODO: Once the power cal is done, trying to play with this a little more and make sure it all 
# makes sense is a good idea!

##===========================================================================
# How about a rudimentary estimate for mixing efficiency?

@dataclass
class PowerSpectrum:
	
	rf1:float = None
	rf2:float = None
	rf3:float = None
	lo1:float = None
	lo2:float = None
	lo3:float = None
	mx1l:float = None
	mx1h:float = None
	mx2l:float = None
	mx2h:float = None

def calc_mixing_data_single(df_cond:pd.Series, df_sa:pd.Series) -> PowerSpectrum:
	
	nps = PowerSpectrum()
	
	frf = df_cond.freq_rf_GHz*1e9
	flo = df_cond.freq_lo_GHz*1e9
	
	nps.rf1 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf)
	nps.rf2 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf*2)
	nps.rf3 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf*3)
	
	nps.lo1 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, flo)
	nps.lo2 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, flo*2)
	nps.lo3 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, flo*3)
	
	nps.mx1l = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf-flo)
	nps.mx1h = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf+flo)
	nps.mx2l = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf-2*flo)
	nps.mx2h = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf+2*flo)
	
	return nps

def calc_mixing_data(df_cond:pd.DataFrame, df_sa:pd.DataFrame) -> pd.DataFrame:
	''' Processes input DataFrame and returns a dataframe with peaks, mixing products, etc.'''
	
	data_block = []
	
	# Assign row one at a time
	for index, row in df_cond.iterrows():
		
		# Get mixing products for this row
		nps = calc_mixing_data_single(df_cond.iloc[index, :], df_sa.iloc[index, :])
		
		nps_list = [nps.rf1, nps.rf2, nps.rf3, nps.lo1, nps.lo2, nps.lo3, nps.mx1l, nps.mx1h, nps.mx2l, nps.mx2h]
		
		# Append to data block
		data_block.append(nps_list)
	
	df_out_peaks = pd.DataFrame(data_block, columns=['peak_rf1', 'peak_rf2', 'peak_rf3', 'peak_lo1', 'peak_lo2', 'peak_lo3', 'peak_mx1l', 'peak_mx1h', 'peak_mx2l', 'peak_mx2h'])
	
	# Merge with df_cond and df_sa
	df_out_meister = pd.merge(df_cond, df_sa, left_index=True, right_index=True, how='outer')
	df_out_meister = pd.merge(df_out_meister, df_out_peaks, left_index=True, right_index=True, how='outer')
	
	# Find change!
	
	# Calibration things??
	
	# Return a dataframe
	
	return df_out_meister

# def dfplot(df_src, xparam:str, yparam:str, zparam:str):
# 	''' Accepts a dataframe as input, and returns a tuple with (X,Y,Z)
# 	2D lists to plot using contourf(). '''
	
# 	# Get unique axis values
# 	unique_x = sorted(list(set(df_src[xparam])))
# 	unique_y = sorted(list(set(df_src[yparam])))
	
# 	# Generate X and Y variables
# 	X, Y = np.meshgrid(unique_y, unique_x)
	
# 	# 

def dfplot(df, xparam:str, yparam:str, zparam:str, skip_plot:bool=False, fig_no:int=1):
	''' Accepts a dataframe as input, and returns a tuple with (X,Y,Z)
# 	2D lists to plot using contourf(). '''
	
	# Thanks copilot!
	
	# Ensure the DataFrame contains the necessary columns
	if not {xparam, yparam, zparam}.issubset(df.columns):
		raise ValueError("DataFrame missing specified columns")

	# Create a grid of points
	x = np.linspace(df[xparam].min(), df[xparam].max(), len(df[xparam].unique()))
	y = np.linspace(df[yparam].min(), df[yparam].max(), len(df[yparam].unique()))
	X, Y = np.meshgrid(x, y)

	# Reshape zparam to match the grid shape
	Z = df.pivot_table(index=yparam, columns=xparam, values=zparam).values
	
	# Skip plot if asked
	if not skip_plot:
		
		# Create the contourf plot
		plt.figure(fig_no)
		plt.contourf(X, Y, Z, levels=15, cmap='viridis')
		plt.colorbar()  # Add a colorbar to a plot
		plt.xlabel(xparam)
		plt.ylabel(yparam)
		plt.show()
	
	# Return data values
	return (X, Y, Z)

# Calculate mixing data DF
df_mix = calc_mixing_data(df_cond, df_sa)

# Generate a Nice N' Juicy plot
mdt = dfplot(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='powermeter_dBm', fig_no=5)
mdt = dfplot(df_mix, xparam='freq_rf_GHz', yparam='freq_lo_GHz', zparam='peak_rf1', fig_no=6)

# X = mdt[0]
# Y = mdt[1]

plt.show()