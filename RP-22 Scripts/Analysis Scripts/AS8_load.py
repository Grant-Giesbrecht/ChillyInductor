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

@dataclass
class PowerSpectrum:
	
	p_fund:float = None
	p_2h:float = None
	p_3h:float = None
	p_pm:float = None

#------------------------------------------------------------
# Import Data
datapath = get_datadir_path(rp=22, smc='A')
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
# filename = "dMS2_t3_04June2024_r1.hdf"
filename = "dMS2_t4_05June2024_r1.hdf"

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
plot_spectrum_df(df_sa, df_cond, index=5, autoshow=False, fig_no=2, print_conditions=True)
plot_spectrum_df(df_sa, df_cond, index=70, autoshow=False, fig_no=3, print_conditions=True)

# Show plots
plt.show()
