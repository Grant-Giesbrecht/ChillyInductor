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
filename = "dMS2_t2_03June2024_r1_autosave.hdf"

analysis_file = os.path.join(datapath, filename)

USE_CONST_SCALE = True

##--------------------------------------------
# Read HDF5 File

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	source_script = fh['conditions']['source_script']
	conf_json = fh['conditions']['conf_json']
	
	coupled_power_dBm = fh['dataset']['coupled_power_dBm']
	rf_enabled = fh['dataset']['rf_enabled']
	lo_enabled = fh['dataset']['lo_enabled']
	f_rf = fh['dataset']['freq_rf_GHz']
	f_lo = fh['dataset']['freq_lo_GHz']
	p_rf = fh['dataset']['power_RF_dBm']
	p_lo = fh['dataset']['power_LO_dBm']
	waveform_f_Hz = fh['dataset']['waveform_f_Hz']
	waveform_s_dBm = fh['dataset']['waveform_s_dBm']
	waveform_rbw_Hz = fh['dataset']['waveform_rbw_Hz']
	
	df_cond_dict = {'powermeter_dBm': fh['dataset']['coupled_power_dBm'],
			 'rf_enabled': fh['dataset']['rf_enabled'],
			 'lo_enabled': fh['dataset']['lo_enabled'],
			 'freq_rf_GHz': fh['dataset']['freq_rf_GHz'],
			 'freq_lo_GHz': fh['dataset']['freq_lo_GHz'],
			 'power_rf_dBm': fh['dataset']['power_RF_dBm'],
			 'power_lo_dBm': fh['dataset']['power_LO_dBm']
			 }
	df_sa_dict = {'wav_f_Hz': list(fh['dataset']['waveform_f_Hz']),
			 'wav_s_dBm': list(fh['dataset']['waveform_s_dBm']),
			 'wav_rbw_Hz': list(fh['dataset']['waveform_rbw_Hz']),
			 }
	
	df_cond = pd.DataFrame(df_cond_dict)
	df_sa = pd.DataFrame(df_sa_dict)
	
	print(f"{Fore.YELLOW}DF_COND:{Style.RESET_ALL}")
	print(df_cond)
	print(f"{Fore.YELLOW}DF_SA:{Style.RESET_ALL}")
	print(df_sa)


