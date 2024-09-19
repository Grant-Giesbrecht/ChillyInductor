'''
First dataset taken with MP5*.py didn't save setpoint in main dataset. This finds
the setpoint in the aux-data and lines it up with the main dataset.
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
from jarnsaxa import hdf_to_dict, dict_to_hdf

#------------------------------------------------------------
# Import Data

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm'])

# if datapath is None:
# 	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
# 	sys.exit()
# else:
# 	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

# # filename = "RP22B_MP3_t1_31July2024_R4C4T1_r1_autosave.hdf"
# filename = "RP22B_MP3_t1_1Aug2024_R4C4T1_r1.hdf"

# analysis_file = os.path.join(datapath, filename)
analysis_file = "/Users/grantgiesbrecht/Downloads/RP22B_MP5_t2_18Sept2024_R4C1T2_r1.hdf"

log = LogPile()

inputdata = hdf_to_dict("/Users/grantgiesbrecht/Downloads/RP22B_MP5_t2_18Sept2024_R4C1T2_r1.hdf")

times_main = inputdata['dataset']['times']
times_main_ds = [datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in times_main]

timestamp_ctl_str = inputdata['aux_dataset']['continuous_temp_logging']['timestamp']
times_aux_ds = [datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in timestamp_ctl_str]

# with h5py.File(analysis_file, 'r') as fh:
#
# 	# Read primary dataset
# 	GROUP = 'dataset'
# 	freq_rf_GHz = fh[GROUP]['freq_rf_GHz'][()]
# 	power_rf_dBm = fh[GROUP]['power_rf_dBm'][()]
#	
# 	waveform_f_Hz = fh[GROUP]['waveform_f_Hz'][()]
# 	waveform_s_dBm = fh[GROUP]['waveform_s_dBm'][()]
# 	waveform_rbw_Hz = fh[GROUP]['waveform_rbw_Hz'][()]
#	
# 	MFLI_V_offset_V = fh[GROUP]['MFLI_V_offset_V'][()]
# 	requested_Idc_mA = fh[GROUP]['requested_Idc_mA'][()]
# 	raw_meas_Vdc_V = fh[GROUP]['raw_meas_Vdc_V'][()]
# 	Idc_mA = fh[GROUP]['Idc_mA'][()]
# 	detect_normal = fh[GROUP]['detect_normal'][()]
#	
# 	temperature_K = fh[GROUP]['temperature_K'][()]
# 	times_str = fh[GROUP]['times'][()]
# 	times = [datetime.datetime.strptime(ts.decode(), '%Y-%m-%d %H:%M:%S.%f') for ts in times_str]
#	
# 	GROUP = 'aux_dataset'
# 	timestamp_ctl_str = fh[GROUP]['continuous_temp_logging']['timestamp'][()]
# 	timestamp_ctl = [datetime.datetime.strptime(ts.decode(), '%Y-%m-%d %H:%M:%S.%f') for ts in timestamp_ctl_str]
# 	temp_sp_ctl = fh[GROUP]['continuous_temp_logging']['temp_setpoint_K'][()]

# Add setpoint variable
inputdata['dataset']['temp_sp_K'] = []

# Find setpoint for each time value
idx_start = 0
for midx, main_time in enumerate(times_main_ds):
	
	found_val = False
	
	# Scan over aux times
	for aidx_, aux_time in enumerate(times_aux_ds[idx_start:]):
		
		# Convert to real index
		aidx = aidx_ + idx_start
		
		# Check if aux time is greater than main
		if aux_time > main_time:
			
			# Record next starting index
			idx_start = aidx-1
			
			# Add value to main dataset
			print(aidx)
			inputdata['dataset']['temp_sp_K'].append(inputdata['aux_dataset']['continuous_temp_logging']['temp_setpoint_K'][aidx-1])
			found_val = True
			break
	
	if not found_val:
		inputdata['dataset']['temp_sp_K'].append(inputdata['aux_dataset']['continuous_temp_logging']['temp_setpoint_K'][-1])

dict_to_hdf(inputdata, "/Users/grantgiesbrecht/Downloads/RP22B_MP5_t2_18Sept2024_R4C1T2_r1_processed.hdf")