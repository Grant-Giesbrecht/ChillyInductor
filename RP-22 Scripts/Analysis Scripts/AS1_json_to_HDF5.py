import numpy as np
import h5py
import json
import time

#---------------------------------------------------
# Read JSON file
t_json_0 = time.time()
with open("E:\ARC0 PhD Data\RP-22 Lk Dil Fridge 2024\Data\SMC-A Downconversion v1\dMS1_28May2024_DC1V0_r1.json", 'r') as fh:
	json_data=json.load(fh)
t_json = time.time() - t_json_0
print(f"Read JSON file in {t_json} sec.")

save_file = "E:\ARC0 PhD Data\RP-22 Lk Dil Fridge 2024\Data\SMC-A Downconversion v1\dMS1_28May2024_DC1V0_r1.hdf"

##---------------------------------------------------
# Collect all JSON data into lists

t_gather_0 = time.time()

# Initialize arrays for speed
N = len(json_data['dataset'])
freq_rf_GHz = np.zeros(N)
freq_lo_GHz = np.zeros(N)
power_LO_dBm = np.zeros(N)
power_RF_dBm = np.zeros(N)
max_len = 0

# Calculate size of 2D arrays
for idx, dp in enumerate(json_data['dataset']):
	max_len = max(max_len, len(dp['waveform_f_Hz']))

# Allocate 2D arrays
waveform_f_Hz = np.full([N, max_len], np.nan)
waveform_s_dBm = np.full([N, max_len], np.nan)
waveform_rbw_Hz = np.full([N, max_len], np.nan)

# Scan over JSON second time
for idx, dp in enumerate(json_data['dataset']):
	
	# Update parameters
	freq_rf_GHz[idx] = dp['freq_rf_GHz']
	freq_lo_GHz[idx] = dp['freq_lo_GHz']
	power_LO_dBm[idx] = dp['power_LO_dBm']
	power_RF_dBm[idx] = dp['power_RF_dBm']
	
	waveform_f_Hz[idx][:] = dp['waveform_f_Hz']
	waveform_s_dBm[idx][:] = dp['waveform_s_dBm']
	waveform_rbw_Hz[idx][:] = dp['waveform_rbw_Hz']

t_gather = time.time()-t_gather_0
print(f"Gather JSON data into lists in {t_gather} sec.")

##---------------------------------------------------
# Convert JSON file to HDF5 file
t_hdf_0 = time.time()
with h5py.File(save_file, 'w') as fh:
	
	# Create two root groups
	fh.create_group("dataset")
	fh.create_group("conditions")
	
	# Add metadata to 'conditions' group
	fh['conditions'].create_dataset('source_script', data=json_data['source_script'])
	fh['conditions'].create_dataset('conf_json', data=json.dumps(json_data['configuration']))
	
	# Add data to 'dataset' group
	fh['dataset'].create_dataset('freq_rf_GHz', data=freq_rf_GHz)
	fh['dataset'].create_dataset('freq_lo_GHz', data=freq_lo_GHz)
	fh['dataset'].create_dataset('power_LO_dBm', data=power_LO_dBm)
	fh['dataset'].create_dataset('power_RF_dBm', data=power_RF_dBm)
	fh['dataset'].create_dataset('waveform_f_Hz', data=waveform_f_Hz)
	fh['dataset'].create_dataset('waveform_s_dBm', data=waveform_s_dBm)
	fh['dataset'].create_dataset('waveform_rbw_Hz', data=waveform_rbw_Hz)

t_hdf = time.time()-t_hdf_0
print(f"Wrote HDF5 file in {t_hdf} sec.")

##---------------------------------------------------
# Read all of the HDF5 file for a test

t_hdfr_0 = time.time()
with h5py.File(save_file, 'r') as fh:
	
	source_script = fh['conditions']['source_script'][()]
	conf_json = fh['conditions']['conf_json'][()]
	
	a = fh['dataset']['freq_rf_GHz'][()]
	b = fh['dataset']['freq_lo_GHz'][()]
	c = fh['dataset']['power_RF_dBm'][()]
	d = fh['dataset']['power_LO_dBm'][()]
	e = fh['dataset']['waveform_f_Hz'][()]
	f = fh['dataset']['waveform_s_dBm'][()]
	g = fh['dataset']['waveform_rbw_Hz'][()]

t_hdfr = time.time()-t_hdfr_0
print(f"Read entire HDF file in {t_hdfr} sec.")
