import time
import numpy as np
import h5py
import json
from sys import platform
import os
import re
import fnmatch

def wildcard_path(base_path:str, partial:str):
	''' Given a partial directory name, like Fold*r, returns
	the full path including that directory. Returns None if
	zero or more than one path matched. '''
	
	if base_path is None:
		return None
	
	# Get contents of base path
	try:
		root, dirs, files = next(os.walk(base_path))
	except Exception as e:
		print(f"Error finding directory contents for '{base_path}'. ({e})")
		return None
		
	# Wildcard compare
	match_list = fnmatch.filter(dirs, partial)
	
	if len(match_list) == 1:
		return os.path.join(base_path, match_list[0])
	else:
		return None


def get_datadir_path(rp:int, smc:str, check_drive_letters:list=None):
	''' Returns the path to the data directory for RP-{rp}, and SMC-{smc}.
	
	Return None if can't find path.
	'''
	
	# Check OS
	if platform == "linux" or platform == "linux2":
		is_unix = True
	elif platform == "darwin":
		is_unix = True
	elif platform == "win32":
		is_unix = False
	
	 # /Volumes/M6\ T7S/ARC0\ PhD\ Data/RP-22\ Lk\ Dil\ Fridge\ 2024/Data/SMC-A\ Downconversion
	
	# Find drive
	if is_unix:
		
		path = os.path.join('/', 'Volumes', 'M6 T7S', 'ARC0 PhD Data')
		path = wildcard_path(path, f"RP-{rp}*")
		path = wildcard_path(path, f"Data")
		path = wildcard_path(path, f"SMC-{smc}*")
		
	else:
		
		# List letters to check
		if check_drive_letters is None:
			check_drive_letters = ['D:\\', 'E:\\', 'F:\\', 'G:\\', 'H:\\', 'I:\\', 'J:\\', 'K:\\', 'L:\\']
		
		# Find drive letter
		drive_letter = None
		for cdl in check_drive_letters:
			try_path = os.path.join(cdl, 'ARC0 PhD Data')
			if os.path.exists(try_path):
				drive_letter = cdl
				break
			else:
				print(f"Path {try_path} doesn't exist.")
		# Quit if can't find drive
		if drive_letter is None:
			return None
		
		# Join paths
		path = os.path.join(drive_letter, 'ARC0 PhD Data')
		path = wildcard_path(path, f"RP-{rp}*")
		path = wildcard_path(path, f"Data")
		path = wildcard_path(path, f"SMC-{smc}*")
	
	return path

def calc_sa_conditions(sa_conf, f_rf:float, f_lo:float, print_error:bool=False, remove_duplicates:bool=True) -> list:
	''' Generates a list of dictionsaries with keys:
		rbw: Resolution bandwidth in Hz
		f_start: Start frequency in Hz
		f_end: End frequency in Hz
	for a given set of spectrum analyzer configution blocks. Can assign f_rf to None if you want
	to skip it's harmonics, and f_lo to None to skip its harmonics. Note this will also void
	all mixing products.
	
	--------------------------------------------------------
	Configuration example block:
	"spectrum_analyzer":{
		[
			"mode": "dynamic",
			"span_Hz": 30e3,
			"RBW_Hz": 1e3,
			"mixing_products_order": 2,
			"lo_harmonics": 5,
			"rf_harmonics": 2
		],
		[
			"mode": "fixed",
			"freq_start_Hz": 100e6,
			"freq_end_Hz": 6e9,
			"RBW_Hz": 100e3
		]
	}
	
	Returns None if error, otherwise returns list of dictionaries
	'''
	
	def remove_duplicate_points(points:list):
		# Thanks microsoft Copilot!
		
		seen = set()
		unique_dicts = []
		
		for d in points:
			# Convert the dictionary to a frozenset to make it hashable
			frozen_d = frozenset(d.items())
			if frozen_d not in seen:
				seen.add(frozen_d)
				unique_dicts.append(d)
		return unique_dicts
			
	
	# If sa_conf input is a list, recursively call each
	if type(sa_conf) is list:
				
		# Iterate over each element of list
		sacl = None
		for sacb in sa_conf:
			if sacl is None:
				sacl = calc_sa_conditions(sacb, f_rf, f_lo, print_error=print_error, remove_duplicates=remove_duplicates)
			else:
				sacl = sacl + calc_sa_conditions(sacb, f_rf, f_lo, print_error=print_error, remove_duplicates=remove_duplicates)
		
		# Remove duplicates
		if remove_duplicates:
			sacl = remove_duplicate_points(sacl)
		
		return sacl
	
	# Otherwise generate the dictionary list from this block
	elif type(sa_conf) is dict:
		
		# Generate list of measurement regions
		sac = []
		
		# Validate conf
		if 'mode' not in sa_conf.keys():
			if print_error:
				print(f"Missing key 'mode'.")
			return None
		
		# Split into modes
		if sa_conf['mode'].lower() == "dynamic":
			
			# Validate conf
			test_key_list = ['span_Hz', 'RBW_Hz', 'mixing_products_order', 'lo_harmonics', 'rf_harmonics']
			if not all(test_key in sa_conf for test_key in test_key_list):
				if print_error:
					print(f"Missing keys")
				return None
			
			# Interpret conf
			try:
				lo_harmonics = int(sa_conf['lo_harmonics'])
				rf_harmonics = int(sa_conf['rf_harmonics'])
				mixing_products_order = int(sa_conf['mixing_products_order'])
				RBW_Hz = int(sa_conf['RBW_Hz'])
				span_Hz = int(sa_conf['span_Hz'])
			except Exception as e:
				if print_error:
					print(f"Invalid data in configuration block ({e}).")
				return None
			
			# Create conditions for each LO harmonics
			if f_lo is not None:
				for i in range(1, 1+lo_harmonics):
					
					# Get frequencies
					f_center = f_lo*i
					f_start = f_center - span_Hz/2
					f_end = f_start + span_Hz
					
					# Create conf dictionary
					cd = {'rbw':RBW_Hz, 'f_start':f_start, 'f_end':f_end}
					
					# Add to list
					sac.append(cd)
			
			# Create conditions for each RF harmonics
			if f_rf is not None:
				for i in range(1, 1+rf_harmonics):
					
					# Get frequencies
					f_center = f_rf*i
					f_start = f_center - span_Hz/2
					f_end = f_start + span_Hz
					
					# Create conf dictionary
					cd = {'rbw':RBW_Hz, 'f_start':f_start, 'f_end':f_end}
					
					# Add to list
					sac.append(cd)
			
			## Create conditions for mixing products
			if f_lo is not None and f_rf is not None:
				
				# Get center frequencies
				cfl = []
				for i in range(1, 1+mixing_products_order):
					cfl.append(f_rf + i*f_lo)
					cfl.append(f_rf - i*f_lo)
				
				## Create conditions for mixing products
				# Generate conditions
				for f_center in cfl:
					
					# Get frequencies
					f_start = f_center - span_Hz/2
					f_end = f_start + span_Hz
					
					# Create conf dictionary
					cd = {'rbw':RBW_Hz, 'f_start':f_start, 'f_end':f_end}
					
					# Add to list
					sac.append(cd)
				
		else:
			
			# Validate conf
			test_key_list = ['freq_start_Hz', 'RBW_Hz', 'freq_end_Hz']
			if not all(test_key in sa_conf for test_key in test_key_list):
				if print_error:
					print(f"Missing keys")
				return None
			
			# Interpret conf
			try:
				freq_start_Hz = int(sa_conf['freq_start_Hz'])
				freq_end_Hz = int(sa_conf['freq_end_Hz'])
				RBW_Hz = int(sa_conf['RBW_Hz'])
			except Exception as e:
				if print_error:
					print(f"Invalid data in configuration block ({e}).")
				return None
			
			# Generate condition
			cd = {'rbw':RBW_Hz, 'f_start':freq_start_Hz, 'f_end':freq_end_Hz}
			
			# Add to list
			sac.append(cd)
		
		# Remove duplicates
		if remove_duplicates:
			sac = remove_duplicate_points(sac)
		
		# Return spectrum analyzer condition list
		return sac
		
	# Otherwise invalid type
	else:
		if print_error:
			print(f"Invalid data type")
		return None

def dict_to_hdf5(json_data:dict, save_file) -> bool:
	''' Converts a dict from MS1-style datasets to an HDF5 file.
	'''
	
	##---------------------------------------------------
	# Collect all JSON data into lists
	
	t_gather_0 = time.time()
	
	# Initialize arrays for speed
	N = len(json_data['dataset'])
	freq_rf_GHz = np.zeros(N)
	freq_lo_GHz = np.zeros(N)
	power_LO_dBm = np.zeros(N)
	power_RF_dBm = np.zeros(N)
	rf_enabled = np.zeros(N)
	lo_enabled = np.zeros(N)
	coupled_pwr_dBm = np.zeros(N)
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
		rf_enabled[idx] = dp['rf_enabled']
		lo_enabled[idx] = dp['lo_enabled']
		coupled_pwr_dBm[idx] = dp['coupled_power_meas_dBm']
		
		local_len = len(dp['waveform_f_Hz'])
		waveform_f_Hz[idx][0:local_len] = dp['waveform_f_Hz']
		waveform_s_dBm[idx][0:local_len] = dp['waveform_s_dBm']
		waveform_rbw_Hz[idx][0:local_len] = dp['waveform_rbw_Hz']

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
		
		fh['dataset'].create_dataset('rf_enabled', data=rf_enabled)
		fh['dataset'].create_dataset('lo_enabled', data=lo_enabled)
		fh['dataset'].create_dataset('coupled_power_dBm', data=coupled_pwr_dBm)
	
	t_hdf = time.time()-t_hdf_0
	print(f"Wrote HDF5 file in {t_hdf} sec.")