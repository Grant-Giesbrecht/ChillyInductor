import time
import numpy as np
import h5py
import json
from sys import platform
import os
import re
import fnmatch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from colorama import Fore, Style

def spectrum_peak(freqs, pwr, f_target, num_points:int=5):
	''' Returns the power at a given frequency. Returns np.nan if invalid '''
	
	# Check bounds
	freqs = np.asarray(freqs)
	if f_target < np.min(freqs) or f_target > np.max(freqs):
		return np.nan
	
	# Get index of closest frequency
	idx = (np.abs(freqs - f_target)).argmin()
	
	# Return Max Power
	try:
		return max(pwr[(idx-num_points//2):(idx-num_points//2+num_points)])
	except:
		return np.nan

def dBm2W(x:float):
	mw = 10**(x/10)
	w = mw/1000
	return w

def W2dBm(x:float):
	mw = x*1000
	dBm = 10*np.log10(mw)
	return dBm

def spectrum_sum(freqs, pwr, f_min:float=None, f_max:float=None):
	''' Returns the total power in a spectrum. '''
	
	all_bws = []
	pwr_sum = 0
	
	if f_min is not None or f_max is not None:
		print(f"{Fore.RED}This hasn't been implemented!{Style.RESET_ALL} Just need to filter pwr in the for loop statement and add the starting index to idx.")
		return None
	
	# Scan over spectrum
	for idx, s_pt in enumerate(pwr):
		
		# Try to find lower freq delta
		bw_list = []
		try:
			bw_new = freqs[idx]-freqs[idx-1]
			if bw_new == 0:
				bw_new = freqs[idx-1]-freqs[idx-2]
			bw_list.append(np.abs(bw_new))
		except:
			pass
		
		# Try to find upper freq delta
		try:
			bw_new = freqs[idx+1]-freqs[idx]
			if bw_new == 0:
				bw_new = freqs[idx+2]-freqs[idx+1]
			bw_list.append(np.abs(bw_new))
		except:
			pass
		
		# Get minimum BW change
		bw = min(bw_list)
		
		# If BW is large, skip it!
		if bw > 10e3:
			continue
		
		# Add to BW list
		all_bws.append(bw)
		
		# Add to power sum
		pwr_sum += dBm2W(s_pt)*bw
	
	# Return sum
	return W2dBm(pwr_sum)

def subplot_size(N:int):
	# Thanks Copilot
	
	desired_aspect_ratio = 16 / 9
	cols = int(np.sqrt(N * desired_aspect_ratio))
	rows = int(np.ceil(N / cols))
	
	print(f"N = {N}, cols={cols}, rows={rows}")
	
	return (rows, cols)

def plot_drive_conditions(df:pd.DataFrame, fig_no:int=1):
	
	# Find number of numeric columns
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	newdf = df.select_dtypes(include=numerics)
	
	# Get subplot count
	n_col = len(newdf.columns)
	spdim = subplot_size(n_col)
	sp_row = spdim[0]
	sp_col = spdim[1]
	
	# Plot each parameter
	plt.figure(fig_no)
	plt.tight_layout()
	for col_idx in range(n_col):
		
		plt.subplot(sp_row, sp_col, col_idx+1)
		
		plt.plot(newdf.iloc[:,col_idx], linewidth=1, linestyle=':', marker='.', markersize=3)
		plt.title(newdf.columns[col_idx])
		plt.grid()
	

def plot_spectrum(waveform_f_Hz, waveform_s_dBm, waveform_rbw_Hz, rbw_threshold_Hz=5e3, linestyle=':', marker='.', fig_no=1, f_rf=None, f_lo=None, autoshow:bool=False, print_conditions:bool=False):
	''' Function w functionality of AS2 '''
	
	# Calcuate indecies
	fine_idx = waveform_rbw_Hz<=rbw_threshold_Hz
	coarse_idx = waveform_rbw_Hz>rbw_threshold_Hz
	
	xlow = min(waveform_f_Hz/1e9)
	xhigh = max(waveform_f_Hz/1e9)
	
	plt.figure(fig_no)
	plt.tight_layout()
	plt.subplot(2, 1, 1)
	plt.scatter(waveform_f_Hz[fine_idx]/1e9, waveform_s_dBm[fine_idx], marker='.')
	plt.grid(True)
	plt.xlabel("Frequency (GHz)")
	plt.ylabel("Power (dBm)")
	plt.title("Fine Sweep")
	plt.xlim(xlow, xhigh)
	
	plt.subplot(2, 1, 2)
	plt.scatter(waveform_f_Hz[coarse_idx]/1e9, waveform_s_dBm[coarse_idx], marker='.')
	plt.grid(True)
	plt.xlabel("Frequency (GHz)")
	plt.ylabel("Power (dBm)")
	plt.title("Coarse Sweep")
	plt.xlim(xlow, xhigh)
	
	box_width = (xhigh-xlow)/100
	rf_c = (0, 0.7, 0)
	mix_c = (0.7, 0, 0.7)
	lo_c = (0.7, 0, 0)
	alp = [0.25, 0.15, 0.075]
	
	
	# Draw harmonics
	if (f_rf is not None) and (f_lo is not None):
		
		def add_box(box_width, center, alp, color):
			ybounds = plt.ylim()
			
			r = Rectangle((center-box_width/2, ybounds[0]), box_width, ybounds[1]-ybounds[0], facecolor=color, alpha=alp)
			
			ax = plt.gca()
			ax.add_patch(r)
			
		for pltN in [1, 2]:
			plt.subplot(2, 1, pltN)
			add_box(box_width, f_rf, alp[0], rf_c)
			add_box(box_width, 2*f_rf, alp[1], rf_c)
			add_box(box_width, f_rf-f_lo, alp[1], mix_c)
			add_box(box_width, f_rf+f_lo, alp[1], mix_c)
			add_box(box_width, f_rf-2*f_lo, alp[2], mix_c)
			add_box(box_width, f_rf+2*f_lo, alp[2], mix_c)
			add_box(box_width, f_lo, alp[0], lo_c)
			add_box(box_width, 2*f_lo, alp[1], lo_c)
			add_box(box_width, 3*f_lo, alp[2], lo_c)
	
	if autoshow:
		plt.show()


def plot_spectrum_df(df_sa, df_cond, index, rbw_threshold_Hz=5e3, linestyle=':', marker='.', fig_no=1, autoshow:bool=False, print_conditions:bool=False):
	''' Plots a spectrum and highlights the mixing products. Accepts a pandas dataframe as input. '''
	
	if print_conditions:
		c1 = Fore.CYAN
		c2 = Fore.LIGHTRED_EX
		print(f"{Fore.YELLOW}Printing conditions for FIG-{fig_no}:{Style.RESET_ALL}")
		
		if df_cond.rf_enabled.iloc[index]:
			print(f"\tRF = (Enabled: {c1}{df_cond.rf_enabled.iloc[index]}{Style.RESET_ALL}), {c1}{df_cond.power_rf_dBm.iloc[index]}{Style.RESET_ALL} dBm @ {c1}{df_cond.freq_rf_GHz.iloc[index]}{Style.RESET_ALL} GHz")
		else:
			print(f"\tRF = (Enabled: {c2}{df_cond.rf_enabled.iloc[index]}{Style.RESET_ALL})")
		if df_cond.lo_enabled.iloc[index]:
			print(f"\tLO = (Enabled: {c1}{df_cond.lo_enabled.iloc[index]}{Style.RESET_ALL}), {c1}{df_cond.power_lo_dBm.iloc[index]}{Style.RESET_ALL} dBm @ {c1}{df_cond.freq_lo_GHz.iloc[index]}{Style.RESET_ALL} GHz")
		else:
			print(f"\tLO = (Enabled: {c2}{df_cond.lo_enabled.iloc[index]}{Style.RESET_ALL})")
		
		print(f"\tPower Meter = {c1}{df_cond.powermeter_dBm.iloc[index]}{Style.RESET_ALL}")
		# print(f"\t = {c1}{df_cond.}{Style.RESET_ALL}")
		# print(f"\t = {c1}{df_cond.}{Style.RESET_ALL}")
	
	
	plot_spectrum(df_sa['wav_f_Hz'].iloc[index], df_sa['wav_s_dBm'].iloc[index], df_sa['wav_rbw_Hz'].iloc[index], f_rf=df_cond['freq_rf_GHz'].iloc[index], f_lo=df_cond['freq_lo_GHz'].iloc[index], rbw_threshold_Hz=rbw_threshold_Hz, linestyle=linestyle, marker=marker, fig_no=fig_no, autoshow=autoshow)

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

def pad_ragged_lists(ragged:list, max_length:int=None):
	
	if max_length is None:
		max_length = max(len(sublist) for sublist in ragged)
	
	# Initialize a new list of lists with NaN values
	padded_list = [[np.nan] * max_length for _ in range(len(ragged))]
	
	# Copy values from the original list of lists to the new padded list
	for i, sublist in enumerate(ragged):
		padded_list[i][:len(sublist)] = sublist
	
	return padded_list

def reform_dictlist(data:list):
	''' Takes a list of dictionaries, each containing exactly the same keys,
	with consistent datatypes between various dictionaries' matching keys (ie. 
	all keys 'freq' have type float, but key 'name' can all have a different
	type like str.) and reforms the data into a dictionary of lists.
	'''
	# Initialize newform
	newform = {}
	for k in data[0].keys():
		newform[k] = []
	
	# Append values to each key
	for dp in data:
		for k in dp.keys():
			newform[k].append(dp[k])
	
	# Make 2D lists non-jagged
	for k in newform.keys():
		
		# Check if item is 2D list
		if (type(newform[k]) == list) and (len(newform[k]) > 0) and (type(newform[k][0]) == list):
			
			# CHeck if all same length
			lengths = np.array([len(sublist) for sublist in newform[k]])
			max_len = max(lengths)
			
			# Skip item if all are same length
			if all(lengths == max_len):
				continue
			# Otherwise reform
			
			newform[k] = pad_ragged_lists(newform[k], max_len)
	
	return newform

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


def write_level(fh:h5py.File, level_data:dict):
	''' Writes a dictionary to the hdf file.
	 
	Recursive function used by  '''
	
	# Scan over each directory of root-data
	for k, v in level_data.items():
		
		# If value is a dictionary, this key represents a directory
		if type(v) == dict:
			
			# Create a new group
			fh.create_group(k)
			
			# Write the dictionary to the group
			write_level(fh[k], v)
				
		else: # Otherwise try to write this datatype (ex. list of floats)
			
			# Write value as a dataset
			fh.create_dataset(k, data=v)

def save_hdf(root_data:dict, save_file:str, use_json_backup:bool=True) -> bool:
	''' Writes a dictionary to an HDF file per the rules used by 'write_level()'. 
	
	* If the value of a key in another dictionary, the key is made a group (directory).
	* If the value of the key is anything other than a dictionary, it assumes
	  it can be saved to HDF (such as a list of floats), and saves it as a dataset (variable).
	
	'''
	
	# Start timer
	t0 = time.time()
	
	# Open HDF
	hdf_successful = True
	exception_str = ""
	
	# Recursively write HDF file
	with h5py.File(save_file, 'w') as fh:
		
		# Try to write dictionary
		try:
			write_level(fh, root_data)
		except Exception as e:
			hdf_successful = False
			exception_str = f"{e}"
	
	# Check success condition
	if hdf_successful:
		print(f"Wrote file in {time.time()-t0} sec.")
		
		return True
	else:
		print(f"Failed to write HDF file! ({exception_str})")
		
		# Write JSON as a backup if requested
		if use_json_backup:
			
			# Add JSON extension
			save_file_json = save_file[:-3]+".json"
			
			# Open and save JSON file
			try:
				with open(save_file_json, "w") as outfile:
					outfile.write(json.dumps(root_data, indent=4))
			except Exception as e:
				print(f"Failed to write JSON backup: ({e}).")
				return False
		
		return True

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
		fh.create_group("info")
		
		# Add calibration data if specified
		if 'calibration_data' in json_data:
			fh.create_group('calibration_data')
			
			# Write each paramter
			for k in json_data['calibration_data'].keys():
				fh['calibration_data'].create_dataset(k, json_data['calibration_data'][k])
		
		# Add metadata to 'conditions' group
		fh['info'].create_dataset('source_script', data=json_data['source_script'])
		fh['info'].create_dataset('conf_json', data=json.dumps(json_data['configuration']))
		fh['info'].create_dataset('conf_json', data=json.dumps(json_data['configuration']))
		
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