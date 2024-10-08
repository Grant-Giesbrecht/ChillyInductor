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
from dataclasses import dataclass
from mpl_toolkits.mplot3d import axes3d
from pylogfile.base import *
from ganymede import *
import skrf

def visualize_dataset_conditions(file:str):
	
	#------------------------------------------------------------
	# Import Data

	datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm'])
	# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 2 43mm'])
	# datapath = '/Volumes/M5 PERSONAL/data_transfer'
	if datapath is None:
		print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
		return
	else:
		print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

	# filename = "RP22B_MP3_t1_31July2024_R4C4T1_r1_autosave.hdf"
	# filename = "RP22B_MP3_t1_1Aug2024_R4C4T1_r1.hdf"
	filename = "RP22B_MP3_t2_8Aug2024_R4C4T1_r1.hdf"
	# filename = "RP22B_MP3a_t3_19Aug2024_R4C4T2_r1.hdf"
	# filename ="RP22B_MP3a_t2_20Aug2024_R4C4T2_r1_autosave.hdf"

	analysis_file = os.path.join(datapath, filename)

	log = LogPile()

	##--------------------------------------------
	# Read HDF5 File

	print("Loading file contents into memory")

	data = hdf_to_dict(analysis_file)
	
	try:
		conf_struct = data['info']['configuration']
	except:
		return

def rd(x:str, places:int=2):
	return f"{round(x*(10**places))/(10**places)}"

def plot_nan(x, y):
	
	print("TODO: This doesn't work but should be fixed!")
	
	# Example DataFrame (replace with your actual data)
	data = {
		'idx': x,
		'col': y
	}
	df = pd.DataFrame(data)

	# Convert 'idx' to datetime if needed
	df['idx'] = pd.to_datetime(df['idx'])

	# Set 'idx' as the index
	df = df.set_index('idx')

	# Code to find the start and stop index of NaN gaps
	is_nan = df['col'].isna()
	n_groups = is_nan.ne(is_nan.shift()).cumsum()
	gap_list = df[is_nan].groupby(n_groups).aggregate(
		lambda x: (
			x.index[0] + pd.DateOffset(days=-1),
			x.index[-1] + pd.DateOffset(days=+1)
		)
	)["col"].values

	# Create the plot
	plt.plot(df.index, df['col'], marker='o')
	plt.xticks(df.index, rotation=45)

	# Highlight gaps in red
	for gap in gap_list:
		plt.axvspan(gap[0], gap[1], facecolor='r', alpha=0.5)

	plt.grid()
	plt.title("Time Series with Gap Highlights")
	plt.xlabel("Date")
	plt.ylabel("Value")
	plt.show()

def list_search_spectrum_peak_harms(freqs_list:list, pwr_list:list, f_targ_fund_list:list, nharms:int=3, scan_bw_Hz:float=500, harm_scan_points:int=7):
	''' 
	Designed to remedy issues that cropped-up with spectrum_peak_list when the 
	RF signal generator deviated by too much from the target frequency. Searches 
	for the max power at a given frequency, but searches within a range of
	scan_bw_Hz (so bw/2 in either direction). Repeats the spectrum search then
	at each requested harmonic, however after identifying the actual frequency
	of the fundamental, it will conduct a narrower search on the harmonics. 
	
	Parameters:
		freq (list): List of frequencies in Hz, as all source data
		pwr (list): List of powers in dBm, as all source data
		f_targ_fund (float): Fundamental frequency to search for, in Hz
		nharms (int): Number of harmonics to include
		scan_bw_Hz (float): Bandwidth over which to scan for actual fundamental peak in Hz.
		harm_scan_points (int): Number of points to search, centered around fundamental frequency based off of found harmonic.
	
	Returns:
		(Tuple) Index 0, list of lists of detected frequencies for each spectral component. Inner list aligns with f_targ_fund_list; Index 1, list of lists of peak powers for each spectral component. Inner list aligns with f_targ_fund_list. Returns None if invalid (eg. f_targ outside range of frequencies). If one of the harmonics is out of frequency bounds for input data, replaces that value with np.nan
	'''
	
	# Initialize output lists
	freqs_out = []
	powers_out = []
	for harm_no in range(nharms):
		freqs_out.append([])
		powers_out.append([])
	
	# Scan over each target value
	for freqs, pwr, f_targ_fund in zip(freqs_list, pwr_list, f_targ_fund_list):
		
		# freqs_nsort = freqs.copy()
		# freqs.sort()
		
		# if not all(freqs_nsort == freqs):
		# 	print(f"{Fore.RED} NOt sorted!{Style.RESET_ALL}")
		
		# Remove all nan values
		valid_indices = np.where(~np.isnan(freqs) & ~np.isnan(pwr)) # Find indices where neither array has NaN
		freqs = freqs[valid_indices]
		pwr = pwr[valid_indices]
		
		# Check bounds
		freq_max = np.max(freqs)
		freq_min = np.min(freqs)
		freqs = np.asarray(freqs)
		
		# print(f"Target freq: {f_targ_fund}")
	
		if f_targ_fund < freq_min or f_targ_fund > freq_max:
			return None
		
		# Get index of closest frequency
		idx_targ = (np.abs(freqs - f_targ_fund)).argmin() # Find index closest to target harm frequency
		idx_ftlow = (np.abs(freqs - (f_targ_fund-scan_bw_Hz/2))).argmin()
		idx_fthigh = (np.abs(freqs - (f_targ_fund+scan_bw_Hz/2))).argmin()
		
		idx_ftlow = np.min([idx_ftlow, idx_targ-harm_scan_points//2])
		idx_fthigh = np.max([idx_fthigh, idx_targ-harm_scan_points//2+harm_scan_points])
		
		
		# print(f"IDX: {idx_ftlow}:({idx_targ}):{idx_fthigh}")
		# fl = freqs[idx_ftlow]
		# fh = freqs[idx_fthigh]
		# f0 = freqs[idx_targ]
		# print(f"IDX: {fl/1e9}:({f0/1e9}):{fh/1e9} GHz")
		
		# Return Max Power
		try:
			idx_peak = np.argmax(pwr[idx_ftlow:idx_fthigh])+idx_ftlow
			ff_peak = freqs[idx_peak]
			pf_peak = pwr[idx_peak]
		except:
			return None
		
		# Add to lists
		freqs_out[0].append(ff_peak)
		powers_out[0].append(pf_peak)
		
		# print(f"   peak: {ff_peak/1e6} MHz")
		# print(f"   target: {f_targ_fund/1e6} MHz")
		# print(f"   Delta: {ff_peak/1e6 - f_targ_fund/1e6} MHz")
		
		# freq_options = np.array(freqs[idx_ftlow:idx_fthigh])/1e6
		# pwr_options = pwr[idx_ftlow:idx_fthigh]
		# print(f"    Freqs: {freq_options} MHz")
		# print(f"    pwrs: {pwr_options} MHz")
		
		
		# print(f"  -> Found peak at: {ff_peak}")
		
		# Calculate for each harmonic
		for harm_no in range(2, nharms+1):
			
			# print(f"  Searching harmonic: {harm_no}")
			
			# Check for out of bounds
			if ff_peak*harm_no > freq_max or ff_peak*harm_no < freq_min:
				freqs_out.append(np.nan)
				powers_out.append(np.nan)
				continue
			
			# Find and add point
			idx_htarg = (np.abs(freqs - ff_peak*harm_no)).argmin() # Find index closest to target harm frequency
			idx_hmax = np.argmax(pwr[(idx_htarg-harm_scan_points//2):(idx_htarg-harm_scan_points//2+harm_scan_points)]) + (idx_htarg-harm_scan_points//2) # Find local maximum
			
			# Append
			freqs_out[harm_no-1].append(freqs[idx_hmax])
			powers_out[harm_no-1].append(pwr[idx_hmax])
	
	# Convert to numpy arrays
	for harm_no in range(nharms):
		freqs_out[harm_no] = np.array(freqs_out[harm_no])
		powers_out[harm_no] = np.array(powers_out[harm_no])
	
	return (freqs_out, powers_out)

def search_spectrum_peak_harms(freqs:list, pwr:list, f_targ_fund:float, nharms:int=3, scan_bw_Hz:float=500, harm_scan_points:int=7):
	''' 
	Designed to remedy issues that cropped-up with spectrum_peak_list when the 
	RF signal generator deviated by too much from the target frequency. Searches 
	for the max power at a given frequency, but searches within a range of
	scan_bw_Hz (so bw/2 in either direction). Repeats the spectrum search then
	at each requested harmonic, however after identifying the actual frequency
	of the fundamental, it will conduct a narrower search on the harmonics. 
	
	Parameters:
		freq (list): List of frequencies in Hz, as all source data
		pwr (list): List of powers in dBm, as all source data
		f_targ_fund (float): Fundamental frequency to search for, in Hz
		nharms (int): Number of harmonics to include
		scan_bw_Hz (float): Bandwidth over which to scan for actual fundamental peak in Hz.
		harm_scan_points (int): Number of points to search, centered around fundamental frequency based off of found harmonic.
	
	Returns:
		(Tuple) Index 0, list of detected frequencies for each spectral component; Index 1, list of peak powers for each spectral component. Returns None if invalid (eg. f_targ outside range of frequencies). If one of the harmonics is out of frequency bounds for input data, replaces that value with np.nan
	'''
	
	#TODO: Implement
	#TODO: Make it listy in wyvern!
	
	# Remove all nan values
	valid_indices = np.where(~np.isnan(freqs) & ~np.isnan(pwr)) # Find indices where neither array has NaN
	freqs = freqs[valid_indices]
	pwr = pwr[valid_indices]
	
	# Check bounds
	freq_max = np.max(freqs)
	freq_min = np.min(freqs)
	freqs = np.asarray(freqs)
	if f_targ_fund < freq_min or f_targ_fund > freq_max:
		return None
	
	# Get index of closest frequency
	# idx_fftarg = (np.abs(freqs - f_targ_fund)).argmin()
	idx_ftlow = (np.abs(freqs - (f_targ_fund-scan_bw_Hz/2))).argmin()
	idx_fthigh = (np.abs(freqs - (f_targ_fund+scan_bw_Hz/2))).argmin()
	
	# Return Max Power
	try:
		idx_peak = np.argmax(pwr[idx_ftlow:idx_fthigh])
		ff_peak = freqs[idx_peak]
		pf_peak = pwr[idx_peak]
	except:
		return None
	
	# Initialize output lists
	freqs_out = [ff_peak]
	powers_out = [pf_peak]
	
	# Calculate for each harmonic
	for harm_no in range(2, nharms+1):
		
		# Check for out of bounds
		if ff_peak*harm_no > freq_max or ff_peak*harm_no < freq_min:
			freqs_out.append(np.nan)
			powers_out.append(np.nan)
			continue
		
		# Find and add point
		idx_htarg = (np.abs(freqs - ff_peak*harm_no)).argmin() # Find index closest to target harm frequency
		idx_hmax = np.argmax(pwr[(idx_htarg-harm_scan_points//2):(idx_htarg-harm_scan_points//2+harm_scan_points)]) # Find local maximum
		freqs_out.append(freqs[idx_peak])
		powers_out.append(pwr[idx_peak])

	return (freqs_out, powers_out)

def spectrum_peak_list(freqs, pwr, f_target, num_points:int=5):
	''' Returns the power at a given frequency. Returns np.nan if invalid '''
	
	rval = []
	for flist, plist, ftarg in zip(freqs, pwr, f_target):
		val = spectrum_peak(flist, plist, ftarg, num_points=num_points)
		rval.append(float(val))
	return np.array(rval)

def spectrum_peak(freqs, pwr, f_target, num_points:int=5):
	''' Returns the power at a given frequency. Returns np.nan if invalid '''
	
	# Remove all nan values
	valid_indices = np.where(~np.isnan(freqs) & ~np.isnan(pwr)) # Find indices where neither array has NaN
	freqs = freqs[valid_indices]
	pwr = pwr[valid_indices]
	
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
	pwr_sum_mW = 0
	
	if f_min is not None or f_max is not None:
		print(f"{Fore.RED}This hasn't been implemented!{Style.RESET_ALL} Just need to filter pwr in the for loop statement and add the starting index to idx.")
		return None
	
	# Scan over spectrum
	for idx, s_pt in enumerate(pwr):
		
		# Try to find lower freq delta
		mean_pwr = None
		bw_new = None
		try:
			bw_new = np.abs(freqs[idx]-freqs[idx-1])
			mean_pwr = np.mean([dBm2W(pwr[idx]), dBm2W(pwr[idx-1])])
			if bw_new == 0:
				bw_new = np.abs(freqs[idx-1]-freqs[idx-2])
				mean_pwr = np.mean([dBm2W(pwr[idx-1]), dBm2W(pwr[idx-2])])
		except:
			pass
		
		# Try to find upper freq delta
		try:
			bw_new2 = np.abs(freqs[idx+1]-freqs[idx])
			mean_pwr2 = np.mean([dBm2W(pwr[idx+1]), dBm2W(pwr[idx])])
			if bw_new == 0:
				bw_new2 = np.abs(freqs[idx+2]-freqs[idx+1])
				mean_pwr2 = np.mean([dBm2W(pwr[idx+2]), dBm2W(pwr[idx+1])])
		except:
			pass
		
		# Pick smallest gap
		if (bw_new is None) or ((bw_new2 is not None) and (bw_new2 < bw_new)):
			bw = bw_new2
			use_pwr = mean_pwr2
		else:
			bw = bw_new
			use_pwr = mean_pwr
		
		# If BW is large, skip it!
		if bw > 10e3:
			continue
		
		# Add to BW list
		all_bws.append(bw)
		
		# print(f"Mean power:{use_pwr}, BW: {bw}, sum: {use_pwr*bw}")
		
		# Add to power sum
		pwr_sum_mW += use_pwr*(bw*1000)
	
	# Return sum
	return W2dBm(pwr_sum_mW/1e3)

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

def wildcard_path(base_path:str, partial:str, print_errors:bool=False, include_files:bool=True):
	''' Given a partial directory name, like Fold*r, returns
	the full path including that directory. Returns None if
	zero or more than one path matched. '''
	
	if base_path is None:
		return None
	
	# Get contents of base path
	try:
		root, dirs, files = next(os.walk(base_path))
	except Exception as e:
		if print_errors:
			print(f"wildcard_path(): Error finding directory contents for '{base_path}'. ({e})")
		return None
		
	# Wildcard compare
	match_list = fnmatch.filter(dirs, partial)
	
	if include_files and len(match_list) != 1:
		match_list = fnmatch.filter(files, partial)
	
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

def get_general_path(sub_dirs:list=[], check_drive_letters:list=None, unix_vol_name:str=None, dos_drive_letter:str="C:\\", dos_id_folder:bool=False, print_details:bool=False):
	''' Returns the path to the directory. Defaults to main disk ('/' on unix and 'C:\\' on Windows). Can change to 
	external volume by providing the volume name in unix systems, or the drive letter on DOS/Windows. DOS/Windows can
	also identify the drive letter by checking which external drive first (in alphabetical order) contains the first 
	folder specified in sub_dirs. You can set which drive letters are checked, and in which order, by providing a list of
	letters to 'check_drive_letters'.
	
	Returns the matching path, otherwise returns None
	'''
	
	# Check OS
	if platform == "linux" or platform == "linux2":
		is_unix = True
	elif platform == "darwin":
		is_unix = True
	elif platform == "win32":
		is_unix = False
		
	# Find drive
	if is_unix:
		if print_details:
			print(f"get_general_path(): Identified system as UNIX")
		if unix_vol_name is None:
			path = '/'
		else:
			path = os.path.join('/', 'Volumes', {unix_vol_name})
	
	else:
		if print_details:
			print(f"get_general_path(): Identified system as DOS")
		
		# List letters to check
		if check_drive_letters is None:
			check_drive_letters = ['D:\\', 'E:\\', 'F:\\', 'G:\\', 'H:\\', 'I:\\', 'J:\\', 'K:\\', 'L:\\']
		
		if dos_id_folder:
			
			try:
				dos_id_folder = sub_dirs[0]
				sub_dirs = sub_dirs[1:]
				if print_details:
					print(f"get_general_path(): Identified ID folder as {dos_id_folder}.")
			except:
				return None
				
			# Find drive letter
			path = None
			for cdl in check_drive_letters:
				try_path = wildcard_path(cdl, dos_id_folder, print_errors=print_details)
				if try_path is not None:
					if print_details:
						print(f"get_general_path(): Drive letter {cdl} identified as match.")
					path = try_path
					break
				elif print_details:
					print(f"get_general_path(): Drive letter {cdl} did not match conditions.")
			# Return None if none found
			if path is None:
				if print_details:
					print(f"get_general_path(): Failed to find drive with ID folder {dos_id_folder}. Exiting.")
				return None
		else:
			path = "C:\\"
	
	# Add on additional paths
	for ad in sub_dirs:
		path = wildcard_path(path, ad)
	
	return path

def get_datadir_path(rp:int, smc:str, sub_dirs:list=[], check_drive_letters:list=None):
	''' Returns the path to the data directory for RP-{rp}, and SMC-{smc}.
	
	sub_dirs accepts wildcarded directory names to append, in order, after the SMC directory.
	
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
	
	# Add on additional paths
	for ad in sub_dirs:
		
		path = wildcard_path(path, ad)
	
	return path

# def get_path(sub_dirs:list=[], check_drive_letters:list=None):
# 	''' Returns the path to the data directory for RP-{rp}, and SMC-{smc}.
	
# 	sub_dirs accepts wildcarded directory names to append, in order, after the SMC directory.
	
# 	Return None if can't find path.
# 	'''
	
# 	# Check OS
# 	if platform == "linux" or platform == "linux2":
# 		is_unix = True
# 	elif platform == "darwin":
# 		is_unix = True
# 	elif platform == "win32":
# 		is_unix = False
	
# 	 # /Volumes/M6\ T7S/ARC0\ PhD\ Data/RP-22\ Lk\ Dil\ Fridge\ 2024/Data/SMC-A\ Downconversion
	
# 	# Find drive
# 	if is_unix:
		
# 		path = os.path.join('/', 'Volumes', 'M6 T7S', 'ARC0 PhD Data')
# 		path = wildcard_path(path, f"RP-{rp}*")
# 		path = wildcard_path(path, f"Data")
# 		path = wildcard_path(path, f"SMC-{smc}*")
		
# 	else:
		
# 		# List letters to check
# 		if check_drive_letters is None:
# 			check_drive_letters = ['D:\\', 'E:\\', 'F:\\', 'G:\\', 'H:\\', 'I:\\', 'J:\\', 'K:\\', 'L:\\']
		
# 		# Find drive letter
# 		drive_letter = None
# 		for cdl in check_drive_letters:
# 			try_path = os.path.join(cdl, 'ARC0 PhD Data')
# 			if os.path.exists(try_path):
# 				drive_letter = cdl
# 				break
				
# 		# Quit if can't find drive
# 		if drive_letter is None:
# 			return None
		
# 		# Join paths
# 		path = os.path.join(drive_letter, 'ARC0 PhD Data')
# 		path = wildcard_path(path, f"RP-{rp}*")
# 		path = wildcard_path(path, f"Data")
# 		path = wildcard_path(path, f"SMC-{smc}*")
	
# 	# Add on additional paths
# 	for ad in sub_dirs:
		
# 		path = wildcard_path(path, ad)
	
# 	return path

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
	
	print(f"{Fore.RED}This function is deprecated and should be replaced with dict_to_hdf.{Style.RESET_ALL}")
	
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
	
	total:float=None

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

def calc_harm_data_single(df_cond:pd.Series, df_sa:pd.Series) -> PowerSpectrum:
	
	nps = PowerSpectrum()
	
	frf = df_cond.freq_rf_GHz*1e9
	flo = df_cond.freq_lo_GHz*1e9
	
	nps.rf1 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf)
	nps.rf2 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf*2)
	nps.rf3 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, frf*3)
	
	nps.lo1 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, flo)
	nps.lo2 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, flo*2)
	nps.lo3 = spectrum_peak(df_sa.wav_f_Hz, df_sa.wav_s_dBm, flo*3)
	
	nps.total = spectrum_sum(df_sa.wav_f_Hz, df_sa.wav_s_dBm)
	
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

def bin_empirical(data:np.ndarray, jump_size:int=None, jump_scale:int=None, show_work:bool=False, autoscale_fact:float=0.1, autoscale_checks:tuple=(0.05, 0.15)):
	''' If you have a list of collected datapoints that are supposed to be discrete steps,
	but have some natural variation, this function automatically attempts to bin them.
	If it's not sure how to bin the data, it'll shout and yell in red text at you in the
	terminal. YOu can then set 'show_work' to true so it'll show you the intermediate products.
	
	jump_scale allows you to manually set how much the median differential must be scaled for a diff
		to count as a jump.
	jump_size allows you to manually set the size of a jump for a differential.
	autoscale_fact sets how, when automatically binning, the ratio of max/median diff is scaled to determine
		the starting threshold.
	autoscale_checks set the autoscale factors that are tried to see if it substantially changes the result.
	
	jump_size is determined automatically by comparing the maximum and median diff. If tweaking the jump_size 
	point by some small about causes the number of selected points to change, it'll
	print a warning.
	
	#TODO: If unsure, check if all bins have same number under some threshold condition. That seems right!
	#TODO: Try to round to a nice number, or make step sizes similar.
	#TODO: This should be an iterative function, it doesn't work well on the dataset: X = [0.9, 1, 1.1, 3.1, 3.1, 2.9, 9, 8.8, 9.1].
		The iterative nature it needs to add is saying, if the diffs [x,y, and z] are jumps, then the actual median is _ because
		we'll only look at the median over specific bins, and exclude the diffs.
	'''
	
	if jump_scale is not None or jump_size is not None:
		print(f"{Fore.RED} This feature has not been implemented. Using autoscale. You should fix me!{Style.RESET_ALL}")
	
	# Calc parameters of the input data
	delta = np.abs(np.diff(data))
	med = np.mean(delta)
	mx = np.max(delta)
	# ratio = mx/med # This is the maximum possible scaling value
	gap = mx - med
	
	# Set threshold and check validity
	thresh = gap*autoscale_fact+med
	thresh_L = gap*autoscale_checks[0]+med
	thresh_H = gap*autoscale_checks[1]+med
	
	# Find jump indeces
	idx_jump = np.where(delta > thresh)[0]
	idx_jump_L = np.where(delta > thresh_L)[0]
	idx_jump_H = np.where(delta > thresh_H)[0]
	
	# Run each threshold
	nbins = len(idx_jump)
	nbins_L = len(idx_jump_L)
	nbins_H = len(idx_jump_H)
	
	# Check for instability
	if nbins != nbins_H or nbins != nbins_L:
		print(f"{Fore.RED}WARNING: bin_empirical() failed to confidently bin the data! Panic! Check my work!{Style.RESET_ALL}")
		show_work = True
		
	if show_work:
		print(f"{Fore.YELLOW}Tried values: {Fore.LIGHTBLACK_EX}{thresh}->{nbins} bins, {thresh_H}->{nbins_H} bins, {thresh_L}->{nbins_L} bins,{Style.RESET_ALL}")
		print(f"{Fore.YELLOW}Max Diff: {Fore.LIGHTBLACK_EX}{mx}{Style.RESET_ALL}")
		print(f"{Fore.YELLOW}Mean Diff: {Fore.LIGHTBLACK_EX}{med}{Style.RESET_ALL}")
		print(f"{Fore.YELLOW}Gap: {Fore.LIGHTBLACK_EX}{gap}{Style.RESET_ALL}")
		
	
	# Initialize output data
	data_out = []
	
	# Edge case, add one more
	idx_jump = np.concatenate([idx_jump, np.array([len(data)-1])])
	
	# Now we can bin each result
	jump_last = 0
	for jump in idx_jump:
		
		# Define bin
		bin = data[jump_last:jump+1]
		jump_last = jump+1
		
		# Create new homogenous bin data
		nb = [np.mean(bin)]*len(bin)
		
		# Add to output list
		data_out = data_out + nb
	
	
	
	return data_out
	
	

def dfplot2d(df, xparam:str, yparam:str, fixedparam:dict=None, skip_plot:bool=False, fig_no:int=1, autoshow:bool=False, subplot_no:tuple=None, show_markers:bool=True, cmap:str='viridis', hovertips:bool=True):
	''' Accepts a dataframe as input, and returns a tuple with (x, y)
 	2D lists to plot using contourf(). 
	
	Fixed params: Dictionary describing parameters in the dataframe to filter such that it's filtered at a specific value. The dictionary
	keys are the column names of the dataframe, and the values are tuples with three elements:
		* Element 0: Value to filter for
		* Element 1: Percent tolerance. Note this is percent, not a fraction.
		* Element 2: Absolute tolerance. Note this applies as value +/- tolerance.
	If the value is set to a scalar instead of a tuple, both tolerance specifiers are set to zero. If both tolerance specifiers are listed,
	the larger resulting tolerance is used.
	'''
	
	# Trim DF per fixed-parameters
	if fixedparam is not None:
		for param, val_tup in fixedparam.items():
			
			# Determine tolerance
			if type(val_tup) != tuple:
				val = val_tup
				tol = 0
			else:
				val = val_tup[0]
				tol = np.max([np.abs(val*val_tup[1]/100), np.abs(val_tup[2])])
			
			# Apply filtering
			if tol == 0:
				df = df[(df[param]==val)]
			else:
				df = df[(df[param]>=(val-tol)) & (df[param] <= (val+tol))]

	
	# Ensure the DataFrame contains the necessary columns
	if not {xparam, yparam}.issubset(df.columns):
		raise ValueError("DataFrame missing specified columns")

	# Create a grid of points
	x = df[xparam].unique()

	# Reshape zparam to match the grid shape - transpose to turn column into row (so it's a 1D numpy array). [0] is to access first row.
	y = df.pivot_table(index=xparam, values=yparam).T.values[0]
	
	# Skip plot if asked
	if not skip_plot:
		
		# Create the contourf plot
		plt.figure(fig_no)
		if subplot_no is not None:
			plt.subplot(subplot_no[0], subplot_no[1], subplot_no[2])
		plt.plot(x, y)
		plt.xlabel(xparam)
		plt.ylabel(yparam)
	
	if autoshow:
		plt.show()
	
	# Return data values
	return (x, y, df)

def dfplotcm(df, xparam:str, yparam:str, zparam:str, fixedparam:dict=None, skip_plot:bool=False, fig_no:int=1, autoshow:bool=False, subplot_no:tuple=None, cmap:str='viridis'):
	''' Accepts a dataframe as input, and returns a tuple with (X,Y,Z)
 	2D lists to plot using contourf(). 
	
	Fixed params: Dictionary describing parameters in the dataframe to filter such that it's filtered at a specific value. The dictionary
	keys are the column names of the dataframe, and the values are tuples with three elements:
		* Element 0: Value to filter for
		* Element 1: Percent tolerance. Note this is percent, not a fraction.
		* Element 2: Absolute tolerance. Note this applies as value +/- tolerance.
	If the value is set to a scalar instead of a tuple, both tolerance specifiers are set to zero. If both tolerance specifiers are listed,
	the larger resulting tolerance is used.
	'''
	
	# Trim DF per fixed-parameters
	if fixedparam is not None:
		for param, val_tup in fixedparam.items():
			
			# Determine tolerance
			if type(val_tup) != tuple:
				val = val_tup
				tol = 0
			else:
				val = val_tup[0]
				tol = np.max([np.abs(val*val_tup[1]/100), np.abs(val_tup[2])])
			
			# Apply filtering
			if tol == 0:
				df = df[(df[param]==val)]
			else:
				df = df[(df[param]>=(val-tol)) & (df[param] <= (val+tol))]

	
	# Ensure the DataFrame contains the necessary columns
	if not {xparam, yparam, zparam}.issubset(df.columns):
		raise ValueError("DataFrame missing specified columns")

	# Create a grid of points
	# x = np.linspace(df[xparam].min(), df[xparam].max(), len(df[xparam].unique()))
	# y = np.linspace(df[yparam].min(), df[yparam].max(), len(df[yparam].unique()))
	x = df[xparam].unique()
	y = df[yparam].unique()
	X, Y = np.meshgrid(x, y)

	# Reshape zparam to match the grid shape
	Z = df.pivot_table(index=yparam, columns=xparam, values=zparam).values
	
	# Skip plot if asked
	if not skip_plot:
		
		# Create the contourf plot
		plt.figure(fig_no)
		if subplot_no is not None:
			plt.subplot(subplot_no[0], subplot_no[1], subplot_no[2])
		plt.contourf(X, Y, Z, levels=35, cmap=cmap)
		# plt.contourf(X, Y, Z)
		plt.colorbar()  # Add a colorbar to a plot
		plt.xlabel(xparam)
		plt.ylabel(yparam)
		plt.title(zparam)
	
	if autoshow:
		plt.show()
	
	# Return data values
	return (X, Y, Z)

def dfplot3d(df, xparam:str, yparam:str, zparam:str, xlabel:str=None, ylabel:str=None, zlabel:str=None, fixedparam:dict=None, skip_plot:bool=False, fig_no:int=1, autoshow:bool=False, show_markers:bool=False, projections=None, hovertips:bool=True, subplot_no:tuple=None, cmap:str='coolwarm', marker:str='.', markersize:float=3, markercolor:tuple=(0, 0.4, 0.7)):
	''' Accepts a dataframe as input, and returns a tuple with (X,Y,Z)
# 	2D lists to plot using contourf(). '''
	
	# Get X Y and Z from dataframe
	X, Y, Z = dfplotcm(df, xparam, yparam, zparam, fixedparam=fixedparam, skip_plot=True, fig_no=fig_no, autoshow=False)
	
	# Generate 3D plot
	lplot3d(X, Y, Z, xparam, yparam, zparam, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, skip_plot=skip_plot, fig_no=fig_no, autoshow=autoshow, show_markers=show_markers, projections=projections, hovertips=hovertips, subplot_no=subplot_no, cmap=cmap, marker=marker, markersize=markersize, markercolor=markercolor)
	
	# Return data values
	return (X, Y, Z)

def lplot3d(X, Y, Z, xparam:str, yparam:str, zparam:str, xlabel:str=None, ylabel:str=None, zlabel:str=None, skip_plot:bool=False, fig_no:int=1, autoshow:bool=False, show_markers:bool=False, projections=None, hovertips:bool=True, subplot_no:tuple=None, cmap:str='coolwarm', marker:str='.', markersize:float=3, markercolor:tuple=(0, 0.4, 0.7), title:str=None):
	''' Accepts a dataframe as input, and returns a tuple with (X,Y,Z)
# 	2D lists to plot using contourf(). '''
	
	fig = plt.figure(fig_no)
	if subplot_no is not None:
		ax = fig.add_subplot(subplot_no[0], subplot_no[1], subplot_no[2], projection='3d')
	else:
		ax = fig.add_subplot(projection='3d')
	fig.tight_layout()
	
	ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0.5, antialiased=True, rstride=1, cstride=1) #edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
	if projections is not None:
		ax.contourf(X,Y,Z, zdir='z', offset=-100, cmap='coolwarm')
		ax.contourf(X,Y,Z, zdir='x', offset=-40, cmap='coolwarm')
		ax.contourf(X,Y,Z, zdir='y', offset=40, cmap='coolwarm')
	
	if xlabel is None:
		xlabel = xparam
	if ylabel is None:
		ylabel = yparam
	if zlabel is None:
		zlabel = zparam
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	
	if title is not None:
		ax.set_title(title)
		
	# Initialize an empty annotation - Thanks Copilot
	annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
	annot.set_visible(False)
	
	if show_markers:
		
		flatX = X.flatten()
		flatY = Y.flatten()
		flatZ = Z.flatten()
		
		sc = ax.scatter(flatX,flatY, flatZ, color=markercolor, s=markersize, marker=marker, label='Data points', picker=True)
	
		# Update the annotation when hovering over points - Thanks Copilot
		def update_annot(annot_text, mouse_event):
			annot.xy = (mouse_event.x, mouse_event.y)
			annot.set_text(annot_text)
			annot.get_bbox_patch().set_facecolor('white')
			annot.get_bbox_patch().set_alpha(0.7)
		
		def annotate_onclick(event):
			print(event.ind)
			# print(event.artist.get_data())
			point_index = int(event.ind[0])
			x_coord, y_coord, z_coord = flatX[point_index], flatY[point_index], flatZ[point_index]
			annot_text = f"Point {point_index}: X={x_coord:.2f}, Y={y_coord:.2f}, Z={z_coord:.2f}"
			print(annot_text)
			annot.set_visible(True)
			
			update_annot(annot_text, event.mouseevent)
		
		# def hover(event):
		# 	vis = annot.get_visible()
		# 	if event.inaxes == ax:
		# 		cont, ind = sc.contains(event)
		# 		if cont:
		# 			update_annot(ind)
		# 			annot.set_visible(True)
		# 			fig.canvas.draw_idle()
		# 		else:
		# 			if vis:
		# 				annot.set_visible(False)
		# 				fig.canvas.draw_idle()
		
		if hovertips:
			print(f"Hovertips on")
			# fig.canvas.mpl_connect("motion_notify_event", hover)
			fig.canvas.mpl_connect("pick_event", annotate_onclick)
	
	if autoshow:
		plt.show()
	
def make_loss_lookup_fn(freq, loss):
	''' Given a list of X and Y values, returns a function that interpolates said values
	given some target.
	'''
	
	# Convert to numpy
	freq_np = np.array(freq)
	loss_np = np.array(loss)
	
	# Create template function
	def template_fn(f, guess_outside:bool=False):
		
		# Target is outside!
		if f < np.min(freq_np) or f > np.max(freq_np):
			#TODO: Check for guess_outside
			return None
		
		closest_idx = np.argmin(np.abs(freq_np-f)) # Find closest freqeucny
		
		try:
			x1, x2 = freq_np[closest_idx], freq_np[closest_idx + 1]
			y1, y2 = loss_np[closest_idx], loss_np[closest_idx + 1]
		except:
			x1, x2 = freq_np[closest_idx-1], freq_np[closest_idx]
			y1, y2 = loss_np[closest_idx-1], loss_np[closest_idx]

		return y1 + (f - x1) * (y2 - y1) / (x2 - x1)
	
	return template_fn

def wildcard(checks:list, pattern:str, ignore_case:bool=False):
	''' Returns None if 0 or 2+ match, else returns the string'''
	
	if ignore_case:
		matched = fnmatch.filter(checks, pattern)
	else:
		matched = [n for n in checks if fnmatch.fnmatchcase(n, pattern)]
		
	if len(matched) != 1:
		return None
	else:		
		matched = matched[0]
	
	return matched


def read_s2p(filename:str) -> pd.DataFrame:
	
	# Create network object from filename
	try:
		data_net = skrf.Network(filename)
	except:
		return None
	
	# Convert to pandas dataframe
	net_df = data_net.to_dataframe(attrs=['s_re', 's_im'])
	
	# Convert to internally standardized format
	dfd = {}
	dfd['freq_Hz'] = list(net_df.index)
	
	dfd['S11_real'] = list(net_df['s_re 11'])
	dfd['S21_real'] = list(net_df['s_re 21'])
	dfd['S12_real'] = list(net_df['s_re 12'])
	dfd['S22_real'] = list(net_df['s_re 22'])
	
	dfd['S11_imag'] = list(net_df['s_im 11'])
	dfd['S21_imag'] = list(net_df['s_im 21'])
	dfd['S12_imag'] = list(net_df['s_im 12'])
	dfd['S22_imag'] = list(net_df['s_im 22'])
	
	df = pd.DataFrame(dfd)
	
	return df
	
def read_rohde_schwarz_csv(filename:str) -> pd.DataFrame:
	''' Reads a CSV file containing S-parameter data from a Rohde & Schwarz ZVA
	vector network analyzer. Returns a Pandas DataFrame.
	
	Returns NOne or throws an error on errors.
	'''
	
	# Read file
	df = pd.read_csv(filename, header=2, sep=',')
	
	# Use wildcard compare to find unnamed column
	cols = list(df.columns)
	ch_unnamed = fnmatch.filter(cols, 'Unnamed*')
	if len(ch_unnamed) != 1:
		
		# df not in expected form - try reading again with tabs
		df = pd.read_csv(filename, header=2, sep='\t')
		cols = list(df.columns)
		ch_unnamed = fnmatch.filter(cols, 'Unnamed*')
		
		# If format is still wrong, return
		if len(ch_unnamed) != 1:
			return None
	
	ch_unnamed = ch_unnamed[0]
	
	# Remove last column (insturments adds extra comma)
	df = df.drop(ch_unnamed, axis=1)
	
	# Rename columns to something more readable
	re_s11 = wildcard(cols, 're*S11', ignore_case=True)
	im_s11 = wildcard(cols, 'im*S11', ignore_case=True)
	if (re_s11 is not None) and (im_s11 is not None):
		df = df.rename(columns={re_s11:"S11_real", im_s11:"S11_imag"})
		
	re_s21 = wildcard(cols, 're*S21', ignore_case=True)
	im_s21 = wildcard(cols, 'im*S21', ignore_case=True)
	if (re_s21 is not None) and (im_s21 is not None):
		df = df.rename(columns={re_s21:"S21_real", im_s21:"S21_imag"})
		
	re_s12 = wildcard(cols, 're*S12', ignore_case=True)
	im_s12 = wildcard(cols, 'im*S12', ignore_case=True)
	if (re_s12 is not None) and (im_s12 is not None):
		df = df.rename(columns={re_s12:"S12_real", im_s12:"S12_imag"})

	re_s22 = wildcard(cols, 're*S22', ignore_case=True)
	im_s22 = wildcard(cols, 'im*S22', ignore_case=True)
	if (re_s22 is not None) and (im_s22 is not None):
		df = df.rename(columns={re_s22:"S22_real", im_s22:"S22_imag"})
	
	freq = wildcard(cols, 'freq*Hz*', ignore_case=True)
	if freq is not None:
		df = df.rename(columns={freq:'freq_Hz'})
	
	return df

def select_evenly_spaced_values(original_list, num_values=11):
	"""
	Selects approximately evenly spaced values from the original list.
	(Thanks Copilot!)

	Args:
		original_list (list): A list of ascending floats.
		num_values (int, optional): Number of values to select (default is 10).

	Returns:
		list: A new list containing approximately evenly spaced values.
	"""
	if len(original_list) <= num_values:
		return original_list  # No need to interpolate; return the entire list

	# Calculate the step size
	step = (len(original_list) - 1) / (num_values - 1)

	# Create the new list by selecting values at evenly spaced indices
	new_list = [original_list[int(i * step)] for i in range(num_values)]

	return new_list


def downsample_labels(original_list, num_values=11):
	"""
	Selects approximately evenly spaced values from the original list.
	(Thanks Copilot!)

	Args:
		original_list (list): A list of ascending floats.
		num_values (int, optional): Number of values to select (default is 10).

	Returns:
		list: A new list containing approximately evenly spaced values.
	"""
	
	if len(original_list) <= num_values:
		return original_list  # No need to interpolate; return the entire list

	# Calculate the step size
	step = (len(original_list) - 1) / (num_values - 1)

	# Create the new list by selecting values at evenly spaced indices
	selected_indecies = [int(i * step) for i in range(num_values)]
	
	new_list = []
	for idx, orig_val in enumerate(original_list):
		if idx in selected_indecies:
			new_list.append(str(orig_val))
		else:
			new_list.append("")

	return new_list
