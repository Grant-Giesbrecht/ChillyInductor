import os
from decimal import Decimal, getcontext
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sys import platform
from graf.base import sample_colormap

def linspace_st(start:float, stop:float, step:float):
	num_points = int(np.floor((stop - start) / step)) + 1
	return np.linspace(start, start + step * (num_points - 1), num_points)

# def generate_decimal_list(start, stop, step):
# 	getcontext().prec = 10  # Set precision high enough
# 	start = Decimal(str(start))
# 	stop = Decimal(str(stop))
# 	step = Decimal(str(step))
# 	
# 	result = []
# 	value = start
# 	while value <= stop:
# 		result.append(f"{value:.3f}")
# 		value += step
# 	
# 	return result

def generate_decimal_list(start, stop, step):
	getcontext().prec = 10  # Set precision high enough
	start = Decimal(str(start))
	stop = Decimal(str(stop))
	step = Decimal(str(step))
	
	result = []
	value = start
	while value <= stop:
		# Normalize to remove trailing zeros, then convert to string
		result.append(str(value.normalize()))
		value += step
	
	return result

#=============================== USER CONFIG =====================

# # Example list of target floats - 100 MHz, -8 dBm
# target_floats = linspace_st(0.025, 1.025, 0.025)
# target_floats = target_floats[:-2]
# target_str = generate_decimal_list(0.025, 1.025, 0.025)
# target_str = target_str[:-2]

# Example list of target floats - 500 MHz, 0 dBm
target_floats = linspace_st(0.1, 0.8, 0.025)
target_floats = target_floats[:-2]
target_str = generate_decimal_list(0.1, 0.8, 0.025)
target_str = target_str[:-2]

# base_dir = '/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-A/Time Domain Measurements/17April2025_DownMix/dechirp_strength_-8dBm'
# trim_time_ns = [-40, -10]

if platform == "darwin":
	# base_dir = '/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-A/Time Domain Measurements/17April2025_DownMix/power_sweep'
	# trim_time_ns = [-50, -10]
	base_dir = '/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-A/Time Domain Measurements/17April2025_DownMix/bias_sweep_500MHz'
	trim_time_ns = [-50, 30]
elif platform == "win32":
	base_dir = os.path.join('G:\\', 'ARC0 PhD Data', 'RP-23 Qubit Readout', 'Data', 'SMC-A', 'Time Domain Measurements', '17April2025_DownMix', 'bias_sweep_500MHz')
	# trim_time_ns = [-50, -10] # 100 MHz
	trim_time_ns = [-50, 30]



time_pt_mult = 3
N_avg = 4

#+========================= DEFINE FUNCTIONS =====================

def find_closest_index(lst, X):
	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
	return closest_index

def float_to_clean_sci_str(f):
	d = Decimal(str(f)).normalize()
	# Get scientific notation string
	sci_str = format(d, 'e')
	base, exp = sci_str.split('e')
	
	# Remove trailing .0 if present
	if base.endswith('.0'):
		base = base[:-2]
	else:
		base = base.rstrip('0').rstrip('.')  # Remove trailing zeros & dot if needed
	
	exp = str(int(exp))  # Remove leading zeros in exponent
	return f"{base}e{exp}"

def find_zero_crossings(x, y):
	''' Finds zero crossings of x, then uses y-data to interpolate between the points.'''
	
	signs = np.sign(y)
	sign_changes = np.diff(signs)
	zc_indices = np.where(sign_changes != 0)[0]
	
	# Trim end points - weird things can occur if the waveform starts or ends at zero
	zc_indices = zc_indices[1:-1]
	
	# Interpolate each zero-crossing
	cross_times = []
	for zci in zc_indices:
		dx = x[zci+1] - x[zci]
		dy = y[zci+1] - y[zci]
		frac = np.abs(y[zci]/dy)
		cross_times.append(x[zci]+dx*frac)
	
	return cross_times

def analyze_file(base_dir, filename, trig_filename, trim_time_ns:list=None, N_avg:int=1):
	
	# Read file
	try:
		df = pd.read_csv(os.path.join(base_dir, filename), skiprows=4, encoding='utf-8')
	except:
		print(f"Failed to find file {filename}. Aborting.")
		sys.exit()
	
	# Read file
	try:
		df_trig = pd.read_csv(os.path.join(base_dir, trig_filename), skiprows=4, encoding='utf-8')
	except:
		print(f"Failed to find file {filename}. Aborting.")
		sys.exit()
	
	# Create local variables
	t_si = np.array(df['Time'])
	v_si = np.array(df['Ampl'])
	
	# Create local variables
	t_si_trig = np.array(df_trig['Time'])
	v_si_trig = np.array(df_trig['Ampl'])
	
	
	if trim_time_ns is not None:
		print("  trim")
		idx_start = find_closest_index(t_si, trim_time_ns[0]*1e-9)
		idx_end = find_closest_index(t_si, trim_time_ns[1]*1e-9)
		t_si = t_si[idx_start:idx_end+1]
		v_si = v_si[idx_start:idx_end+1]
		print(f"    --> idx:{idx_start}:{idx_end}")
		t0 = t_si[0]
		tf = t_si[-1]
		print(f"    --> t:{t0*1e9}:{tf*1e9} ns")
	
	tzc = find_zero_crossings(t_si, v_si)
	
	# Select every other zero crossing to see full periods and become insensitive to y-offsets.
	tzc_fullperiod = tzc[::2*N_avg]
	periods = np.diff(tzc_fullperiod)
	freqs = (1/periods)*N_avg

	t_freqs = tzc_fullperiod[:-1] + periods/2
	
	return (t_si, v_si, t_si_trig, v_si_trig, t_freqs, freqs)

#========================= LOCATE FILES AUTOMATICALLY =====================

# Convert floats to strings exactly as they would appear in the filename
target_strings = [f"_{f}VBias".replace(".", ",") for f in target_str]
target_strings2 = [f"_{f}VBias" for f in target_str]

print("Target strings:")
for ts in target_strings:
	print(f"  {ts}")

# Get all filenames (e.g., from a folder)
filenames = os.listdir(base_dir)

# # Filter filenames
# filtered_files = [
# 	fname for fname in filenames
# 	if fname.startswith("C1") and any(ts in fname for ts in target_strings)
# ]

filtered_files = []
filtered_files_trig = []
for idx, targ_string in enumerate(target_strings):
	
	found_C1 = False
	found_C4 = False
	
	for fname in filenames:
		
		# Look for C1
		if fname.startswith("C1") and ((targ_string in fname) or (target_strings2[idx] in fname)):
			filtered_files.append(fname)
			found_C1 = True
		
		# Look for C4
		if fname.startswith("C4") and ((targ_string in fname) or (target_strings2[idx] in fname)):
			filtered_files_trig.append(fname)
			found_C4 = True
		
		# Abort when both found
		if found_C4 and found_C1:
			break
	
	if not (found_C1 and found_C4):
		print(f"Failed to find file for target string:")
		print(f"  {targ_string}")
		print("Filenames:")
		count = 0
		for f in filenames:
			if count == 2:
				count = 0
				print(f"")
			print(f"    {f}", end="")
			count += 1
		sys.exit()

print(f"Filtered files:")
for ff in filtered_files:
	print(f"  {ff}")

##============= Analyze Files ===========================

times = []
volts = []
ftimes = []
freqs = []
times_trig = []
volts_trig = []

t_min = -1e9
t_max = 1e9

# Analyze each file
for dechirp, ff, ftrig in zip(target_floats, filtered_files, filtered_files_trig):
	
	print(f"Analyzing for target: {dechirp}")
	
	# Analyze file
	(t_, v_, t_trig_, v_trig_, tf_, fr_) = analyze_file(base_dir, ff, ftrig, trim_time_ns=trim_time_ns, N_avg=N_avg)
	
	# Get time limits
	t_min = np.max([np.min(tf_), t_min])
	t_max = np.min([np.max(tf_), t_max])
	
	# Append to master data
	times.append(t_)
	volts.append(v_)
	ftimes.append(tf_)
	freqs.append(fr_)
	times_trig.append(t_trig_)
	volts_trig.append(v_trig_)

print(f"t_min: {t_min*1e9}")
print(f"t_max: {t_max*1e9}")

# # Get grid of common times
# all_freq_times = np.concatenate(ftimes)

# Get new universal times
# uni_ftime = np.linspace(np.min(all_freq_times), np.max(all_freq_times), int(len(all_freq_times)*time_pt_mult))
uni_ftime = np.linspace(t_min, t_max, int(len(tf_)*time_pt_mult))

# Scan over times and interpolate to uni time
uni_freqs = []
for idx, ft_ in enumerate(ftimes):
	
	# Interpolate
	fi = interp1d(ft_, freqs[idx], kind='linear')
	new_freq = fi(uni_ftime)
	
	uni_freqs.append(np.array(new_freq)/1e6)

Z = np.array(uni_freqs)          # Shape: (n_files, len(t))
X = np.array(uni_ftime)               # Same t for all, so just take one
Y = np.array(target_floats) 

#Create 2D meshgrid for plotting or analysis
X_grid, Y_grid = np.meshgrid(X*1e9, Y)

#================ Plot Data ========================
cmap = sample_colormap(cmap_name='viridis', N=len(volts))

fig1 = plt.figure()
plt.pcolormesh(X_grid, Y_grid, Z, shading='auto')
plt.xlabel("Time (ns)")
plt.ylabel("dc Bias (V)")
plt.title("2D grid from analyze_file")
plt.colorbar(label="Frequency (MHz)")

# Optional: visualize the result
# plt.pcolormesh(X_grid, Y_grid, Z, shading='auto')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z )
plt.xlabel("Time (ns)")
plt.ylabel("dc Bias (V)")
plt.gca().set_zlabel("Frequency (MHz)")
plt.title("2D grid from analyze_file")
# plt.colorbar(label="Frequency (GHz)")

idx = 0
fig4 = plt.figure(4)
for t, v in zip(times, volts):
	pwr = target_floats[idx]
	plt.subplot(2, 1, 1)
	plt.plot(t*1e9, v*1e3, linestyle='-', color=cmap[idx], alpha=0.35, label=f"{pwr} V")
	plt.subplot(2, 1, 2)
	plt.plot(t*1e9, v*1e3/np.max(v), linestyle='-', color=cmap[idx], alpha=0.35, label=f"{pwr} V")
	idx += 1

plt.subplot(2, 1, 1)
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (mV)")
plt.grid(True)
plt.legend(ncol=3)

plt.subplot(2, 1, 2)
plt.xlabel("Time (ns)")
plt.ylabel("Normalized Voltage (1)")
plt.grid(True)

idx = 0
fig5 = plt.figure(5)
for t, v in zip(times_trig, volts_trig):
	pwr = target_floats[idx]
	plt.subplot(2, 1, 1)
	plt.plot(t*1e9, v*1e3, linestyle='-', color=cmap[idx], alpha=0.35, label=f"{pwr} V")
	plt.subplot(2, 1, 2)
	plt.plot(t*1e9, v*1e3/np.max(v), linestyle='-', color=cmap[idx], alpha=0.35, label=f"{pwr} V")
	idx += 1

plt.subplot(2, 1, 1)
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (mV)")
plt.grid(True)
plt.legend(ncol=3)

plt.subplot(2, 1, 2)
plt.xlabel("Time (ns)")
plt.ylabel("Normalized Voltage (1)")
plt.grid(True)

plt.suptitle(f"Trigger Channel")

plt.show()