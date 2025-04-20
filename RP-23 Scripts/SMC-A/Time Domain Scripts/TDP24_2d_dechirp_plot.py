import os
from decimal import Decimal
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#=============================== USER CONFIG =====================

# Example list of target floats
target_floats = [5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3]
target_floats = [5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4]


# base_dir = '/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-A/Time Domain Measurements/17April2025_DownMix/dechirp_strength_-8dBm'
# trim_time_ns = [-40, -10]

base_dir = '/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-A/Time Domain Measurements/17April2025_DownMix/dechirp_strength_-11dBm'
trim_time_ns = [-55, 0]
trim_time_ns = [-40, -10]

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

def analyze_file(base_dir, filename, trim_time_ns:list=None, N_avg:int=1):
	
	# Read file
	try:
		df = pd.read_csv(os.path.join(base_dir, filename), skiprows=4, encoding='utf-8')
	except:
		print(f"Failed to find file {filename}. Aborting.")
		sys.exit()
	
	# Create local variables
	t_si = np.array(df['Time'])
	v_si = np.array(df['Ampl'])
	
	if trim_time_ns is not None:
		print("trim")
		idx_start = find_closest_index(t_si, trim_time_ns[0]*1e-9)
		idx_end = find_closest_index(t_si, trim_time_ns[1]*1e-9)
		t_si = t_si[idx_start:idx_end+1]
		v_si = v_si[idx_start:idx_end+1]
	
	tzc = find_zero_crossings(t_si, v_si)
	
	# Select every other zero crossing to see full periods and become insensitive to y-offsets.
	tzc_fullperiod = tzc[::2*N_avg]
	periods = np.diff(tzc_fullperiod)
	freqs = (1/periods)*N_avg

	t_freqs = tzc_fullperiod[:-1] + periods/2
	
	return (t_si, v_si, t_freqs, freqs)

#========================= LOCATE FILES AUTOMATICALLY =====================

# Convert floats to strings exactly as they would appear in the filename
target_strings = [f"Dechirp{float_to_clean_sci_str(f)}D" for f in target_floats]

print("Target strings:")
for ts in target_strings:
	print(f"  {ts}")

# Get all filenames (e.g., from a folder)
filenames = os.listdir(base_dir)

# Filter filenames
filtered_files = [
	fname for fname in filenames
	if fname.startswith("C1") and any(ts in fname for ts in target_strings)
]

filtered_files = []
for targ_string in target_strings:
	
	found = False
	
	for fname in filenames:
		if fname.startswith("C1") and (targ_string in fname):
			filtered_files.append(fname)
			found = True
	
	if not found:
		print(f"Failed to find file for target string:")
		print(f"  {targ_string}")
		sys.exit()

print(f"Filtered files:")
for ff in filtered_files:
	print(f"  {ff}")

##============= Analyze Files ===========================

times = []
volts = []
ftimes = []
freqs = []
all_dechirp = []

t_min = -1e9
t_max = 1e9

# Analyze each file
for dechirp, ff in zip(target_floats, filtered_files):
	
	# Analyze file
	(t_, v_, tf_, fr_) = analyze_file(base_dir, ff, trim_time_ns=trim_time_ns, N_avg=N_avg)
	
	# Get time limits
	t_min = np.max([np.min(tf_), t_min])
	t_max = np.min([np.max(tf_), t_max])
	
	# Append to master data
	times.append(t_)
	volts.append(v_)
	ftimes.append(tf_)
	freqs.append(fr_)
	all_dechirp.append(np.ones(len(tf_))*dechirp)

print(t_min)
print(t_max)

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
	
	uni_freqs.append(np.array(new_freq)/1e9)

Z = np.array(uni_freqs)          # Shape: (n_files, len(t))
X = np.array(uni_ftime)               # Same t for all, so just take one
Y = np.array(target_floats) 

#Create 2D meshgrid for plotting or analysis
X_grid, Y_grid = np.meshgrid(X*1e9, np.log10(Y))

fig1 = plt.figure()
plt.pcolormesh(X_grid, Y_grid, Z, shading='auto')
plt.xlabel("Time (ns)")
plt.ylabel("log10(Dechirp Parameter)")
plt.title("2D grid from analyze_file")
plt.colorbar(label="Frequency (GHz)")

#Create 2D meshgrid for plotting or analysis
X_grid_lin, Y_grid_lin = np.meshgrid(X*1e9, Y)

fig2 = plt.figure()
plt.pcolormesh(X_grid_lin, Y_grid_lin, Z, shading='auto')
plt.xlabel("Time (ns)")
plt.ylabel("Dechirp Parameter")
plt.title("2D grid from analyze_file")
plt.colorbar(label="Frequency (GHz)")

# Optional: visualize the result
# plt.pcolormesh(X_grid, Y_grid, Z, shading='auto')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z, )
plt.xlabel("Time (ns)")
plt.ylabel("log10(Dechirp Parameter)")
plt.title("2D grid from analyze_file")
plt.show()
# plt.colorbar(label="Frequency (GHz)")

# Flatten all values for plotting
# X_all = np.concatenate(ftimes)
# X_all = [item for sublist in ftimes for item in sublist]
# Y_all = [item for sublist in all_dechirp for item in sublist]
# Z_all = [item for sublist in freqs for item in sublist]
# Y_all = np.concatenate(all_dechirp)
# Z_all = np.concatenate(freqs)

# # Create scatter or interpolated 2D plot (pcolormesh won't work directly on ragged t)
# plt.figure(figsize=(10, 6))
# sc = plt.scatter(X_all, Y_all, c=Z_all/1e9, cmap='viridis', s=5)
# plt.xlabel("t")
# plt.ylabel("idx")
# plt.colorbar(sc, label="v")
# plt.title("2D scatter plot with variable t")
# plt.show()