import os
from decimal import Decimal
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#=============================== USER CONFIG =====================

# Example list of target floats
target_floats = [5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3]

#+========================= DEFINE FUNCTIONS =====================

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

def analyze_file(base_dir, filename):
	
	# Read file
	try:
		df = pd.read_csv(os.path.join(base_dir, filename), skiprows=4, encoding='utf-8')
	except:
		print(f"Failed to find file {filename}. Aborting.")
		sys.exit()
	
	# Create local variables
	t = np.array(df['Time']*1e9)
	v = np.array(df['Ampl']*1e3)

	t_si = np.array(df['Time'])
	v_si = np.array(df['Ampl'])
	
	tzc = find_zero_crossings(t_si, v_si)
	N_avg = 1
	
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
base_dir = '/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-A/Time Domain Measurements/17April2025_DownMix/dechirp_strength_-8dBm'
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

for dechirp, ff in zip(target_floats, filtered_files):
	
	# Analyze file
	(t_, v_, tf_, fr_) = analyze_file(base_dir, ff)
	
	# Append to master data
	times.append(t_)
	volts.append(v_)
	ftimes.append(tf_)
	freqs.append(fr_)
	all_dechirp.append(np.ones(len(tf_))*dechirp)

# Z = np.array(freqs)          # Shape: (n_files, len(t))
# X = np.array(times[0])               # Same t for all, so just take one
# Y = np.array(target_floats) 

# Create 2D meshgrid for plotting or analysis
# X_grid, Y_grid = np.meshgrid(X, Y)

# # Optional: visualize the result
# plt.pcolormesh(X_grid, Y_grid, Z, shading='auto')
# plt.xlabel("Time (ns)")
# plt.ylabel("Dechirp Parameter")
# plt.colorbar(label="Frequency (GHz)")
# plt.title("2D grid from analyze_file")
# plt.show()

# Flatten all values for plotting
X_all = np.concatenate(ftimes)
# X_all = [item for sublist in ftimes for item in sublist]
# Y_all = [item for sublist in all_dechirp for item in sublist]
# Z_all = [item for sublist in freqs for item in sublist]
Y_all = np.concatenate(all_dechirp)
Z_all = np.concatenate(freqs)

# Create scatter or interpolated 2D plot (pcolormesh won't work directly on ragged t)
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_all, Y_all, c=Z_all/1e9, cmap='viridis', s=5)
plt.xlabel("t")
plt.ylabel("idx")
plt.colorbar(sc, label="v")
plt.title("2D scatter plot with variable t")
plt.show()