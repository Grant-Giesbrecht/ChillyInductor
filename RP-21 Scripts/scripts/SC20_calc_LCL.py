import skrf as rf
import os.path
import numpy as np
import matplotlib.pyplot as plt
import math

def find_nearest(array,value):
	""" Finds closest value.
	
	Thanks to StackExchange:
	https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
	"""
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
		return idx-1
	else:
		return idx

def TAE34_LC(filename):
	# Read S2P File
	net = rf.Network(filename)
	S11 = net.s[:, 0, 0]
	freqs = net.f

	# Windows in which to look for minima (GHz)
	ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3)]

	# Find minima
	fn = []
	S11n = []
	for rng in ranges:
		
		# Find frequencies
		idx_start = find_nearest(freqs, rng[0]*1e9)
		idx_end = find_nearest(freqs, rng[1]*1e9)
		
		# Find minimum
		S11_min = np.min(abs(S11[idx_start:idx_end])) # Minimum S11 value
		idx_min = list(abs(S11)).index(S11_min) # Index of that value
		
		# Add to lists
		f_new = freqs[idx_min]
		fn.append(f_new)
		S11n.append(S11_min)
		
		print(f"Adding point: f={f_new} GHz, S11 = {S11_min} (lin.)")

	# Convert result to Numpy array
	fn = np.array(fn)
	S11n = np.array(S11n)

	# Save as individual variables for printing
	f1 = fn[0]
	f2 = fn[1]
	f3 = fn[2]

	# Calculate detuned frequencies
	fdt1 = f1/2
	fdt2 = (f1+f2)/2
	fdt3 = (f2+f3)/2
	fdt = np.array([fdt1, fdt2, fdt3])
	S11dt = []
	for f in fdt:
		idx_f = find_nearest(freqs, f)
		S11dt.append(abs(S11[idx_f]))
	S11dt = np.array(S11dt)

	S11dt_avg = (np.sum(S11dt[1:]))/2

	# Calculate L
	f1comp = f3/3
	G = S11dt_avg
	l_phys = 0.5
	PI = 3.1415926535
	ZG = 50
	L = ZG/2/l_phys/f1comp*np.sqrt((1+G)/(1-G))
	C = (1/2/l_phys/f1comp/np.sqrt(L))**2

	Z0 = np.sqrt(L/C)
	Vp = 1/np.sqrt(L*C)
	Vp_c = Vp/3e8
	first_null = PI/l_phys/np.sqrt(L*C)
	nlambda_10G = l_phys/Vp*10e9

	# Print result in table
	print(f"\nfn Summary:")
	print(f"\tf1: {f1/1e6} MHz")
	print(f"\tf2: {f2/1e6} MHz = {f2/f1}xf1")
	print(f"\tf3: {f3/1e6} MHz = {f3/f1}xf1")
	print(f"\t|S21| de-tuned: {(20*np.log10(S11dt_avg))} dB")
	print(f"\tL': {L*1e9} nH")
	print(f"\tC': {C*1e12} pF")
	print(f"\n")
	print(f"Resulting Values:")
	print(f"\tZ0: {Z0} Ohms")
	print(f"\tVp: {Vp} m/s")
	print(f"\tVp/c: {Vp_c} ")
	print(f"\tFirst Null: {first_null/1e6} MHz")
	print(f"\tNo. Lambda @ 10GHz: {nlambda_10G} ")

	# Generate Graph
	plt.plot(freqs/1e9, 20*np.log10(abs(S11)), linestyle='dotted', marker='+', color=(0, 0, 0.7), label="All Data")
	plt.scatter(fn/1e9, 20*np.log10(S11n), marker='o', color=(0, 0.6, 0), label="Minima")
	plt.scatter(fdt/1e9, 20*np.log10(S11dt), marker='o', color=(0.6, 0.0, 0.4), label="De-Tuned")
	plt.xlabel("Frequency (GHz)")
	plt.ylabel("S11 (dB)")
	plt.title("Sweep 3 - 23 June 2023")
	plt.grid()
	plt.legend()

	plt.show()
	
	return (L, C)

if __name__ == "__main__":
	
	TAE34_LC(os.path.join("script_assets", "SC20", "sweep3_300M.s2p"))