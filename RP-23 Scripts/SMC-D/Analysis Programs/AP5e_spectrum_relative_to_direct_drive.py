from RP23SMCD import *
import mplcursors

file_trad = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f07_00000.txt")
file_doubler = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f10_00000.txt")
file_tripler = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f14_00000.txt")

def diff_on_common_x(x1, y1, x2, y2, *, xgrid="x2", extrapolate=False):
	"""
	Returns (X, Ydiff) where Ydiff = y2_interp(X) - y1_interp(X).

	xgrid:
	  - "x2": use x2 as the output X grid (default)
	  - "x1": use x1 as the output X grid
	  - "union": use sorted union of x1 and x2 as the output X grid
	  - array-like: explicit X grid to use

	extrapolate:
	  - False: restrict to overlapping X-range (recommended)
	  - True: allow linear extrapolation outside range (use with care)
	"""
	x1 = np.asarray(x1, dtype=float); y1 = np.asarray(y1, dtype=float)
	x2 = np.asarray(x2, dtype=float); y2 = np.asarray(y2, dtype=float)

	if x1.shape != y1.shape or x2.shape != y2.shape:
		raise ValueError("x1/y1 and x2/y2 must be same shape respectively.")

	# Sort by x (np.interp requires increasing x)
	i1 = np.argsort(x1); x1s, y1s = x1[i1], y1[i1]
	i2 = np.argsort(x2); x2s, y2s = x2[i2], y2[i2]

	# Choose output grid
	if isinstance(xgrid, str):
		if xgrid == "x2":
			X = x2s
		elif xgrid == "x1":
			X = x1s
		elif xgrid == "union":
			X = np.unique(np.concatenate([x1s, x2s]))
		else:
			raise ValueError("xgrid must be 'x1', 'x2', 'union', or an array-like.")
	else:
		X = np.asarray(xgrid, dtype=float)
		X = np.sort(X)

	# If not extrapolating, restrict to overlap so both interps are valid
	if not extrapolate:
		lo = max(x1s[0], x2s[0])
		hi = min(x1s[-1], x2s[-1])
		X = X[(X >= lo) & (X <= hi)]

	if X.size == 0:
		raise ValueError("No overlapping X-range between datasets (or chosen X grid).")

	# Interpolate
	if extrapolate:
		y1i = np.interp(X, x1s, y1s)  # np.interp extrapolates with edge values, not linear
		y2i = np.interp(X, x2s, y2s)
		# If you truly want *linear* extrapolation, use scipy.interpolate.interp1d(..., fill_value="extrapolate")
	else:
		y1i = np.interp(X, x1s, y1s)
		y2i = np.interp(X, x2s, y2s)

	return X, (y2i - y1i)


p_trad = LoadParameters(file_trad)
p_trad.t_start = -1435*1e-9
p_trad.t_end = 0
r_trad = full_analysis(p_trad, fignum=1)

p_doub = LoadParameters(file_doubler)
p_doub.t_start = -1600*1e-9
p_doub.t_end = -100*1e-9
r_doub = full_analysis(p_doub, fignum=2)

p_trip = LoadParameters(file_tripler)
p_trip.t_start = -1600*1e-9
p_trip.t_end = -100*1e-9
r_trip = full_analysis(p_trip, fignum=3)

fig4, ax4a = plot_spectrum_overlay([r_trad, r_doub, r_trip], fignum=4)
ax4a.legend(["Traditional", "Doubler", "Tripler"])


(freq_com, tripler_delta) = diff_on_common_x(r_trad.freqs, r_trad.spectrum, r_trip.freqs, r_trip.spectrum)

plt.figure(5)
plt.plot(np.array(freq_com)/1e9, tripler_delta, linestyle=":", marker=".")
plt.grid(True)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Tripler PSD - Direct Drive PSD (dBm/Hz)")

mplcursors.cursor(multiple=True)

plt.show()