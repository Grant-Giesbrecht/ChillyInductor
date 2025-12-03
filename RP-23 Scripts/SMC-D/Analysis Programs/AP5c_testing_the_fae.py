from RP23SMCD import *

file_trad = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f07_00000.txt")
file_doubler = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f10_00000.txt")
file_tripler = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f14_00000.txt")




p_trad = LoadParameters(file_trad)
p_trad.t_start = -1435*1e-9
p_trad.t_end = 0
r_trad = full_analysis(p_trad, fignum=1)

def analyze_carrier(t, v, R=50, harmonics=(1, 2, 3), window_type="hann", linewidth_scan_hz=50e6):
	"""
	Analyze a modulated carrier waveform:
	  • carrier frequency (refined)
	  • amplitude (RMS and dBm)
	  • harmonic powers (in dBm)
	  • tone width (-3 dB and -20 dB)
	  • phase-noise PSD around the carrier

	Parameters
	----------
	t : np.ndarray
		Time axis (seconds)
	v : np.ndarray
		Voltage waveform (volts)
	R : float
		Load resistance (ohms). Default 50.
	harmonics : tuple
		Harmonic numbers to evaluate.
	window_type : str
		"hann" or "nuttall".
	linewidth_scan_hz : float
		Frequency span (±X) used to estimate tone width.

	Returns
	-------
	dict with:
		f0              refined carrier frequency (Hz)
		A0_rms          RMS carrier amplitude (Volts)
		P0_dBm          Carrier power in dBm
		harmonics_dBm   dict: harmonic → power in dBm
		linewidth_3dB   approximate -3 dB linewidth (Hz)
		linewidth_20dB  approximate -20 dB linewidth (Hz)
		PN_freq         offsets for phase-noise PSD (Hz)
		PN_dBc          phase noise in dBc/Hz
		freq            full frequency vector
		PSD_dBm_per_Hz  full PSD
	"""
	
	# ---- Basic sampling ----
	t = np.asarray(t)
	v = np.asarray(v)
	dt = t[1] - t[0]
	fs = 1.0 / dt
	N = len(v)

	# ---- Window ----
	if window_type.lower() == "hann":
		w = np.hanning(N)
	elif window_type.lower() == "nuttall":
		# This gives the cleanest sidelobes
		n = np.arange(N)
		w = (0.355768 
			 - 0.487396*np.cos(2*np.pi*n/(N-1))
			 + 0.144232*np.cos(4*np.pi*n/(N-1))
			 - 0.012604*np.cos(6*np.pi*n/(N-1)))
	else:
		raise ValueError("Unsupported window type")
	
	U = np.mean(w**2)  # power loss factor
	v_win = v * w 

	# ---- FFT ----
	V = np.fft.rfft(v_win)
	freq = np.fft.rfftfreq(N, d=dt)

	# ---- Single-sided PSD ----
	PSD = (np.abs(V)**2) / (R * fs * N * U)      # W/Hz
	PSD_dBm_per_Hz = 10*np.log10(PSD*1e3)        # dBm/Hz

	# ---- Find coarse carrier peak ----
	k0 = np.argmax(np.abs(V))

	# ---- Sub-bin refinement (log-parabolic) ----
	# ensure not at boundaries
	if k0 <= 1 or k0 >= len(V)-2:
		f0 = freq[k0]  # fallback
	else:
		alpha = np.log(np.abs(V[k0-1]))
		beta  = np.log(np.abs(V[k0]))
		gamma = np.log(np.abs(V[k0+1]))
		delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
		f0 = freq[k0] + delta * (fs / N)

	# ---- Carrier amplitude ----
	# amplitude estimate: magnitude at peak corrected for window
	A0 = (2.0 / np.sum(w)) * np.abs(V[k0])   # approximate amplitude
	A0_rms = A0 / np.sqrt(2)
	P0 = (A0_rms**2) / R
	P0_dBm = 10*np.log10(P0*1e3)

	# ---- Harmonic analysis ----
	harmonics_dBm = {}
	for h in harmonics:
		fh = h * f0
		# nearest bin
		kh = int(round(fh / (fs/N)))
		if kh >= 0 and kh < len(V):
			Ah = (2.0 / np.sum(w)) * np.abs(V[kh])
			Ah_rms = Ah/np.sqrt(2)
			Ph = (Ah_rms**2)/R
			harmonics_dBm[h] = 10*np.log10(Ph*1e3)
		else:
			harmonics_dBm[h] = None

	# ---- Tone width estimation (coarse but robust) ----
	# Look ±linewidth_scan_hz around f0
	df = freq[1] - freq[0]
	span_bins = int(linewidth_scan_hz / df)
	k_center = k0
	k_min = max(k_center - span_bins, 0)
	k_max = min(k_center + span_bins, len(freq)-1)

	local_freq = freq[k_min:k_max]
	local_P = PSD[k_min:k_max]

	# find -3 dB and -20 dB bandwidths
	P_peak = PSD[k_center]
	P_3db = P_peak / 2
	P_20db = P_peak / 100

	def width_at(P_target):
		# indices where PSD crosses P_target
		mask = local_P >= P_target
		if np.sum(mask) < 2:
			return None
		fmin = local_freq[np.where(mask)[0][0]]
		fmax = local_freq[np.where(mask)[0][-1]]
		return fmax - fmin

	linewidth_3dB = width_at(P_3db)
	linewidth_20dB = width_at(P_20db)

	# ---- Phase-noise (dBc/Hz) ----
	# Exclude the carrier bin; compute offset frequencies
	PN_mask = freq > 0
	PN_mask[np.argmax(PSD)] = False  # remove carrier bin
	PN_freq = np.abs(freq[PN_mask] - f0)
	PN_dBc = PSD_dBm_per_Hz[PN_mask] - P0_dBm

	return {
		"f0": f0,
		"A0_rms": A0_rms,
		"P0_dBm": P0_dBm,
		"harmonics_dBm": harmonics_dBm,
		"linewidth_3dB": linewidth_3dB,
		"linewidth_20dB": linewidth_20dB,
		"PN_freq": PN_freq,
		"PN_dBc": PN_dBc,
		"freq": freq,
		"PSD_dBm_per_Hz": PSD_dBm_per_Hz,
	}

acr_trad = analyze_carrier(r_trad.t_si, r_trad.v_si)

plt.show()