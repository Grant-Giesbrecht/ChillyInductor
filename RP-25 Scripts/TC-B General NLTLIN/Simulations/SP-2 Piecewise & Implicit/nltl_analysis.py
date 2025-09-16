

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt

def analytic_signal(x: np.ndarray) -> np.ndarray:
	''' Creates an analytic signal from a real-valued signal: runs an FFT, removes all negative
	components and doubles positive components, runs an inverse fourier transform to return to
	time domain.
	'''
	
	x = np.asarray(x)
	N = x.size
	Xf = np.fft.fft(x, n=N)
	
	# Create single sided spectrum w mask `H` from double sided spectrum
	H = np.zeros(N, dtype=float)
	if N % 2 == 0:
		H[0] = 1.0
		H[N//2] = 1.0
		H[1:N//2] = 2.0
	else:
		H[0] = 1.0
		H[1:(N+1)//2] = 2.0
	Zf = Xf * H
	
	# 
	z = np.fft.ifft(Zf, n=N)
	return z

@dataclass
class HilbertResult:
	''' Dataclass to save results from `hilbert_phase_diagnostic` function.
	'''
	
	envelope: np.ndarray
	phase: np.ndarray
	inst_freq_hz: np.ndarray
	analytic: np.ndarray
	t: Optional[np.ndarray] = None

def hilbert_phase_diagnostic(v: np.ndarray, dt: float, center_freq_hz: Optional[float] = None, smooth_points: int = 0) -> HilbertResult:
	''' Calculates the envelope, phase, and inst. frequency from a real-valued signal. NOTE: The inst. frequency
	is based solely on ∆phase/∆t, so smoothing (set smoothing_points > 1) is recommended. 
	'''
	
	# Ensure v is a numpy array
	v = np.asarray(v, dtype=float)
	
	# Get the analytic version of the real valued signal v
	z = analytic_signal(v)
	
	# Envelope is the magnitude of the complex-valued signal
	env = np.abs(z)
	
	# Get phase from complex 
	phase = np.unwrap(np.angle(z))
	
	# discretized inst. frequency from dt
	inst_omega = np.gradient(phase, dt)
	
	# Convert to frequency
	inst_freq = inst_omega / (2.0 * np.pi)
	
	# Perform smoothing
	if smooth_points and smooth_points > 1:
		k = int(max(1, smooth_points))
		w = np.ones(k, dtype=float) / k
		inst_freq = np.convolve(inst_freq, w, mode='same')
	
	# Shift frequency as offset from center if requested
	if center_freq_hz is not None:
		inst_freq = inst_freq - float(center_freq_hz)
	
	# Create time list
	t = np.arange(v.size) * dt
	
	# Save results in dataclass
	return HilbertResult(envelope=env, phase=phase, inst_freq_hz=inst_freq, analytic=z, t=t)

@dataclass
class SpectrumResult:
	''' Dataclass to save results from `spectrum_probe`. '''
	
	freqs_hz: np.ndarray
	spec: np.ndarray
	scaling: Literal["magnitude","power","psd"]

def spectrum_probe(v: np.ndarray, dt: float, window: Literal["hann","rect"] = "hann", nfft: Optional[int] = None, onesided: bool = True, detrend: bool = True, scaling: Literal["magnitude","power","psd"] = "psd") -> SpectrumResult:
	''' Calculates the spectrum of a signal.
	
	Parameters:
		v (list): List of amplitude values
		dt (list): List of time values
		detrend (bool): Removes DC offset from input signal. Default = True
		nfft (int): Number of points to inlcude in FFT. If longer, will pad. Else will trim. (Defaults to length of series)
		window (str): `hann` or `rect`. Select window type for FFT. Default to `hann`.
		scaling (str): `magnitude`, `power`, or `psd`. Selects output scaling type. Defaults to `psd`.
		onesided (bool): Selects one vs two-sided FFT. Default is True.
		
	Returns:
		SpectrumResult: Dataclass containing the results.
	'''
	
	# Ensure input is a numpy array
	x = np.asarray(v, dtype=float)
	
	# Remove DC offset if requested
	if detrend:
		x = x - np.mean(x)
		
	# Get FFT length
	N = x.size
	if nfft is None:
		nfft = N
	
	# Select a windowing method
	if window == "hann":
		w = np.hanning(N)
	elif window == "rect":
		w = np.ones(N)
	else:
		raise ValueError("Unsupported window: %r" % window)
	
	# Apply window
	xw = x * w
	
	# Trim/lengthen time series until length == nfft
	if nfft != N:
		if nfft > N:
			xw = np.pad(xw, (0, nfft - N))
			w = np.pad(w, (0, nfft - N))
		else:
			xw = xw[:nfft]
			w = w[:nfft]
	
	# Get sample rate
	Fs = 1.0 / dt
	
	# Compute single or double ended FFT
	if onesided:
		V = np.fft.rfft(xw)
		freqs = np.fft.rfftfreq(nfft, d=dt) # real-valued fft
	else:
		V = np.fft.fft(xw)
		freqs = np.fft.fftfreq(nfft, d=dt) # complex(?) valued fft
	
	# Select signal output scaling
	if scaling == "magnitude":
		spec = np.abs(V)
	elif scaling == "power":
		spec = np.abs(V)**2
	elif scaling == "psd":
		U = (w[:N]**2).mean() if nfft == N else (w**2).mean()
		spec = (np.abs(V)**2) / (Fs * nfft * U)
	else:
		raise ValueError("Unsupported scaling: %r" % scaling)
	
	
	return SpectrumResult(freqs_hz=freqs, spec=spec, scaling=scaling)

def probe_fdtd_voltage(fdtd_result, x: Optional[float] = None, index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
	t = np.asarray(fdtd_result.t)
	if index is None:
		if x is None:
			raise ValueError("Provide either x or index for FDTD probe.")
		idx = int(np.argmin(np.abs(np.asarray(fdtd_result.x) - x)))
	else:
		idx = int(index)
	v_t = np.asarray(fdtd_result.v_xt)[:, idx]
	return t, v_t

# def probe_ladder_voltage(ladder_result, node: int) -> Tuple[np.ndarray, np.ndarray]:
# 	t = np.asarray(ladder_result.t)
# 	node = int(node)
# 	v_t = np.asarray(ladder_result.v_nodes)[:, node]
# 	return t, v_t

def time_gate(t: np.ndarray, x: np.ndarray, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
	t = np.asarray(t)
	x = np.asarray(x)
	m = (t >= t0) & (t <= t1)
	y = np.zeros_like(x)
	y[m] = x[m]
	return t, y

def plot_signal_diagnostics(t, v, dt, center_freq_hz=None, smooth_points=0, fmax=None):
	"""
	Make a 3-panel diagnostic plot:
	1. Time-domain waveform and envelope
	2. Instantaneous frequency (chirp)
	3. Spectrum (log scale)
	"""
	h = hilbert_phase_diagnostic(v, dt, center_freq_hz=center_freq_hz, smooth_points=smooth_points)
	spec = spectrum_probe(v, dt, window="hann", scaling="psd")

	fig, axes = plt.subplots(3, 1, figsize=(10, 8))
	ax1, ax2, ax3 = axes

	# Panel 1: waveform + envelope
	ax1.plot(t*1e9, v, label="waveform")
	ax1.plot(h.t*1e9, h.envelope, label="envelope", lw=2)
	ax1.set_xlabel("Time (ns)")
	ax1.set_ylabel("Voltage (V)")
	ax1.legend()
	ax1.set_title("Waveform and envelope")

	# Panel 2: instantaneous frequency
	ax2.plot(h.t*1e9, h.inst_freq_hz/1e9)
	ax2.set_xlabel("Time (ns)")
	ax2.set_ylabel("Inst. freq (GHz)")
	ax2.set_title("Instantaneous frequency (chirp)")

	# Panel 3: spectrum
	ax3.semilogy(spec.freqs_hz/1e9, spec.spec)
	ax3.set_xlabel("Frequency (GHz)")
	ax3.set_ylabel(spec.scaling)
	ax3.set_title("Spectrum")
	if fmax is not None:
		ax3.set_xlim(0, fmax/1e9)

	plt.tight_layout()
	return fig, axes