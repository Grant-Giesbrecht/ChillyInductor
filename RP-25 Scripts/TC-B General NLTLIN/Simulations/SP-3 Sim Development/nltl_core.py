
"""
nltl_sim.py
-----------

Nonlinear transmission line simulators with **explicit** or **implicit** (local Newton) updates
and **piecewise spatial variation** of per-unit-length parameters.

Includes:
- Uniform ladder (NLTLadder) and FDTD (NLTFDTD)
- Piecewise ladder (NLTLadderPW) and FDTD (NLTFDTD_PW)
- Demos for both uniform and piecewise cases

Implicit option: set `nonlinear_update="implicit"` in the Params. This performs a **local**
Newton solve per inductor/half-cell per time step for the stiff nonlinearity, while voltage
updates remain explicit (semi-implicit scheme).

Author: ChatGPT — PhD Theory project
"""

from dataclasses import dataclass
import numpy as np
from typing import Callable, List, Literal, Tuple, Optional

# -----------------------
# Nonlinearity (L_d(I))
# -----------------------

@dataclass
class KinInductance:
	"""Differential inductance model L_d(I) for kinetic inductance nonlinearity.

	Default model: L_d(I) = L0 * (1 + alpha * I^2)
	"""
	L0: float       # base inductance (H or H/m)
	alpha: float    # nonlinearity coefficient (1/A^2)

	def Ld(self, I: np.ndarray) -> np.ndarray:
		return self.L0 * (1.0 + self.alpha * I**2)

	def min_Ld_over_range(self, Imax: float) -> float:
		return self.L0


# ------------------
# Newton Utilities
# ------------------

# TODO: Explain
def _newton_i_update(i0: np.ndarray, s: np.ndarray, L0: np.ndarray, alpha: np.ndarray,
					 max_iter: int = 15, tol: float = 1e-12) -> np.ndarray:
	"""Solve per-element for i:  F(i) = i - i0 - s / (L0 * (1 + alpha*i^2)) = 0.

	Parameters
	----------
	i0 : array
		Previous current (same shape as s).
	s : array
		s = dt * Δv  (ladder)   OR   s = - dt * (∂v/∂x) (FDTD)
	L0, alpha : arrays
		Base inductance and nonlinearity per element (section or half-cell).
	"""
	# Initial guess: explicit update
	i = i0 + s / (L0 * (1.0 + alpha * i0**2))

	for _ in range(max_iter):
		denom = (1.0 + alpha * i**2)
		invLd = 1.0 / (L0 * denom)
		F = i - i0 - s * invLd
		# d(1/Ld)/di = -(1/L0) * (2*alpha*i) / (1 + alpha*i^2)^2
		dinvLd_di = -(1.0 / L0) * (2.0 * alpha * i) / (denom**2 + 1e-300)
		dF = 1.0 - s * dinvLd_di
		step = F / (dF + 1e-300)
		i_new = i - step
		if np.max(np.abs(step)) < tol:
			return i_new
		i = i_new
	return i

#NOTE: Previously LadderParams
@dataclass
class LumpedElementParams:
	N: int
	L0: float
	alpha: float
	C: float
	Rs: float
	RL: float
	dt: float
	T: float
	Vs_func: Callable[[float], float]
	nonlinear_update: Literal["explicit","implicit"] = "explicit"

#NOTE: Previously LadderResult
@dataclass
class LumpedElementResult:
	t: np.ndarray
	v_nodes: np.ndarray    # (Nt, N+1)
	i_L: np.ndarray        # (Nt, N)
	
	#NOTE: Was probe_ladder_voltage()
	def probe_voltage(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
		''' Gets the waveform at the specified node.
		
		Params:
			node (int): Node to probe
			
		Returns:
			tuple: (time, voltage) of the waveform at the node.
		'''
		
		t = np.asarray(self.t)
		node = int(node)
		v_t = np.asarray(self.v_nodes)[:, node]
		return t, v_t

# class NLTLadder:
# 	def __init__(self, p: LumpedElementParams):
# 		self.p = p
# 		self.kin = KinInductance(L0=p.L0, alpha=p.alpha)
# 		self.Nt = int(np.round(p.T / p.dt)) + 1

# 	def run(self) -> LumpedElementResult:
# 		p = self.p
# 		N, dt, Nt = p.N, p.dt, self.Nt

# 		v = np.zeros(N+1)
# 		iL_half = np.zeros(N)
# 		v_hist = np.zeros((Nt, N+1))
# 		i_hist = np.zeros((Nt, N))

# 		for n in range(Nt):
# 			t_n = n * dt
# 			Vs = p.Vs_func(t_n)

# 			# Inductor update
# 			dv = v[:-1] - v[1:]
# 			if p.nonlinear_update == "explicit":
# 				Ld = self.kin.Ld(iL_half)
# 				iL_half = iL_half + dt * dv / Ld
# 			else:
# 				L0_arr = np.full_like(iL_half, self.kin.L0)
# 				alpha_arr = np.full_like(iL_half, self.kin.alpha)
# 				s = dt * dv
# 				iL_half = _newton_i_update(iL_half, s, L0_arr, alpha_arr)

# 			# Node voltage updates
# 			dvdt = np.zeros_like(v)
# 			if p.Rs == 0:
# 				v[0] = Vs
# 			else:
# 				i_src = 0.0 if np.isinf(p.Rs) else (Vs - v[0]) / p.Rs
# 				dvdt[0] = (i_src - iL_half[0]) / p.C

# 			if N > 1:
# 				dvdt[1:-1] = (iL_half[:-1] - iL_half[1:]) / p.C

# 			i_load = 0.0 if np.isinf(p.RL) else v[-1] / p.RL
# 			dvdt[-1] = (iL_half[-1] - i_load) / p.C

# 			v = v + dt * dvdt

# 			v_hist[n, :] = v
# 			i_hist[n, :] = iL_half

# 		t = np.arange(Nt) * dt
# 		return LumpedElementResult(t=t, v_nodes=v_hist, i_L=i_hist)


#NOTE: Previously FDTDParams
@dataclass
class FiniteDiffParams:
	Nx: int
	L: float
	L0_per_m: float
	alpha: float
	C_per_m: float
	dt: float
	T: float
	Rs: float
	RL: float
	Vs_func: Callable[[float], float]
	nonlinear_update: Literal["explicit","implicit"] = "explicit"

#NOTE: Previously FDTDResult
@dataclass
class FiniteDiffResult:
	t: np.ndarray
	x: np.ndarray
	v_xt: np.ndarray   # (Nt, Nx+1)
	i_xt: np.ndarray   # (Nt, Nx)
	
	# Note: Was probe_fdtd_voltage()
	def probe_voltage(self, x: Optional[float] = None, index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
		''' Returns the waveform at the specified x position or index.
		
		Params:
			x (float): Optional, x position to probe.
			index (int): Optional, index to probe. Must specify `x` or `index`.
		
		Returns:
			(tuple): (time, voltage) of waveform at probe location
		'''
		
		t = np.asarray(self.t)
		if index is None:
			if x is None:
				raise ValueError("Provide either x or index for FDTD probe.")
			idx = int(np.argmin(np.abs(np.asarray(self.x) - x)))
		else:
			idx = int(index)
		v_t = np.asarray(self.v_xt)[:, idx]
		return t, v_t
	
# class NLTFDTD:
# 	def __init__(self, p: FiniteDiffParams):
# 		self.p = p
# 		self.dx = p.L / p.Nx
# 		self.Nt = int(np.round(p.T / p.dt)) + 1
# 		self.kin = KinInductance(L0=p.L0_per_m, alpha=p.alpha)

# 	@staticmethod
# 	def cfl_dt(dx: float, Ld_min: float, C: float, safety: float = 0.9) -> float:
# 		v = 1.0 / np.sqrt(Ld_min * C)
# 		return safety * dx / v

# 	def run(self) -> FiniteDiffResult:
# 		p = self.p
# 		Nx, dt, Nt = p.Nx, p.dt, self.Nt
# 		dx = self.dx

# 		v = np.zeros(Nx+1)
# 		i_half = np.zeros(Nx)
# 		v_hist = np.zeros((Nt, Nx+1))
# 		i_hist = np.zeros((Nt, Nx))

# 		for n in range(Nt):
# 			t_n = n * dt
# 			Vs = p.Vs_func(t_n)

# 			# i update at half-step
# 			dv_dx = (v[1:] - v[:-1]) / dx
# 			if p.nonlinear_update == "explicit":
# 				Ld = self.kin.Ld(i_half)
# 				i_half = i_half - dt * dv_dx / Ld
# 			else:
# 				L0_arr = np.full_like(i_half, self.kin.L0)
# 				alpha_arr = np.full_like(i_half, self.kin.alpha)
# 				s = - dt * dv_dx
# 				i_half = _newton_i_update(i_half, s, L0_arr, alpha_arr)

# 			# boundary currents
# 			if p.Rs == 0:
# 				i_left = None
# 			else:
# 				i_left = (Vs - v[0]) / p.Rs
# 			i_right = 0.0 if np.isinf(p.RL) else v[-1] / p.RL

# 			# v update at full-step
# 			di_dx = np.zeros_like(v)
# 			if Nx > 1:
# 				di_dx[1:-1] = (i_half[1:] - i_half[:-1]) / dx
# 			di_dx[0]  = (i_half[0] - (0.0 if p.Rs==0 else i_left)) / dx
# 			di_dx[-1] = (i_right - i_half[-1]) / dx
# 			v = v - dt * di_dx / p.C_per_m

# 			if p.Rs == 0:
# 				v[0] = Vs

# 			v_hist[n, :] = v
# 			i_hist[n, :] = i_half

# 		t = np.arange(Nt) * dt
# 		x = np.linspace(0.0, p.L, Nx+1)
# 		return FiniteDiffResult(t=t, x=x, v_xt=v_hist, i_xt=i_hist)

@dataclass
class LadderRegion:
	x0: float
	x1: float
	L0_per_m: float
	C_per_m: float
	alpha: float

#NOTE: Previously LumpedElementParamsPW
@dataclass
class LumpedElementParams:
	N: int
	L: float
	Rs: float
	RL: float
	dt: float
	T: float
	Vs_func: Callable[[float], float]
	regions: List[LadderRegion]
	nonlinear_update: Literal["explicit","implicit"] = "explicit"

@dataclass
class FDTDRegion:
	x0: float
	x1: float
	L0_per_m: float
	C_per_m: float
	alpha: float

#NOTE: Previously FiniteDiffParamsPW
@dataclass
class FiniteDiffParams:
	Nx: int
	L: float
	dt: float
	T: float
	Rs: float
	RL: float
	Vs_func: Callable[[float], float]
	regions: List[FDTDRegion]
	nonlinear_update: Literal["explicit","implicit"] = "explicit"

def _sample_regions_on_grid(regions: List, grid: np.ndarray, field: str) -> np.ndarray:
	vals = np.zeros_like(grid, dtype=float)
	for r in regions:
		mask = (grid >= r.x0) & (grid < r.x1)
		vals[mask] = getattr(r, field)
	# ensure last grid point gets last region's value
	end = max(r.x1 for r in regions)
	if np.isclose(grid[-1], end):
		for r in regions:
			if np.isclose(end, r.x1):
				vals[-1] = getattr(r, field)
	return vals

# NOTE: Previously called NLTLadderPW
class LumpedElementSim:
	''' Simulator for L-C ladder based non-linear transmission line. '''
	
	def __init__(self, p: LumpedElementParams):
		self.p = p
		self.Nt = int(np.round(p.T / p.dt)) + 1
		self.dx = p.L / p.N
		x_sec = (np.arange(p.N) + 0.5) * self.dx
		x_nodes = np.arange(p.N + 1) * self.dx
		self.L0_sec = _sample_regions_on_grid(p.regions, x_sec, 'L0_per_m') * self.dx
		self.C_nodes = _sample_regions_on_grid(p.regions, x_nodes, 'C_per_m') * self.dx
		self.alpha_sec = _sample_regions_on_grid(p.regions, x_sec, 'alpha')

	def run(self) -> LumpedElementResult:
		p = self.p
		N, dt, Nt = p.N, p.dt, self.Nt

		v = np.zeros(N+1)
		iL_half = np.zeros(N)
		v_hist = np.zeros((Nt, N+1))
		i_hist = np.zeros((Nt, N))

		for n in range(Nt):
			t_n = n * dt
			Vs = p.Vs_func(t_n)

			dv = v[:-1] - v[1:]
			if p.nonlinear_update == "explicit":
				Ld = self.L0_sec * (1.0 + self.alpha_sec * iL_half**2)
				iL_half = iL_half + dt * dv / Ld
			else:
				s = dt * dv
				iL_half = _newton_i_update(iL_half, s, self.L0_sec, self.alpha_sec)

			# Node updates
			dvdt = np.zeros_like(v)
			if p.Rs == 0:
				v[0] = Vs
			else:
				i_src = 0.0 if np.isinf(p.Rs) else (Vs - v[0]) / p.Rs
				dvdt[0] = (i_src - iL_half[0]) / self.C_nodes[0]

			if N > 1:
				dvdt[1:-1] = (iL_half[:-1] - iL_half[1:]) / self.C_nodes[1:-1]

			i_load = 0.0 if np.isinf(p.RL) else v[-1] / p.RL
			dvdt[-1] = (iL_half[-1] - i_load) / self.C_nodes[-1]

			v = v + dt * dvdt

			v_hist[n, :] = v
			i_hist[n, :] = iL_half

		t = np.arange(Nt) * dt
		return LumpedElementResult(t=t, v_nodes=v_hist, i_L=i_hist)


#NOTE: Previously NLTFDTD_PW
class FiniteDiffSim:
	def __init__(self, p: FiniteDiffParams):
		self.p = p
		self.dx = p.L / p.Nx
		self.Nt = int(np.round(p.T / p.dt)) + 1
		x_nodes = np.linspace(0.0, p.L, p.Nx + 1)
		x_half  = (np.arange(p.Nx) + 0.5) * self.dx
		self.C_nodes = _sample_regions_on_grid(p.regions, x_nodes, 'C_per_m')
		self.L0_half = _sample_regions_on_grid(p.regions, x_half,  'L0_per_m')
		self.alpha_half = _sample_regions_on_grid(p.regions, x_half, 'alpha')

	@staticmethod
	def cfl_dt(dx: float, Lmin: float, Cmin: float, safety: float = 0.9) -> float:
		v = 1.0 / np.sqrt(max(Lmin, 1e-300) * max(Cmin, 1e-300))
		return safety * dx / v

	def run(self) -> FiniteDiffResult:
		p = self.p
		Nx, dt, Nt = p.Nx, p.dt, self.Nt
		dx = self.dx

		v = np.zeros(Nx+1)
		i_half = np.zeros(Nx)
		v_hist = np.zeros((Nt, Nx+1))
		i_hist = np.zeros((Nt, Nx))

		for n in range(Nt):
			t_n = n * dt
			Vs = p.Vs_func(t_n)

			dv_dx = (v[1:] - v[:-1]) / dx
			if p.nonlinear_update == "explicit":
				Ld_half = self.L0_half * (1.0 + self.alpha_half * i_half**2)
				i_half = i_half - dt * dv_dx / Ld_half
			else:
				s = - dt * dv_dx
				i_half = _newton_i_update(i_half, s, self.L0_half, self.alpha_half)

			# boundaries
			if p.Rs == 0:
				i_left = None
			else:
				i_left = (Vs - v[0]) / p.Rs
			i_right = 0.0 if np.isinf(p.RL) else v[-1] / p.RL

			di_dx = np.zeros_like(v)
			if Nx > 1:
				di_dx[1:-1] = (i_half[1:] - i_half[:-1]) / dx
			di_dx[0]  = (i_half[0] - (0.0 if p.Rs==0 else i_left)) / dx
			di_dx[-1] = (i_right - i_half[-1]) / dx

			v = v - dt * (di_dx / self.C_nodes)

			if p.Rs == 0:
				v[0] = Vs

			v_hist[n, :] = v
			i_hist[n, :] = i_half

		t = np.arange(Nt) * dt
		x = np.linspace(0.0, p.L, Nx+1)
		return FiniteDiffResult(t=t, x=x, v_xt=v_hist, i_xt=i_hist)


# ----------------- Waveform Helpers ----------------------------

def gaussian_pulse(t:float, t0:float, sigma:float, V0:float) -> float:
	return V0 * np.exp(-0.5*((t - t0)/sigma)**2)

def raised_cosine_step(t:float, t0:float, rise:float, V1:float=1) -> float:
	''' Function for helping to create input stimuli. 
	
	Parameters:
		t (float): Time point to evaluate
		t0 (float): Center of ramp up time
		rise (float): Rise time
		v1 (float): High voltage. 
	'''
	
	if t < t0 - 0.5*rise:
		return 0.0
	if t > t0 + 0.5*rise:
		return V1
	phase = (t - (t0 - 0.5*rise)) / rise * np.pi
	return 0.5*V1*(1 - np.cos(phase))

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


def time_gate(t: np.ndarray, x: np.ndarray, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
	''' Masks a time list s.t. it returns only the waveform at times between t0
	and t1, and returns 0 for everything else. 
	'''
	
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