
"""
nltl_sim.py
-----------

Two simulators for a 1D nonlinear transmission line with series kinetic-inductance
nonlinearity (L depends on current) and linear shunt capacitance to ground:

1) Discrete ladder (SPICE-like) transient integrator
2) Continuum FDTD integrator for the nonlinear Telegrapher PDEs

Physics model (lossless core, optional resistive term only at terminations):

- Nonlinear inductor uses differential inductance L_d(I), so v_L = L_d(I) dI/dt.
  For kinetic inductance a simple polynomial is common:
	  L_d(I) = L0 * (1 + alpha * I^2)
  You may swap in other models (e.g., with a characteristic current I*).

- Telegrapher PDEs (lossless, nonlinear L only):
	  ∂v/∂x = - L_d(i) ∂i/∂t
	  ∂i/∂x = - C       ∂v/∂t

Boundary conditions here are implemented as simple resistive terminations (Rs on the left
to a source Vs(t), and RL on the right to ground). For the FDTD, boundary currents at the
edges are set by the termination relations; for the ladder, KCL includes Rs/RL currents.

Author: ChatGPT — PhD Theory project
"""

from dataclasses import dataclass
import numpy as np
from typing import Callable, Tuple, Optional

# -----------------------
# Nonlinearity (L_d(I))
# -----------------------

@dataclass
class KinInductance:
	"""Differential inductance model L_d(I) for kinetic inductance nonlinearity.

	Model: L_d(I) = L0 * (1 + alpha * I^2)
	"""
	L0: float       # base inductance (H or H/m)
	alpha: float    # nonlinearity coefficient (1/A^2)

	def Ld(self, I: np.ndarray) -> np.ndarray:
		return self.L0 * (1.0 + self.alpha * I**2)

	def min_Ld_over_range(self, Imax: float) -> float:
		"""Conservative min Ld in |I|<=Imax for CFL estimates (monotone↑ in this model -> min=L0)."""
		return self.L0


# ---------------------------------------------------------------
# 1) Discrete ladder (SPICE-like) transient for N lumped sections
# ---------------------------------------------------------------

@dataclass
class LadderParams:
	N: int                     # number of L-C sections
	L0: float                  # base inductance per section (H)
	alpha: float               # nonlinearity coefficient (1/A^2) for L_d(I)
	C: float                   # capacitance to ground per node (F)
	Rs: float                  # source resistance at left (Ohm). Use np.inf for open, 0 for stiff source
	RL: float                  # load resistance at right (Ohm). Use np.inf for open
	dt: float                  # time step (s)
	T: float                   # total time (s)
	Vs_func: Callable[[float], float]  # Vs(t) source (V) applied via Rs to node 0

@dataclass
class LadderResult:
	t: np.ndarray
	v_nodes: np.ndarray    # shape (Nt, N+1)
	i_L: np.ndarray        # shape (Nt, N) inductor currents (positive from node k to k+1)

class NLTLadder:
	"""Explicit leapfrog-like update for a nonlinear LC ladder with series nonlinear L and shunt C."""
	def __init__(self, p: LadderParams):
		self.p = p # Ladder parameter dict
		self.kin = KinInductance(L0=p.L0, alpha=p.alpha) # Kinetic inductance model
		self.Nt = int(np.round(p.T / p.dt)) + 1 # Number of time points

	def run(self) -> LadderResult:
		p = self.p
		N, dt, Nt = p.N, p.dt, self.Nt

		# State arrays
		v = np.zeros(N+1)               # node voltages at time n*dt
		iL_half = np.zeros(N)           # inductor currents at (n+1/2)*dt (staggered)
		v_hist = np.zeros((Nt, N+1))
		i_hist = np.zeros((Nt, N))

		# Helper for finite/inf resistors
		def g_of(R):
			return 0.0 if np.isinf(R) or R == 0 else 1.0/R

		gRs = 0.0 if np.isinf(p.Rs) else (np.inf if p.Rs == 0 else 1.0/p.Rs)
		gRL = g_of(p.RL)

		# Scan over all time points
		for n in range(Nt):
			
			# Get new absolute time
			t_n = n * dt
			
			# Get stimulus voltage, as defined in the ladder parameters struct
			Vs = p.Vs_func(t_n)

			# 1) Update inductor currents (iL) to half-step using node voltages at full-step
			#    di/dt = (v_k - v_{k+1}) / L_d(i)
			#    i^{n+1/2} = i^{n-1/2} + dt * (v_k^n - v_{k+1}^n) / Ld(i^{n-1/2})
			Ld = self.kin.Ld(iL_half)
			dv = v[:-1] - v[1:] # Get voltage delta for each step
			iL_half = iL_half + dt * dv / Ld

			# 2) Update node voltages using KCL at nodes with inductor currents at half-steps
			#    C dv/dt = i_in - i_out - g*v (no shunt G here; only terminations)
			# Internal nodes j=1..N-1:
			#    C dv_j/dt = iL_{j-1} - iL_{j}
			dvdt = np.zeros_like(v)
			if N > 1:
				dvdt[1:-1] = (iL_half[:-1] - iL_half[1:]) / p.C
			
			# Left boundary j=0: include source via Rs (or stiff source if Rs==0)
			if p.Rs == 0:
				v[0] = Vs  # stiff Dirichlet source
			else:
				i_src = 0.0 if np.isinf(p.Rs) else (Vs - v[0]) / p.Rs
				dvdt[0] = (i_src - iL_half[0]) / p.C
			
			# Right boundary j=N: include RL
			i_load = 0.0 if np.isinf(p.RL) else v[-1] / p.RL
			dvdt[-1] = (iL_half[-1] - i_load) / p.C
			
			v = v + dt * dvdt
			
			v_hist[n, :] = v
			i_hist[n, :] = iL_half

		t = np.arange(Nt) * dt
		return LadderResult(t=t, v_nodes=v_hist, i_L=i_hist)


# ------------------------------------------------------
# 2) Continuum FDTD for nonlinear Telegrapher PDE system
# ------------------------------------------------------

@dataclass
class FDTDParams:
	Nx: int                    # number of spatial cells
	L: float                   # line length (m)
	L0_per_m: float            # base inductance per meter (H/m)
	alpha: float               # nonlinearity coefficient (1/A^2)
	C_per_m: float             # capacitance per meter (F/m)
	dt: float                  # time step (s)
	T: float                   # total time (s)
	Rs: float                  # source resistance at x=0 (Ohm). 0 => stiff source
	RL: float                  # load resistance at x=L (Ohm). inf => open
	Vs_func: Callable[[float], float]  # source voltage Vs(t) at x=0

@dataclass
class FDTDResult:
	t: np.ndarray
	x: np.ndarray
	v_xt: np.ndarray      # shape (Nt, Nx+1) voltages at nodes x_k
	i_xt: np.ndarray      # shape (Nt, Nx) currents on half-cells x_{k+1/2}

class NLTFDTD:
	"""Leapfrog FDTD for nonlinear telegrapher equations with kinetic inductance nonlinearity."""
	def __init__(self, p: FDTDParams):
		self.p = p
		self.dx = p.L / p.Nx
		self.Nt = int(np.round(p.T / p.dt)) + 1
		self.kin = KinInductance(L0=p.L0_per_m, alpha=p.alpha)

	@staticmethod
	def cfl_dt(dx: float, Ld_min: float, C: float, safety: float = 0.9) -> float:
		"""Return a stable-ish dt ~ safety * dx / v_ph, where v_ph = 1/sqrt(L*C)."""
		v = 1.0 / np.sqrt(Ld_min * C)
		return safety * dx / v

	def run(self) -> FDTDResult:
		p = self.p
		Nx, dt, Nt = p.Nx, p.dt, self.Nt
		dx = self.dx
		
		# Staggering: v at integer nodes [0..Nx], i at half nodes [0..Nx-1]
		v = np.zeros(Nx+1)            # v_k^n
		i_half = np.zeros(Nx)         # i_{k+1/2}^{n+1/2}
		v_hist = np.zeros((Nt, Nx+1))
		i_hist = np.zeros((Nt, Nx))

		for n in range(Nt):
			t_n = n * dt
			
			# Call function returning voltage stimulus at this time point. 
			Vs = p.Vs_func(t_n)

			# 1) Update i at half step using spatial derivative of v at time n
			#    ∂i/∂t = - (1/Ld(i)) ∂v/∂x
			dv_dx = (v[1:] - v[:-1]) / dx  # length Nx - finds ∆v / ∆x for each step.
			Ld = self.kin.Ld(i_half) # Find differential inductance
			i_half = i_half - dt * dv_dx / Ld # Update i at half step - just i_old-∆i/∆t*∆t
			
			# 2) Boundary currents from resistive terminations
			# Left boundary half-cell current i_{1/2}: set by source resistor
			if p.Rs == 0:
				# Stiff voltage source: force v[0] after v update
				i_left = None  # not used in KCL; we'll set v[0] explicitly
			else:
				i_left = (Vs - v[0]) / p.Rs  # current entering the line at left

			# Right boundary half-cell current i_{Nx-1/2} to load
			i_right = 0.0 if np.isinf(p.RL) else v[-1] / p.RL

			# 3) Update v at full step using spatial derivative of i at n+1/2
			#    ∂v/∂t = - (1/C) ∂i/∂x
			di_dx = np.zeros_like(v)
			# interior nodes: (i_{k+1/2} - i_{k-1/2})/dx
			if Nx > 1:
				di_dx[1:-1] = (i_half[1:] - i_half[:-1]) / dx

			# left boundary: use i_left as i_{-1/2}
			if p.Rs == 0:
				# Stiff source set after update; use zero-flux approximation for update then fix
				di_dx[0] = (i_half[0] - 0.0) / dx
			else:
				di_dx[0] = (i_half[0] - i_left) / dx

			# right boundary: use i_right as i_{Nx+1/2}= - i to ground (sign handled by KCL form here)
			di_dx[-1] = (i_right - i_half[-1]) / dx

			v = v - dt * di_dx / p.C_per_m

			# Enforce stiff source if Rs == 0
			if p.Rs == 0:
				v[0] = Vs

			v_hist[n, :] = v
			i_hist[n, :] = i_half

		t = np.arange(Nt) * dt
		x = np.linspace(0.0, p.L, Nx+1)
		return FDTDResult(t=t, x=x, v_xt=v_hist, i_xt=i_hist)


# -----------------
# Utility waveforms
# -----------------

def gaussian_pulse(t: float, t0: float, sigma: float, V0: float) -> float:
	"""Temporal Gaussian pulse centered at t0 with RMS width sigma and peak V0."""
	return V0 * np.exp(-0.5*((t - t0)/sigma)**2)

def raised_cosine_step(t: float, t0: float, rise: float, V1: float) -> float:
	"""Smooth step from 0 to V1 starting around t0 with rise time 'rise' (0-100% approx)."""
	if t < t0 - 0.5*rise:
		return 0.0
	if t > t0 + 0.5*rise:
		return V1
	# Half-cosine ramp
	phase = (t - (t0 - 0.5*rise)) / rise * np.pi
	return 0.5*V1*(1 - np.cos(phase))


# -----------------
# Convenience demos
# -----------------

def demo_ladder() -> Tuple[LadderResult, LadderParams]:
	# Example: 200 sections, Z0 ~ sqrt(L0/C), Gaussian pulse source
	N = 200
	L0 = 0.5e-9       # H per section
	C = 0.5e-12       # F per node
	alpha = 5.0e-3    # 1/A^2 — tweak to taste
	Z0 = np.sqrt(L0/C)
	Rs = Z0           # well-matched source
	RL = Z0           # matched load

	dt = 2.0e-12      # s (ensure stability; explicit scheme)
	T  = 3.0e-9       # s
	t0 = 0.6e-9
	sigma = 0.15e-9
	V0 = 0.5

	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)

	p = LadderParams(N=N, L0=L0, alpha=alpha, C=C, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs)
	sim = NLTLadder(p)
	out = sim.run()
	return out, p

def demo_fdtd() -> Tuple[FDTDResult, FDTDParams]:
	Nx = 400
	L = 0.2            # meters
	L0m = 400e-9       # H/m
	Cm  = 160e-12      # F/m => Z0 ~ sqrt(L/C) ~ 50 Ohm
	alpha = 5.0e-3     # 1/A^2

	kin = KinInductance(L0=L0m, alpha=alpha)
	dx = L / Nx
	Ld_min = kin.min_Ld_over_range(Imax=0.5)  # conservative
	dt_cfl = NLTFDTD.cfl_dt(dx, Ld_min, Cm, safety=0.9)

	dt = dt_cfl
	T  = 3.0e-9
	Rs = 50.0
	RL = 50.0

	t0 = 0.6e-9
	sigma = 0.15e-9
	V0 = 0.5
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)

	p = FDTDParams(Nx=Nx, L=L, L0_per_m=L0m, alpha=alpha, C_per_m=Cm, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs)
	sim = NLTFDTD(p)
	out = sim.run()
	return out, p

def demo_ladder_v2() -> Tuple[LadderResult, LadderParams]:
	# Example: 200 sections, Z0 ~ sqrt(L0/C), Gaussian pulse source
	N = 400
	L0 = 0.2e-9       # H per section
	C = 0.08e-12       # F per node
	alpha = 5.0e-3    # 1/A^2 — tweak to taste
	Z0 = np.sqrt(L0/C)
	Rs = Z0           # well-matched source
	RL = Z0           # matched load

	dt = 2.0e-12      # s (ensure stability; explicit scheme)
	T  = 3.0e-9       # s
	t0 = 0.6e-9
	sigma = 0.15e-9
	V0 = 0.5

	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)

	p = LadderParams(N=N, L0=L0, alpha=alpha, C=C, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs)
	sim = NLTLadder(p)
	out = sim.run()
	return out, p

def demo_fdtd_v2() -> Tuple[FDTDResult, FDTDParams]:
	Nx = 400
	L = 0.2            # meters
	L0m = 400e-9       # H/m
	Cm  = 160e-12      # F/m => Z0 ~ sqrt(L/C) ~ 50 Ohm
	alpha = 5.0e-3     # 1/A^2

	kin = KinInductance(L0=L0m, alpha=alpha)
	dx = L / Nx
	Ld_min = kin.min_Ld_over_range(Imax=0.5)  # conservative
	dt_cfl = NLTFDTD.cfl_dt(dx, Ld_min, Cm, safety=0.9)

	dt = dt_cfl
	T  = 3.0e-9
	Rs = 50.0
	RL = 50.0

	t0 = 0.6e-9
	sigma = 0.15e-9
	V0 = 0.5
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)

	p = FDTDParams(Nx=Nx, L=L, L0_per_m=L0m, alpha=alpha, C_per_m=Cm, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs)
	sim = NLTFDTD(p)
	out = sim.run()
	return out, p

if __name__ == "__main__":
	# Quick smoke test when run directly (no plotting here).
	out_lad, p_lad = demo_ladder()
	out_fdtd, p_fdtd = demo_fdtd()
	print("Ladder sim:", out_lad.v_nodes.shape, out_lad.i_L.shape)
	print("FDTD sim:  ", out_fdtd.v_xt.shape, out_fdtd.i_xt.shape)
