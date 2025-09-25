
"""
nltl_sim.py
-----------

Adds piecewise spatial variation for L0, C, and nonlinearity alpha for BOTH solvers:
 - NLTLadderPW: ladder with per-section L and C derived from per-unit-length profiles
 - NLTFDTD_PW: continuum FDTD with spatially varying L0_per_m(x), C_per_m(x), alpha(x)

You can still use the original NLTLadder / NLTFDTD (uniform parameters) and demos.
New "piecewise" demos are provided at the bottom.
"""

from dataclasses import dataclass
import numpy as np
from typing import Callable, Tuple, Optional, List

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


# ---------------------------------------------------------------
# 1) Discrete ladder (uniform) — original
# ---------------------------------------------------------------

@dataclass
class LadderParams:
	N: int
	L0: float
	alpha: float
	C: float
	Rs: float
	RL: float
	dt: float
	T: float
	Vs_func: Callable[[float], float]

@dataclass
class LadderResult:
	t: np.ndarray
	v_nodes: np.ndarray    # (Nt, N+1)
	i_L: np.ndarray        # (Nt, N)

class NLTLadder:
	def __init__(self, p: LadderParams):
		self.p = p
		self.kin = KinInductance(L0=p.L0, alpha=p.alpha)
		self.Nt = int(np.round(p.T / p.dt)) + 1

	def run(self) -> LadderResult:
		p = self.p
		N, dt, Nt = p.N, p.dt, self.Nt
		v = np.zeros(N+1)
		iL_half = np.zeros(N)
		v_hist = np.zeros((Nt, N+1))
		i_hist = np.zeros((Nt, N))

		for n in range(Nt):
			t_n = n * dt
			Vs = p.Vs_func(t_n)

			# Inductor update (nonlinear, explicit)
			Ld = self.kin.Ld(iL_half)
			dv = v[:-1] - v[1:]
			iL_half = iL_half + dt * dv / Ld

			# Node voltage updates
			dvdt = np.zeros_like(v)
			if p.Rs == 0:
				v[0] = Vs
			else:
				i_src = 0.0 if np.isinf(p.Rs) else (Vs - v[0]) / p.Rs
				dvdt[0] = (i_src - iL_half[0]) / p.C

			if N > 1:
				dvdt[1:-1] = (iL_half[:-1] - iL_half[1:]) / p.C

			i_load = 0.0 if np.isinf(p.RL) else v[-1] / p.RL
			dvdt[-1] = (iL_half[-1] - i_load) / p.C

			v = v + dt * dvdt

			v_hist[n, :] = v
			i_hist[n, :] = iL_half

		t = np.arange(Nt) * dt
		return LadderResult(t=t, v_nodes=v_hist, i_L=i_hist)


# ------------------------------------------------------
# 2) Continuum FDTD (uniform) — original
# ------------------------------------------------------

@dataclass
class FDTDParams:
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

@dataclass
class FDTDResult:
	t: np.ndarray
	x: np.ndarray
	v_xt: np.ndarray   # (Nt, Nx+1)
	i_xt: np.ndarray   # (Nt, Nx)

class NLTFDTD:
	def __init__(self, p: FDTDParams):
		self.p = p
		self.dx = p.L / p.Nx
		self.Nt = int(np.round(p.T / p.dt)) + 1
		self.kin = KinInductance(L0=p.L0_per_m, alpha=p.alpha)

	@staticmethod
	def cfl_dt(dx: float, Ld_min: float, C: float, safety: float = 0.9) -> float:
		v = 1.0 / np.sqrt(Ld_min * C)
		return safety * dx / v

	def run(self) -> FDTDResult:
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

			# i update at half-step
			dv_dx = (v[1:] - v[:-1]) / dx
			Ld = self.kin.Ld(i_half)
			i_half = i_half - dt * dv_dx / Ld

			# boundary currents
			if p.Rs == 0:
				i_left = None
			else:
				i_left = (Vs - v[0]) / p.Rs
			i_right = 0.0 if np.isinf(p.RL) else v[-1] / p.RL

			# v update at full-step
			di_dx = np.zeros_like(v)
			if Nx > 1:
				di_dx[1:-1] = (i_half[1:] - i_half[:-1]) / dx
			di_dx[0]  = (i_half[0] - (0.0 if p.Rs==0 else i_left)) / dx
			di_dx[-1] = (i_right - i_half[-1]) / dx
			v = v - dt * di_dx / p.C_per_m

			if p.Rs == 0:
				v[0] = Vs

			v_hist[n, :] = v
			i_hist[n, :] = i_half

		t = np.arange(Nt) * dt
		x = np.linspace(0.0, p.L, Nx+1)
		return FDTDResult(t=t, x=x, v_xt=v_hist, i_xt=i_hist)


# --------------------------------------------------------------------
# Piecewise spatial variation support (new)
# --------------------------------------------------------------------

@dataclass
class LadderRegion:
	x0: float
	x1: float
	L0_per_m: float
	C_per_m: float
	alpha: float

@dataclass
class LadderParamsPW:
	N: int
	L: float
	Rs: float
	RL: float
	dt: float
	T: float
	Vs_func: Callable[[float], float]
	regions: List[LadderRegion]

@dataclass
class FDTDRegion:
	x0: float
	x1: float
	L0_per_m: float
	C_per_m: float
	alpha: float

@dataclass
class FDTDParamsPW:
	Nx: int
	L: float
	dt: float
	T: float
	Rs: float
	RL: float
	Vs_func: Callable[[float], float]
	regions: List[FDTDRegion]

def _sample_regions_on_grid(regions: List, grid: np.ndarray, field: str) -> np.ndarray:
	vals = np.zeros_like(grid, dtype=float)
	for r in regions:
		mask = (grid >= r.x0) & (grid < r.x1)
		vals[mask] = getattr(r, field)
	if np.isclose(grid[-1], max(r.x1 for r in regions)):
		for r in regions:
			if np.isclose(grid[-1], r.x1):
				vals[-1] = getattr(r, field)
	return vals

class NLTLadderPW:
	def __init__(self, p: LadderParamsPW):
		self.p = p
		self.Nt = int(np.round(p.T / p.dt)) + 1
		self.dx = p.L / p.N
		x_sec = (np.arange(p.N) + 0.5) * self.dx
		x_nodes = np.arange(p.N + 1) * self.dx
		self.L0_sec = _sample_regions_on_grid(p.regions, x_sec, 'L0_per_m') * self.dx
		self.C_nodes = _sample_regions_on_grid(p.regions, x_nodes, 'C_per_m') * self.dx
		self.alpha_sec = _sample_regions_on_grid(p.regions, x_sec, 'alpha')

	def run(self) -> LadderResult:
		p = self.p
		N, dt, Nt = p.N, p.dt, self.Nt
		v = np.zeros(N+1)
		iL_half = np.zeros(N)
		v_hist = np.zeros((Nt, N+1))
		i_hist = np.zeros((Nt, N))

		for n in range(Nt):
			t_n = n * dt
			Vs = p.Vs_func(t_n)

			Ld = self.L0_sec * (1.0 + self.alpha_sec * iL_half**2)
			dv = v[:-1] - v[1:]
			iL_half = iL_half + dt * dv / Ld

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
		return LadderResult(t=t, v_nodes=v_hist, i_L=i_hist)


class NLTFDTD_PW:
	def __init__(self, p: FDTDParamsPW):
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

	def run(self) -> FDTDResult:
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
			Ld_half = self.L0_half * (1.0 + self.alpha_half * i_half**2)
			i_half = i_half - dt * dv_dx / Ld_half

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
		return FDTDResult(t=t, x=x, v_xt=v_hist, i_xt=i_hist)


# -----------------
# Utility waveforms
# -----------------

def gaussian_pulse(t: float, t0: float, sigma: float, V0: float) -> float:
	return V0 * np.exp(-0.5*((t - t0)/sigma)**2)

def raised_cosine_step(t: float, t0: float, rise: float, V1: float) -> float:
	if t < t0 - 0.5*rise:
		return 0.0
	if t > t0 + 0.5*rise:
		return V1
	phase = (t - (t0 - 0.5*rise)) / rise * np.pi
	return 0.5*V1*(1 - np.cos(phase))


# -----------------
# Uniform demos
# -----------------

def demo_ladder():
	N = 200
	L0 = 0.5e-9
	C = 0.5e-12
	alpha = 5.0e-3
	Z0 = np.sqrt(L0/C)
	Rs = Z0
	RL = Z0
	dt = 2.0e-12
	T  = 3.0e-9
	t0 = 0.6e-9
	sigma = 0.15e-9
	V0 = 0.5
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)
	p = LadderParams(N=N, L0=L0, alpha=alpha, C=C, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs)
	out = NLTLadder(p).run()
	return out, p

def demo_fdtd():
	Nx = 400
	L = 0.2
	L0m = 400e-9
	Cm  = 160e-12
	alpha = 5.0e-3
	dx = L / Nx
	kin = KinInductance(L0=L0m, alpha=alpha)
	dt_cfl = NLTFDTD.cfl_dt(dx, kin.min_Ld_over_range(0.5), Cm, safety=0.9)
	dt = dt_cfl
	T  = 3.0e-9
	Rs = 50.0
	RL = 50.0
	t0 = 0.6e-9
	sigma = 0.15e-9
	V0 = 0.5
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)
	p = FDTDParams(Nx=Nx, L=L, L0_per_m=L0m, alpha=alpha, C_per_m=Cm, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs)
	out = NLTFDTD(p).run()
	return out, p


# -----------------
# Piecewise demos
# -----------------

def demo_fdtd_piecewise():
	Nx = 600
	L = 0.3
	Rs = 50.0
	RL = 50.0
	t0 = 0.7e-9
	sigma = 0.18e-9
	V0 = 0.6
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)

	reg = [
		FDTDRegion(x0=0.0,     x1=L/2, L0_per_m=400e-9, C_per_m=160e-12, alpha=2.0e-3),
		FDTDRegion(x0=L/2,     x1=L,   L0_per_m=600e-9, C_per_m=250e-12, alpha=8.0e-3),
	]

	dx = L / Nx
	Lmin = min(r.L0_per_m for r in reg)
	Cmax = max(r.C_per_m for r in reg)
	dt = NLTFDTD_PW.cfl_dt(dx, Lmin, Cmax, safety=0.85)

	T  = 3.0e-9
	p = FDTDParamsPW(Nx=Nx, L=L, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs, regions=reg)
	out = NLTFDTD_PW(p).run()
	return out, p

def demo_ladder_piecewise():
	N = 240
	L = 0.24
	Rs = 50.0
	RL = 50.0
	t0 = 0.7e-9
	sigma = 0.18e-9
	V0 = 0.6
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)

	reg = [
		LadderRegion(x0=0.0,   x1=L/3, L0_per_m=380e-9, C_per_m=150e-12, alpha=2.0e-3),
		LadderRegion(x0=L/3,   x1=2*L/3, L0_per_m=420e-9, C_per_m=170e-12, alpha=4.0e-3),
		LadderRegion(x0=2*L/3, x1=L,   L0_per_m=600e-9, C_per_m=250e-12, alpha=8.0e-3),
	]

	dt = 2.0e-12
	T  = 3.0e-9
	p = LadderParamsPW(N=N, L=L, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs, regions=reg)
	out = NLTLadderPW(p).run()
	return out, p

def demo_fdtd_piecewise_v2():
	Nx = 600
	L = 0.3
	Rs = 50.0
	RL = 50.0
	t0 = 0.7e-9
	sigma = 0.18e-9
	V0 = 0.6
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)

	reg = [
		FDTDRegion(x0=0.0,     x1=L/4, L0_per_m=1e-6, C_per_m=130e-12, alpha=1e-3),
		FDTDRegion(x0=L/4,     x1=3*L/4,   L0_per_m=1e-6, C_per_m=130e-12, alpha=10000),
		FDTDRegion(x0=3*L/4,     x1=L,   L0_per_m=1e-6, C_per_m=130e-12, alpha=1e-3),
	]

	dx = L / Nx
	Lmin = min(r.L0_per_m for r in reg)
	Cmax = max(r.C_per_m for r in reg)
	dt = NLTFDTD_PW.cfl_dt(dx, Lmin, Cmax, safety=0.85)

	T  = 3.0e-9
	p = FDTDParamsPW(Nx=Nx, L=L, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs, regions=reg)
	out = NLTFDTD_PW(p).run()
	return out, p

def demo_ladder_piecewise_v2():
	N = 240
	L = 0.24
	Rs = 50.0
	RL = 50.0
	t0 = 0.7e-9
	sigma = 0.18e-9
	V0 = 0.6
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0)

	reg = [
		LadderRegion(x0=0.0,     x1=L/4, L0_per_m=1e-6, C_per_m=130e-12, alpha=1e-3),
		LadderRegion(x0=L/4,     x1=3*L/4,   L0_per_m=1e-6, C_per_m=130e-12, alpha=10000),
		LadderRegion(x0=3*L/4,     x1=L,   L0_per_m=1e-6, C_per_m=130e-12, alpha=1e-3),
	]

	dt = 2.0e-12
	T  = 3.0e-9
	p = LadderParamsPW(N=N, L=L, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs, regions=reg)
	out = NLTLadderPW(p).run()
	return out, p


if __name__ == "__main__":
	out1, p1 = demo_fdtd_piecewise()
	out2, p2 = demo_ladder_piecewise()
	print("PW-FDTD:", out1.v_xt.shape)
	print("PW-LADD:", out2.v_nodes.shape)
