
"""
run_sine_spectrum.py
--------------------
Simulate a piecewise nonlinear transmission line driven by a continuous sine wave,
and analyze the spectral content at the load — with ZERO changes to nltl_sim.py.

It supports both FDTD (piecewise PDE) and Ladder (piecewise LC chain) solvers.
We run long enough to approach steady-state, then window and FFT the tail.

Usage (examples)
---------------
# FDTD, 8 GHz tone, analyze last 200 ns, PSD up to 20 GHz
python run_sine_spectrum.py --solver fdtd --f0 8e9 --V0 0.5 --T 3e-9 --tail 1.0e-9 --fmax 20e9

# Ladder, same stimulus
python run_sine_spectrum.py --solver ladder --f0 8e9 --V0 0.5 --T 3e-9 --tail 1.0e-9 --fmax 20e9
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from nltl_sim_implicit import (
	FDTDRegion, FDTDParamsPW, NLTFDTD_PW,
	LadderRegion, LadderParamsPW, NLTLadderPW,
	NLTFDTD_PW as FDTDClass, NLTLadderPW as LadderClass,
)

from nltl_sim import NLTFDTD_PW, NLTLadderPW  # explicit names
from nltl_analysis import probe_fdtd_voltage, probe_ladder_voltage, spectrum_probe

def build_fdtd_piecewise(f0: float, V0: float, T: float):
	# Regions (reuse the ones in your demo but you can edit here)
	L = 0.3
	Nx = 600
	Rs = 50.0
	RL = 50.0

	reg = [
		FDTDRegion(x0=0.0,     x1=L/2, L0_per_m=400e-9, C_per_m=160e-12, alpha=2.0e-3),
		FDTDRegion(x0=L/2,     x1=L,   L0_per_m=600e-9, C_per_m=250e-12, alpha=8.0e-3),
	]

	dx = L / Nx
	# Conservative CFL based on min(L), max(C)
	Lmin = min(r.L0_per_m for r in reg)
	Cmax = max(r.C_per_m for r in reg)
	dt = NLTFDTD_PW.cfl_dt(dx, Lmin, Cmax, safety=0.85)

	# Continuous sine source
	w0 = 2*np.pi*f0
	Vs = lambda t: V0 * np.sin(w0 * t)

	p = FDTDParamsPW(Nx=Nx, L=L, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs, regions=reg,
					 nonlinear_update="explicit")
	return p

def build_ladder_piecewise(f0: float, V0: float, T: float):
	L = 0.24
	N = 240
	Rs = 50.0
	RL = 50.0

	reg = [
		LadderRegion(x0=0.0,   x1=L/3,   L0_per_m=380e-9, C_per_m=150e-12, alpha=2.0e-3),
		LadderRegion(x0=L/3,   x1=2*L/3, L0_per_m=420e-9, C_per_m=170e-12, alpha=4.0e-3),
		LadderRegion(x0=2*L/3, x1=L,     L0_per_m=600e-9, C_per_m=250e-12, alpha=8.0e-3),
	]

	dt = 2.0e-12  # maintain your earlier ladder dt
	w0 = 2*np.pi*f0
	Vs = lambda t: V0 * np.sin(w0 * t)

	p = LadderParamsPW(N=N, L=L, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs, regions=reg,
					   nonlinear_update="explicit")
	return p

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--solver", choices=["fdtd", "ladder"], default="fdtd",
					help="Which piecewise solver to run")
	ap.add_argument("--f0", type=float, default=8e9, help="Carrier frequency [Hz]")
	ap.add_argument("--V0", type=float, default=0.5, help="Sine amplitude [V]")
	ap.add_argument("--T", type=float, default=30e-9, help="Total simulation time [s]")
	ap.add_argument("--tail", type=float, default=None,
					help="Analyze only the last 'tail' seconds of the waveform for spectrum")
	ap.add_argument("--fmax", type=float, default=None, help="Max frequency to show [GHz]")
	ap.add_argument("--implicit", action="store_true",
					help="Use implicit nonlinear update for robustness")
	args = ap.parse_args()

	f0 = args.f0
	V0 = args.V0
	T = args.T

	if args.solver == "fdtd":
		p = build_fdtd_piecewise(f0, V0, T)
		if args.implicit:
			p.nonlinear_update = "implicit"
		out = NLTFDTD_PW(p).run()
		# Probe at load (last node)
		t = out.t
		v_t = out.v_xt[:, -1]
		dt = t[1] - t[0]
	else:
		p = build_ladder_piecewise(f0, V0, T)
		if args.implicit:
			p.nonlinear_update = "implicit"
		out = NLTLadderPW(p).run()
		# Probe at load node (last)
		t = out.t
		v_t = out.v_nodes[:, -1]
		dt = t[1] - t[0]

	# Optionally take only the tail (to suppress initial transients)
	if args.tail is not None and args.tail > 0:
		t0 = max(0.0, t[-1] - args.tail)
		m = t >= t0
		t = t[m]
		v_t = v_t[m]

	# Spectrum with Hann window and PSD scaling
	from nltl_analysis import spectrum_probe
	spec = spectrum_probe(v_t, dt, window="hann", scaling="psd")

	# Plots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7))

	# Time series (last 10 cycles visible)
	ncyc = 10
	T0 = 1.0 / f0
	t_end = t[-1]
	t_start = max(t[0], t_end - ncyc*T0)
	m = (t >= t_start)
	ax1.plot(t[m]*1e9, v_t[m])
	ax1.set_xlabel("Time (ns)")
	ax1.set_ylabel("V_load (V)")
	ax1.set_title(f"{args.solver.upper()} — sine drive at {f0/1e9:.3f} GHz, V0={V0} V")

	# Spectrum
	ax2.semilogy(spec.freqs_hz/1e9, spec.spec)
	ax2.set_xlabel("Frequency (GHz)")
	ax2.set_ylabel(spec.scaling)
	ax2.set_title("Load spectrum (PSD, Hann window)")
	if args.fmax:
		ax2.set_xlim(0, args.fmax)
	
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()
