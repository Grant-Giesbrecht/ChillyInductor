
"""
run_sine_sweep_length.py
------------------------
Sweep the TOTAL LINE LENGTH L, drive with a continuous sine at f0, and measure the
power at the fundamental, 2nd, and 3rd harmonics at the load. Works with either
piecewise FDTD or piecewise Ladder WITHOUT modifying nltl_sim.py.

Method
------
- For each L in [L_min, L_max], we keep the region *fractions* the same (e.g., [0, 1/2, 1]).
  Per-unit parameters (L0_per_m, C_per_m, alpha) remain fixed; only the total length changes.
- Spatial resolution is kept roughly constant by scaling Nx (FDTD) or N (Ladder) ~ L/dx_ref.
- We simulate for duration T, optionally analyze only the last 'tail' seconds to suppress transients.
- Spectrum is computed (Hann + PSD). Harmonic power is estimated by integrating PSD within a small
  bandwidth around each target frequency and converting to power into RL (default 50 Ω).

Usage
-----
python run_sine_sweep_length.py --solver fdtd --f0 8e9 --V0 0.5 --Lmin 0.10 --Lmax 0.40 --nL 7 \
	--T 4e-9 --tail 1.5e-9 --dx_ref 0.3/600 --implicit --bw_bins 3 --out sweep_fdtd.csv

python run_sine_sweep_length.py --solver ladder --f0 8e9 --V0 0.5 --Lmin 0.10 --Lmax 0.40 --nL 7 \
	--T 4e-9 --tail 1.5e-9 --dx_ref 0.24/240 --implicit --bw_bins 3 --out sweep_ladder.csv
"""

import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

from nltl_sim_implicit import (
	FDTDRegion, FDTDParamsPW, NLTFDTD_PW,
	LadderRegion, LadderParamsPW, NLTLadderPW,
)

from nltl_analysis import spectrum_probe

# ----------------------------
# Region templates (fractions)
# ----------------------------

def fdtd_regions_for_length(L: float):
	"""Return FDTDRegion list for total length L using fixed fractions and per-unit params."""
	return [
		FDTDRegion(x0=0.0,   x1=0.5*L, L0_per_m=400e-9, C_per_m=160e-12, alpha=2.0e-3),
		FDTDRegion(x0=0.5*L, x1=1.0*L, L0_per_m=600e-9, C_per_m=250e-12, alpha=8.0e-3),
	]

def ladder_regions_for_length(L: float):
	"""Return LadderRegion list for total length L using fixed fractions and per-unit params."""
	return [
		LadderRegion(x0=0.0,     x1=L/3,   L0_per_m=380e-9, C_per_m=150e-12, alpha=2.0e-3),
		LadderRegion(x0=L/3,     x1=2*L/3, L0_per_m=420e-9, C_per_m=170e-12, alpha=100),
		LadderRegion(x0=2*L/3,   x1=L,     L0_per_m=600e-9, C_per_m=250e-12, alpha=8.0e-3),
	]

# ----------------------------
# Builders
# ----------------------------

def build_fdtd(L: float, f0: float, V0: float, T: float, dx_ref: float, implicit: bool):
	Rs = 50.0; RL = 50.0
	reg = fdtd_regions_for_length(L)
	Nx = max(50, int(np.round(L / dx_ref)))
	dx = L / Nx
	Lmin = min(r.L0_per_m for r in reg)
	Cmax = max(r.C_per_m for r in reg)
	dt = NLTFDTD_PW.cfl_dt(dx, Lmin, Cmax, safety=0.85)
	w0 = 2*np.pi*f0
	Vs = lambda t: V0 * np.sin(w0 * t)
	p = FDTDParamsPW(Nx=Nx, L=L, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs, regions=reg,
					 nonlinear_update=("implicit" if implicit else "explicit"))
	return p

def build_ladder(L: float, f0: float, V0: float, T: float, dx_ref: float, implicit: bool):
	Rs = 50.0; RL = 50.0
	reg = ladder_regions_for_length(L)
	N = max(30, int(np.round(L / dx_ref)))
	dt = 2.0e-12  # fixed ladder dt (edit if needed)
	w0 = 2*np.pi*f0
	Vs = lambda t: V0 * np.sin(w0 * t)
	p = LadderParamsPW(N=N, L=L, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs, regions=reg,
					   nonlinear_update=("implicit" if implicit else "explicit"))
	return p

# ----------------------------
# Harmonic power estimator
# ----------------------------

def harmonic_power_from_psd(v: np.ndarray, dt: float, freqs: np.ndarray, psd: np.ndarray,
							f_target: float, bw_bins: int, RL: float = 50.0):
	"""
	Estimate power at f_target (fundamental/harmonics) by integrating PSD over +/- bw_bins bins.
	PSD has units ~ V^2/Hz. Multiply by bin bandwidth (df) and sum to get V^2, then P=V_rms^2/RL.
	"""
	if len(freqs) < 3:
		return np.nan
	df = freqs[1] - freqs[0]
	k0 = int(np.argmin(np.abs(freqs - f_target)))
	k_lo = max(0, k0 - bw_bins)
	k_hi = min(len(freqs)-1, k0 + bw_bins)
	v2 = np.trapz(psd[k_lo:k_hi+1], dx=df)  # integrate PSD over band -> V^2
	P = v2 / RL                             # watts (since V_rms^2/R)
	return P

# ----------------------------
# Main
# ----------------------------

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--solver", choices=["fdtd", "ladder"], default="fdtd")
	ap.add_argument("--f0", type=float, default=8e9, help="Stimulus frequency [Hz]")
	ap.add_argument("--V0", type=float, default=0.5, help="Stimulus amplitude [V]")
	ap.add_argument("--Lmin", type=float, default=0.10, help="Min total length [m]")
	ap.add_argument("--Lmax", type=float, default=0.40, help="Max total length [m]")
	ap.add_argument("--nL", type=int, default=7, help="Number of sweep points")
	ap.add_argument("--dx_ref", type=float, default=None,
					help="Reference spatial step [m]. If None, use 0.3/600 for FDTD, 0.24/240 for Ladder")
	ap.add_argument("--T", type=float, default=4e-9, help="Total simulation time [s]")
	ap.add_argument("--tail", type=float, default=1.5e-9, help="Analyze only last 'tail' seconds")
	ap.add_argument("--bw_bins", type=int, default=3, help="Half-width in FFT bins for harmonic integration")
	ap.add_argument("--implicit", action="store_true", help="Use implicit nonlinearity update")
	ap.add_argument("--out", type=str, default="sweep_length.csv", help="Output CSV path")
	ap.add_argument("--plot", action="store_true", help="Plot P(dBm) vs L for harmonics")
	args = ap.parse_args()

	if args.dx_ref is None:
		args.dx_ref = (0.3/600) if args.solver == "fdtd" else (0.24/240)

	L_vals = np.linspace(args.Lmin, args.Lmax, args.nL)
	rows = [["L_m", "P1_W", "P2_W", "P3_W", "P1_dBm", "P2_dBm", "P3_dBm"]]

	P1_list = []; P2_list = []; P3_list = []

	for L in L_vals:
		if args.solver == "fdtd":
			p = build_fdtd(L, args.f0, args.V0, args.T, args.dx_ref, args.implicit)
			out = NLTFDTD_PW(p).run()
			t = out.t; v_t = out.v_xt[:, -1]
		else:
			p = build_ladder(L, args.f0, args.V0, args.T, args.dx_ref, args.implicit)
			out = NLTLadderPW(p).run()
			t = out.t; v_t = out.v_nodes[:, -1]

		# Tail for steady-state
		if args.tail is not None and args.tail > 0:
			t0 = max(0.0, t[-1] - args.tail)
			m = t >= t0
			t = t[m]; v_t = v_t[m]

		dt = t[1] - t[0]
		spec = spectrum_probe(v_t, dt, window="hann", scaling="psd")
		f = spec.freqs_hz; psd = spec.spec

		P1 = harmonic_power_from_psd(v_t, dt, f, psd, args.f0, args.bw_bins, RL=50.0)
		P2 = harmonic_power_from_psd(v_t, dt, f, psd, 2*args.f0, args.bw_bins, RL=50.0)
		P3 = harmonic_power_from_psd(v_t, dt, f, psd, 3*args.f0, args.bw_bins, RL=50.0)

		def w2dbm(Pw):
			return 10*np.log10(Pw/1e-3) if Pw > 0 else -np.inf

		P1_dBm = w2dbm(P1); P2_dBm = w2dbm(P2); P3_dBm = w2dbm(P3)

		rows.append([f"{L:.6g}", f"{P1:.6e}", f"{P2:.6e}", f"{P3:.6e}",
					 f"{P1_dBm:.2f}", f"{P2_dBm:.2f}", f"{P3_dBm:.2f}"])

		P1_list.append(P1_dBm); P2_list.append(P2_dBm); P3_list.append(P3_dBm)
		print(f"L={L:.3f} m  ->  P1={P1_dBm:.2f} dBm,  P2={P2_dBm:.2f} dBm,  P3={P3_dBm:.2f} dBm")

	# Write CSV
	out_path = args.out
	with open(out_path, "w", newline="") as fcsv:
		writer = csv.writer(fcsv)
		writer.writerows(rows)
	print(f"Saved: {out_path}")

	if args.plot:
		import matplotlib.pyplot as plt
		plt.figure(figsize=(8,5))
		plt.plot(L_vals, P1_list, marker='o', label='Fundamental')
		plt.plot(L_vals, P2_list, marker='s', label='2nd harmonic')
		plt.plot(L_vals, P3_list, marker='^', label='3rd harmonic')
		plt.xlabel("Total line length L (m)")
		plt.ylabel("Power at load (dBm)")
		plt.title(f"{args.solver.upper()} — Harmonic power vs. L (f0={args.f0/1e9:.2f} GHz, V0={args.V0} V)")
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	main()
