import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

from nltl_core import *
from nltl_analysis import *
from system import *

parser = argparse.ArgumentParser()
parser.add_argument("--solver", choices=["fdtd", "ladder"], default="fdtd")
parser.add_argument("--f0", type=float, default=8e9, help="Stimulus frequency [Hz]")
parser.add_argument("--V0", type=float, default=0.5, help="Stimulus amplitude [V]")
parser.add_argument("--Vmin", type=float, default=-1, help="Min total length [m]")
parser.add_argument("--Vmax", type=float, default=1, help="Max total length [m]")
parser.add_argument("--num_sweep", type=int, default=9, help="Number of sweep points")
parser.add_argument("--dx_ref", type=float, default=None,
				help="Reference spatial step [m]. If None, use 0.3/600 for FDTD, 0.24/240 for Ladder")
parser.add_argument("--T", type=float, default=4e-9, help="Total simulation time [s]")
parser.add_argument("--tail", type=float, default=1.5e-9, help="Analyze only last 'tail' seconds")
parser.add_argument("--bw_bins", type=int, default=3, help="Half-width in FFT bins for harmonic integration")
parser.add_argument("--implicit", action="store_true", help="Use implicit nonlinearity update")
parser.add_argument("--out", type=str, default="sweep_length.csv", help="Output CSV path")
parser.add_argument("--plot", action="store_true", help="Plot P(dBm) vs L for harmonics")
parser.add_argument("--csv", action="store_true", help="Save result to CSV")
args = parser.parse_args()

# ---------------------------- Main ----------------------------

def main():
	
	
	if args.dx_ref is None:
		args.dx_ref = (0.3/600) if args.solver == "fdtd" else (0.24/240)
	
	bias_vals = np.linspace(args.Vmin, args.Vmax, args.num_sweep)
	# L_vals = linstep(0.1, 1, 0.05)
	rows = [["L_m", "P1_W", "P2_W", "P3_W", "P1_dBm", "P2_dBm", "P3_dBm"]]
	
	P1_list = []
	P2_list = []
	P3_list = []
	
	phys_length = 0.2
	
	# Scan over all lengths
	for vb in bias_vals:
		
		# L = 0.5
		
		# Define system parameters
		fdtd_params, le_params = define_system(phys_length, args.f0, args.V0, args.T, args.dx_ref, args.implicit, V_bias=vb)
		
		# 
		if args.solver == "fdtd":
			sim = FiniteDiffSim(fdtd_params) # Create sim
			out = sim.run() # run
			v_t = out.v_xt[:, -1]
		else:
			sim = LumpedElementSim(le_params) # Create sim
			out = sim.run() # Run
			v_t = out.v_nodes[:, -1]
		
		# Get numpy arrays from simulation results
		t = out.t
		
		
		# Tail for steady-state
		if args.tail is not None and args.tail > 0:
			t0 = max(0.0, t[-1] - args.tail)
			m = t >= t0
			t = t[m]; v_t = v_t[m]
		
		dt = t[1] - t[0]
		spec = spectrum_probe(v_t, dt, window="hann", scaling="psd")
		f = spec.freqs_hz
		psd = spec.spec

		P1 = tone_power_from_psd(f, psd, args.f0, args.bw_bins, RL=50.0)
		P2 = tone_power_from_psd(f, psd, 2*args.f0, args.bw_bins, RL=50.0)
		P3 = tone_power_from_psd(f, psd, 3*args.f0, args.bw_bins, RL=50.0)
		
		def w2dbm(Pw):
			return 10*np.log10(Pw/1e-3) if Pw > 0 else -np.inf

		P1_dBm = w2dbm(P1); P2_dBm = w2dbm(P2); P3_dBm = w2dbm(P3)

		rows.append([f"{vb:.6g}", f"{P1:.6e}", f"{P2:.6e}", f"{P3:.6e}",
					 f"{P1_dBm:.2f}", f"{P2_dBm:.2f}", f"{P3_dBm:.2f}"])

		P1_list.append(P1_dBm); P2_list.append(P2_dBm); P3_list.append(P3_dBm)
		print(f"Vdc={vb:.3f} m  ->  P1={P1_dBm:.2f} dBm,  P2={P2_dBm:.2f} dBm,  P3={P3_dBm:.2f} dBm")
		
		# plt.plot(f/1e9, psd, linestyle=':', marker='o')
		# plt.xlim([0, args.f0/1e9*4])
		# plt.show()
		
	# Write CSV
	if args.csv:
		out_path = args.out
		with open(out_path, "w", newline="") as fcsv:
			writer = csv.writer(fcsv)
			writer.writerows(rows)
		print(f"Saved: {out_path}")

	if args.plot:
		
		plt.figure(figsize=(8,5))
		plt.plot(bias_vals, P1_list, marker='o', label='Fundamental')
		plt.plot(bias_vals, P2_list, marker='s', label='2nd harmonic')
		plt.plot(bias_vals, P3_list, marker='^', label='3rd harmonic')
		plt.xlabel("Bias Voltage (V)")
		plt.ylabel("Power at load (dBm)")
		plt.title(f"{args.solver.upper()} — Harmonic power vs. L (f0={args.f0/1e9:.2f} GHz, V0={args.V0} V)")
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	main()
