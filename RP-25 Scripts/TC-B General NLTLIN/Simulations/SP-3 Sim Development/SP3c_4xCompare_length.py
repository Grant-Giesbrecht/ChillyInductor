import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

from nltl_core import *
from nltl_analysis import *
from system import *

parser = argparse.ArgumentParser()
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
parser.add_argument("--out", type=str, default="sweep_length.csv", help="Output CSV path")
parser.add_argument("--plot", action="store_true", help="Plot P(dBm) vs L for harmonics")
parser.add_argument("--csv", action="store_true", help="Save result to CSV")
args = parser.parse_args()

# ---------------------------- Main ----------------------------

@dataclass
class HarmonicPowers:
	P1_list: np.ndarray
	P2_list: np.ndarray
	P3_list: np.ndarray

def process_ouput(out, vb, hp_obj):
	
	if isinstance(out, FiniteDiffResult):
		v_t = out.v_xt[:, -1]
	else:
		v_t = out.v_nodes[:, -1]
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
	
	# def w2dbm(Pw):
	# 	return 10*np.log10(Pw/1e-3) if Pw > 0 else -np.inf

	P1_dBm = w2dbm(P1)
	P2_dBm = w2dbm(P2)
	P3_dBm = w2dbm(P3)

	# rows.append([f"{vb:.6g}", f"{P1:.6e}", f"{P2:.6e}", f"{P3:.6e}",f"{P1_dBm:.2f}", f"{P2_dBm:.2f}", f"{P3_dBm:.2f}"])

	hp_obj.P1_list.append(P1_dBm)
	hp_obj.P2_list.append(P2_dBm)
	hp_obj.P3_list.append(P3_dBm)
	
	print(f"Vdc={vb:.3f} m  ->  P1={P1_dBm:.2f} dBm,  P2={P2_dBm:.2f} dBm,  P3={P3_dBm:.2f} dBm")

def main():
	
	# ???
	if args.dx_ref is None:
		args.dx_ref = 5e-4
	
	bias_vals = np.linspace(args.Vmin, args.Vmax, args.num_sweep)
	# L_vals = linstep(0.1, 1, 0.05)
	# rows = [["L_m", "P1_W", "P2_W", "P3_W", "P1_dBm", "P2_dBm", "P3_dBm"]]
	
	fdtd_exp_powers = HarmonicPowers([], [], [])
	fdtd_imp_powers = HarmonicPowers([], [], [])
	le_exp_powers = HarmonicPowers([], [], [])
	le_imp_powers = HarmonicPowers([], [], [])
	
	phys_length = 0.2
	
	# Scan over all lengths
	for vb in bias_vals:
		
		# L = 0.5
		
		# Define system parameters
		fdtd_params_exp, le_params_exp = define_system(phys_length, args.f0, args.V0, args.T, args.dx_ref, False, V_bias=vb) # Explicit params
		fdtd_params_imp, le_params_imp = define_system(phys_length, args.f0, args.V0, args.T, args.dx_ref, True, V_bias=vb) # Implicit params
		
		fdtd_exp_sim = FiniteDiffSim(fdtd_params_exp) # Create sim
		fdtd_exp_out = fdtd_exp_sim.run() # run
		process_ouput(fdtd_exp_out, vb, fdtd_exp_powers)
		# fdtd_exp_vt = fdtd_exp_out.v_xt[:, -1]
		# fdtd_exp_t = fdtd_exp_out.t
		
		le_exp_sim = LumpedElementSim(le_params_exp) # Create sim
		le_exp_out = le_exp_sim.run() # Run
		process_ouput(le_exp_out, vb, le_exp_powers)
		
		# le_exp_vt = le_exp_out.v_nodes[:, -1]
		# le_exp_t = le_exp_out.t
		
		fdtd_imp_sim = FiniteDiffSim(fdtd_params_imp) # Create sim
		fdtd_imp_out = fdtd_imp_sim.run() # run
		process_ouput(fdtd_imp_out, vb, fdtd_imp_powers)
		# fdtd_imp_vt = fdtd_imp_out.v_xt[:, -1]
		# fdtd_imp_t = fdtd_imp_out.t
		
		le_imp_sim = LumpedElementSim(le_params_imp) # Create sim
		le_imp_out = le_imp_sim.run() # Run
		process_ouput(le_imp_out, vb, le_imp_powers)
		# le_imp_vt = le_imp_out.v_nodes[:, -1]
		# le_imp_t = le_imp_out.t
		

		
		# plt.plot(f/1e9, psd, linestyle=':', marker='o')
		# plt.xlim([0, args.f0/1e9*4])
		# plt.show()
		
	# # Write CSV
	# if args.csv:
	# 	out_path = args.out
	# 	with open(out_path, "w", newline="") as fcsv:
	# 		writer = csv.writer(fcsv)
	# 		writer.writerows(rows)
	# 	print(f"Saved: {out_path}")

	# if args.plot:
	
	c_fund = 'tab:blue'
	c_2h = 'tab:orange'
	c_3h = 'tab:green'
	
	c_fundL = 'tab:cyan'
	c_2hL = 'tab:purple'
	c_3hL = 'tab:gray'
	
	plt.figure(1, figsize=(8,5))
	
	plt.plot(bias_vals, fdtd_exp_powers.P1_list, marker='x', label='Fundamental, Explicit', color=c_fund, linestyle='--')
	plt.plot(bias_vals, fdtd_exp_powers.P2_list, marker='x', label='2nd harmonic, Explicit', color=c_2h, linestyle='--')
	plt.plot(bias_vals, fdtd_exp_powers.P3_list, marker='x', label='3rd harmonic, Explicit', color=c_3h, linestyle='--')
	
	plt.plot(bias_vals, fdtd_imp_powers.P1_list, marker='+', label='Fundamental, Implicit', color=c_fund, linestyle=':')
	plt.plot(bias_vals, fdtd_imp_powers.P2_list, marker='+', label='2nd harmonic, Implicit', color=c_2h, linestyle=':')
	plt.plot(bias_vals, fdtd_imp_powers.P3_list, marker='+', label='3rd harmonic, Implicit', color=c_3h, linestyle=':')
	
	plt.xlabel("Bias Voltage (V)")
	plt.ylabel("Power at load (dBm)")
	plt.title(f"FDTD Simulation, Explicit vs Implicit Updates")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	
	plt.figure(2, figsize=(8,5))
	
	plt.plot(bias_vals, le_exp_powers.P1_list, marker='x', label='Fundamental, Explicit', color=c_fundL, linestyle='--')
	plt.plot(bias_vals, le_exp_powers.P2_list, marker='x', label='2nd harmonic, Explicit', color=c_2hL, linestyle='--')
	plt.plot(bias_vals, le_exp_powers.P3_list, marker='x', label='3rd harmonic, Explicit', color=c_3hL, linestyle='--')
	
	plt.plot(bias_vals, le_imp_powers.P1_list, marker='+', label='Fundamental, Implicit', color=c_fundL, linestyle=':')
	plt.plot(bias_vals, le_imp_powers.P2_list, marker='+', label='2nd harmonic, Implicit', color=c_2hL, linestyle=':')
	plt.plot(bias_vals, le_imp_powers.P3_list, marker='+', label='3rd harmonic, Implicit', color=c_3hL, linestyle=':')
	
	plt.xlabel("Bias Voltage (V)")
	plt.ylabel("Power at load (dBm)")
	plt.title(f"Lumped-Element Simulation, Explicit vs Implicit Updates")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	
	plt.figure(3, figsize=(8,5))
	
	plt.plot(bias_vals, fdtd_imp_powers.P1_list, marker='x', label='Fundamental, FDTD', color=c_fund, linestyle='--')
	plt.plot(bias_vals, fdtd_imp_powers.P2_list, marker='x', label='2nd harmonic, FDTD', color=c_2h, linestyle='--')
	plt.plot(bias_vals, fdtd_imp_powers.P3_list, marker='x', label='3rd harmonic, FDTD', color=c_3h, linestyle='--')
	
	plt.plot(bias_vals, le_imp_powers.P1_list, marker='+', label='Fundamental, Lumped-Element', color=c_fundL, linestyle=':')
	plt.plot(bias_vals, le_imp_powers.P2_list, marker='+', label='2nd harmonic, Lumped-Element', color=c_2hL, linestyle=':')
	plt.plot(bias_vals, le_imp_powers.P3_list, marker='+', label='3rd harmonic, Lumped-Element', color=c_3hL, linestyle=':')
	
	plt.xlabel("Bias Voltage (V)")
	plt.ylabel("Power at load (dBm)")
	plt.title(f"Implicit Updates, FDTD vs Lumped-Element Simulations")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	
	
	plt.show()

if __name__ == "__main__":
	main()
