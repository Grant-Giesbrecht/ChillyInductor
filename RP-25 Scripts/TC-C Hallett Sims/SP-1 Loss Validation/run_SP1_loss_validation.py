import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from hallett.nltsim.core import *
from hallett.nltsim.analysis import *
from system_SP1 import *

parser = argparse.ArgumentParser()
parser.add_argument("--f0", type=float, default=4e9, help="Stimulus frequency [Hz]")
parser.add_argument("--V0", type=float, default=1, help="Stimulus amplitude [V]")
parser.add_argument("--Lmin", type=float, default=-1, help="Min total length [m]")
parser.add_argument("--Lmax", type=float, default=1, help="Max total length [m]")
parser.add_argument("--num_sweep", type=int, default=9, help="Number of sweep points")
parser.add_argument("--dx_ref", type=float, default=None,
				help="Reference spatial step [m]. If None, use 0.3/600 for FDTD, 0.24/240 for Ladder")
parser.add_argument("--T", type=float, default=4e-9, help="Total simulation time [s]")
parser.add_argument("-p", "--parallel", action='store_true', help="Run simulations in parallel")
parser.add_argument("--tail", type=float, default=1.5e-9, help="Analyze only last 'tail' seconds")
parser.add_argument("--bw_bins", type=int, default=3, help="Half-width in FFT bins for harmonic integration")
parser.add_argument("--Vbias", type=float, default="1", help="Applied bias voltage.")

args = parser.parse_args()

# ---------------------------- Main ----------------------------

def main():
	
	
	if args.dx_ref is None:
		args.dx_ref = 5e-4
	
	total_lenths = np.linspace(args.Lmin, args.Lmax, args.num_sweep)
	
	harm_powers = HarmonicPowers(x_parameter="Bias voltage", x_values=bias_vals)
	
	t0 = time.time()
	print(f"Running sequentially.")
	
	# Scan over all lengths
	for tot_len in total_lenths:
		
		# Define system parameters
		fdtd_params_exp= define_system(tot_len, args.f0, args.V0, args.T, args.dx_ref, False, V_bias=args.Vbias)
		
		fdtd_exp_sim = FiniteDiffSim(fdtd_params_exp) # Create sim
		fdtd_exp_out = fdtd_exp_sim.run() # run
		load_harmonics_probe(fdtd_exp_out, harm_powers, args.f0, args.tail, args.bw_bins, 3)
	
	print(f"Sequential simulation finished ({time.time()-t0} sec).")
	
	c_fund = 'tab:blue'
	c_2h = 'tab:orange'
	c_3h = 'tab:green'
	
	c_fundL = 'tab:cyan'
	c_2hL = 'tab:purple'
	c_3hL = 'tab:gray'
	
	plt.figure(1, figsize=(8,5))
	
	plt.plot(bias_vals, harm_powers.f0, marker='x', label='Fundamental, Explicit', color=c_fund, linestyle='--')
	plt.plot(bias_vals, harm_powers.h2, marker='x', label='2nd harmonic, Explicit', color=c_2h, linestyle='--')
	plt.plot(bias_vals, harm_powers.h3, marker='x', label='3rd harmonic, Explicit', color=c_3h, linestyle='--')
	
	plt.xlabel("Bias Voltage (V)")
	plt.ylabel("Power at load (dBm)")
	plt.title(f"FDTD Simulation, Explicit vs Implicit Updates")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	
	
	plt.show()

if __name__ == "__main__":
	main()
