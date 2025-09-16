import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

from nltl_core import *
from nltl_analysis import *

def define_system(L: float, f0: float, V0: float, T: float, dx_ref: float, implicit: bool):
	
	# Define source and load impedances
	Rs = 50.0
	RL = 50.0
	
	# Define regions
	regions = [
		TLINRegion(x0=0.0,     x1=L/3,   L0_per_m=380e-9, C_per_m=150e-12, alpha=2.0e-3),
		TLINRegion(x0=L/3,     x1=2*L/3, L0_per_m=420e-9, C_per_m=170e-12, alpha=100),
		TLINRegion(x0=2*L/3,   x1=L,     L0_per_m=600e-9, C_per_m=250e-12, alpha=8.0e-3)
	]
	
	# Select a ∆t using the CFL condition
	Nx = max(50, int(np.round(L / dx_ref))) 
	dx = L / Nx # Get ∆x
	Lmin = min(r.L0_per_m for r in regions) # Get max L
	Cmax = max(r.C_per_m for r in regions) # Get max C
	dt = FiniteDiffSim.cfl_dt(dx, Lmin, Cmax, safety=0.85) # Get CFL. Smaller `safety` makes smaller ∆t
	
	# 
	w0 = 2*np.pi*f0
	Vs = lambda t: V0 * np.sin(w0 * t)
	p = FiniteDiffParams(Nx=Nx, L=L, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs, regions=regions, nonlinear_update=("implicit" if implicit else "explicit"))

	
	Rs = 50.0; RL = 50.0
	
	N = max(30, int(np.round(L / dx_ref)))
	dt = 2.0e-12  # fixed ladder dt (edit if needed)
	w0 = 2*np.pi*f0
	Vs = lambda t: V0 * np.sin(w0 * t)
	p = LumpedElementParams(N=N, L=L, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs, regions=regions, nonlinear_update=("implicit" if implicit else "explicit"))
	return p, p