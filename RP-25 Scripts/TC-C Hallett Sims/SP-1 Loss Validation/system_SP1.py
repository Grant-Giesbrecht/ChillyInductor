import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

from hallett.core import dB_to_Np
from hallett.nltsim.core import *
from hallett.nltsim.analysis import *

def loss_to_conductance(C:float, R:float, L:float, loss_dB_per_m):
	''' Converts loss per meter to conductance (approx).
	'''
	
	alpha = dB_to_Np(loss_dB_per_m) # Np/m
	return 4*C*(alpha**2 - R/L)

def define_system(total_length:float, f0:float, V0:float, t_end:float, dx_ref:float, implicit:bool, V_bias:float=0):
	''' Returns parameter objects for the two sim types. Defined in a function so it's easier
	to recycle the same system into multiple simulations.
	
	Parameters:
		total_length (float): Length overall of system. All regions scale with total_length.
		f0 (float): Fundamental tone in Hz
		t_end (float):
	
	'''
	
	dist_L0 = 1e-6
	dist_C0 = 130e-12
	
	# Define source and load impedances
	Rs = 50.0
	RL = 50.0
	
	# Define regions
	regions = [
		TLINRegion(x0=0.0,     x1=total_length,   L0_per_m=dist_L0, C_per_m=dist_C0, G_per_m=0, alpha=1e-9),
	]
	
	# Select a ∆t using the CFL condition
	Nx = max(50, int(np.round(total_length / dx_ref))) 
	dx = total_length / Nx # Get ∆x
	Lmin = min(r.L0_per_m for r in regions) # Get max total_length
	Cmax = max(r.C_per_m for r in regions) # Get max C
	dt = FiniteDiffSim.cfl_dt(dx, Lmin, Cmax, safety=0.85) # Get CFL. Smaller `safety` makes smaller ∆t 
	
	# Calculate omega
	w0 = 2*np.pi*f0
	
	# Define voltage stimulus
	Vs = lambda t: V0 * np.sin(w0 * t) + V_bias
	
	update_type = ("implicit" if implicit else "explicit")
	# print(f"update = {update_type}")
	
	# Prepare parameter object for FDTD simulation
	fdtd_params = FiniteDiffParams(Nx=Nx, total_length=total_length, dt=dt, t_end=t_end, Rs=Rs, RL=RL, Vs_func=Vs, regions=regions, nonlinear_update=update_type)

	# Return param objects
	return fdtd_param