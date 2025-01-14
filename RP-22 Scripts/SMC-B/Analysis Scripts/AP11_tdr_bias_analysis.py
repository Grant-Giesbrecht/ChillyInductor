'''
Reads TDR data from an MP4-HDF5 file and plots the results.

'''


import matplotlib.pyplot as plt
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
import os
from ganymede import *
from pylogfile.base import *
import sys
import numpy as np
import pickle
from jarnsaxa import *

plt.rcParams['font.family'] = 'Consolas'

#------------------------------------------------------------
# Import Data

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R3C1*D', 'Track 3 454mm', 'tdr_bias'], )
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

filename = "RP22B_MP6_05Oct2024_R3C1T3_r1.hdf"
Z0x = 49.5

analysis_file = os.path.join(datapath, filename)

log = LogPile()

##--------------------------------------------
# Read HDF5 File

print("Loading file contents into memory")

data = hdf_to_dict(analysis_file)

print(f"Summary of datafile:")
dict_summary(data)

def atten_tdr_transform(Z_data:list, atten_dB:float, Z0:float=50):
	
	# Calculate attenuation
	# atten_lin = dB_to_lin(np.abs(atten_dB), 10)
	# atten_lin = dB_to_lin(np.abs(atten_dB), 20)
	atten_lin = dB_to_lin(np.abs(atten_dB), use10=True)
	Z_data = np.array(Z_data)
	
	# Calculate reflection coefficeint
	gamma_a = (Z_data - Z0)/(Z_data + Z0)
	gamma_0 = gamma_a*atten_lin
	
	# Apply correction
	Z_corrected = Z0*(gamma_0 + 1)/(1 - gamma_0)
	
	return Z_corrected
	

def gamma_to_Z(gvals:list):
	
	gvals = np.array(gvals)
	
	Ztdr = 50*(1+gvals)/(1-gvals)
	return Ztdr

def get_colormap_colors(colormap_name, n):
	"""
	Returns 'n' colors as tuples that are evenly spaced in the specified colormap.
	
	Parameters:
	colormap_name (str): The name of the colormap.
	n (int): The number of colors to return.
	
	Returns:
	list: A list of 'n' colors as tuples.
	"""
	cmap = plt.get_cmap(colormap_name)
	colors = [cmap(i / (n - 1)) for i in range(n)]
	return colors

atten_dB = 13
Z0x = 49.5
trace_idx = 0

# plt.figure(1)
# trace_idx = 0
# plt.plot(data['dataset']['TDR_x'][trace_idx], atten_tdr_transform(data['dataset']['TDR_y'][trace_idx], atten_dB, Z0x))
# trace_idx = 1
# plt.plot(data['dataset']['TDR_x'][trace_idx], atten_tdr_transform(data['dataset']['TDR_y'][trace_idx], atten_dB, Z0x))
# trace_idx = 3
# plt.plot(data['dataset']['TDR_x'][trace_idx], atten_tdr_transform(data['dataset']['TDR_y'][trace_idx], atten_dB, Z0x))
# trace_idx = 4
# plt.plot(data['dataset']['TDR_x'][trace_idx], atten_tdr_transform(data['dataset']['TDR_y'][trace_idx], atten_dB, Z0x))
# trace_idx = 5
# plt.plot(data['dataset']['TDR_x'][trace_idx], atten_tdr_transform(data['dataset']['TDR_y'][trace_idx], atten_dB, Z0x))


# plt.ylim([0, 100])
# plt.show()


TWILIGHT5 = get_colormap_colors('plasma', 5)

plt.figure(1)
trace_idx = 0
Idc = data['dataset']['I_dc_mA'][trace_idx]
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[0], label=f"I_$dc$ = {rd(Idc)} mA")
trace_idx = 1
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[1], label=f"I_$dc$ = {rd(Idc)} mA")
trace_idx = 3
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[2], label=f"I_$dc$ = {rd(Idc)} mA")
trace_idx = 4
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[3], label=f"I_$dc$ = {rd(Idc)} mA")
trace_idx = 5
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[4], label=f"I_$dc$ = {rd(Idc)} mA")
plt.legend()
plt.grid(True)
plt.ylim([50, 60])


plt.figure(2)
trace_idx = 6
Idc = data['dataset']['I_dc_mA'][trace_idx]
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[0], label=f"I_$dc$ = {rd(Idc)} mA")
trace_idx = 7
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[1], label=f"I_$dc$ = {rd(Idc)} mA")
trace_idx = 8
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[2], label=f"I_$dc$ = {rd(Idc)} mA")
trace_idx = 9
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[3], label=f"I_$dc$ = {rd(Idc)} mA")
trace_idx = 10
plt.plot(data['dataset']['TDR_x'][trace_idx], data['dataset']['TDR_y'][trace_idx], color=TWILIGHT5[4], label=f"I_$dc$ = {rd(Idc)} mA")
plt.legend()
plt.grid(True)
# plt.ylim([50, 60])


plt.show()