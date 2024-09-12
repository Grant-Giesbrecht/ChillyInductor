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

#------------------------------------------------------------
# Import Data

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm'])
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

filename = "MP4_13Aug2024_R4C4T1_r1.hdf"



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
	

##-------------------------------------------
# Plot TDR graphs - 10 dB

display_name = '10-dB'

XLIM = [42, 62]
YLIM1 = [45, 55]
YLIM2T = [0, 400]
YLIM2B = [40, 80]

fig1, (ax1_1, ax1_2, ax1_3, ax1_4) = plt.subplots(4, 1, figsize=(8, 10))
fig2, (ax2_1, ax2_2, ax2_3, ax2_4) = plt.subplots(4, 1, figsize=(8, 10))

src = data['dataset']['atten10dB']['open']
# fig1.subplots(4, 1, 1)
ax1_1.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_1.set_xlabel("Time (ns)")
ax1_1.set_ylabel("Impedance (Ohms)")
ax1_1.grid(True)
ax1_1.set_title("Cable Open-Circuit")
ax1_1.set_xlim(XLIM)
ax1_1.set_ylim(YLIM1)

src = data['dataset']['atten10dB']['50_term']
# ax1_2.subplots(4, 1, 2)
ax1_2.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_2.set_xlabel("Time (ns)")
ax1_2.set_ylabel("Impedance (Ohms)")
ax1_2.grid(True)
ax1_2.set_title("Cable 50-Ohm Termination")
ax1_2.set_xlim(XLIM)
ax1_2.set_ylim(YLIM1)

src = data['dataset']['atten10dB']['cryostat']['opposite_50_term']
# fig1.subplots(4, 1, 3)
ax1_3.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_3.set_xlabel("Time (ns)")
ax1_3.set_ylabel("Impedance (Ohms)")
ax1_3.grid(True)
ax1_3.set_title("Cable Cryostat-R")
ax1_3.set_xlim(XLIM)
ax1_3.set_ylim(YLIM1)

src = data['dataset']['atten10dB']['cryostat_left']['opposite_50_term']
# ax1_4.subplots(4, 1, 4)
ax1_4.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_4.set_xlabel("Time (ns)")
ax1_4.set_ylabel("Impedance (Ohms)")
ax1_4.grid(True)
ax1_4.set_title("Cable Cryostat-L")
ax1_4.set_xlim(XLIM)
ax1_4.set_ylim(YLIM1)

# fig1.suptitle("10 dB Attenuator")
fig1.tight_layout()

##-------------------------------------------
# Plot TDR graphs - 13 dB

display_name = '13-dB'

# fig1 = plt.figure(1, figsize=(8, 10))

src = data['dataset']['atten13dB']['open']
# plt.subplot(4, 1, 1)
ax1_1.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_1.set_xlabel("Time (ns)")
ax1_1.set_ylabel("Impedance (Ohms)")
ax1_1.grid(True)
ax1_1.set_title("Cable Open-Circuit")
ax1_1.set_xlim(XLIM)
ax1_1.set_ylim(YLIM1)

src = data['dataset']['atten13dB']['50_term']
# plt.subplot(4, 1, 2)
ax1_2.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_2.set_xlabel("Time (ns)")
ax1_2.set_ylabel("Impedance (Ohms)")
ax1_2.grid(True)
ax1_2.set_title("Cable 50-Ohm Termination")
ax1_2.set_xlim(XLIM)
ax1_2.set_ylim(YLIM1)

src = data['dataset']['atten13dB']['cryostat']['opposite_50_term']
# plt.subplot(4, 1, 3)
ax1_3.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_3.set_xlabel("Time (ns)")
ax1_3.set_ylabel("Impedance (Ohms)")
ax1_3.grid(True)
ax1_3.set_title("Cable Cryostat-R")
ax1_3.set_xlim(XLIM)
ax1_3.set_ylim(YLIM1)

src = data['dataset']['atten13dB']['cryostat_left']['opposite_50_term']
# plt.subplot(4, 1, 4)
ax1_4.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_4.set_xlabel("Time (ns)")
ax1_4.set_ylabel("Impedance (Ohms)")
ax1_4.grid(True)
ax1_4.set_title("Cable Cryostat-L")
ax1_4.set_xlim(XLIM)
ax1_4.set_ylim(YLIM1)

# fig1.suptitle("13 dB Attenuator")
fig1.tight_layout()

##-------------------------------------------
# Plot TDR graphs - 20 dB

display_name = '20-dB'

fig1 = plt.figure(1, figsize=(8, 10))

src = data['dataset']['atten20dB']['open']
# ax1_1.subplot(4, 1, 1)
ax1_1.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_1.set_xlabel("Time (ns)")
ax1_1.set_ylabel("Impedance (Ohms)")
ax1_1.grid(True)
ax1_1.set_title("Cable Open-Circuit")
ax1_1.set_xlim(XLIM)
ax1_1.set_ylim(YLIM1)
ax1_1.legend()

src = data['dataset']['atten20dB']['50_term']
# ax1_2.subplot(4, 1, 2)
ax1_2.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_2.set_xlabel("Time (ns)")
ax1_2.set_ylabel("Impedance (Ohms)")
ax1_2.grid(True)
ax1_2.set_title("Cable 50-Ohm Termination")
ax1_2.set_xlim(XLIM)
ax1_2.set_ylim(YLIM1)
ax1_2.legend()

src = data['dataset']['atten20dB']['cryostat']['opposite_50_term']
# plt.subplot(4, 1, 3)
ax1_3.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_3.set_xlabel("Time (ns)")
ax1_3.set_ylabel("Impedance (Ohms)")
ax1_3.grid(True)
ax1_3.set_title("Cable Cryostat-R")
ax1_3.set_xlim(XLIM)
ax1_3.set_ylim(YLIM1)
ax1_3.legend()

src = data['dataset']['atten20dB']['cryostat_left']['opposite_50_term']
# plt.subplot(4, 1, 4)
ax1_4.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_4.set_xlabel("Time (ns)")
ax1_4.set_ylabel("Impedance (Ohms)")
ax1_4.grid(True)
ax1_4.set_title("Cable Cryostat-L")
ax1_4.set_xlim(XLIM)
ax1_4.set_ylim(YLIM1)
ax1_4.legend()

# fig1.suptitle("20 dB Attenuator")
fig1.tight_layout()

###################################################################
# Figure 2 - transformed to compensate for attenuator

##-------------------------------------------
# Plot TDR graphs - 10 dB

display_name = '10-dB'
atten_dB = 10

src = data['dataset']['atten10dB']['open']
# fig1.subplots(4, 1, 1)
ax2_1.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_1.set_xlabel("Time (ns)")
ax2_1.set_ylabel("Impedance (Ohms)")
ax2_1.grid(True)
ax2_1.set_title("Corrected: Cable Open-Circuit")
ax2_1.set_xlim(XLIM)
ax2_1.set_ylim(YLIM2T)

src = data['dataset']['atten10dB']['50_term']
# ax2_2.subplots(4, 1, 2)
ax2_2.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_2.set_xlabel("Time (ns)")
ax2_2.set_ylabel("Impedance (Ohms)")
ax2_2.grid(True)
ax2_2.set_title("Corrected: Cable 50-Ohm Termination")
ax2_2.set_xlim(XLIM)
ax2_2.set_ylim(YLIM2T)

src = data['dataset']['atten10dB']['cryostat']['opposite_50_term']
# fig1.subplots(4, 1, 3)
ax2_3.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_3.set_xlabel("Time (ns)")
ax2_3.set_ylabel("Impedance (Ohms)")
ax2_3.grid(True)
ax2_3.set_title("Corrected: Cable Cryostat-R")
ax2_3.set_xlim(XLIM)
ax2_3.set_ylim(YLIM2B)

src = data['dataset']['atten10dB']['cryostat_left']['opposite_50_term']
# ax2_4.subplots(4, 1, 4)
ax2_4.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_4.set_xlabel("Time (ns)")
ax2_4.set_ylabel("Impedance (Ohms)")
ax2_4.grid(True)
ax2_4.set_title("Corrected: Cable Cryostat-L")
ax2_4.set_xlim(XLIM)
ax2_4.set_ylim(YLIM2B)

# fig1.suptitle("10 dB Attenuator")
fig1.tight_layout()

##-------------------------------------------
# Plot TDR graphs - 13 dB

display_name = '13-dB'
atten_dB = 13

# fig1 = plt.figure(1, figsize=(8, 10))

src = data['dataset']['atten13dB']['open']
# plt.subplot(4, 1, 1)
ax2_1.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_1.set_xlabel("Time (ns)")
ax2_1.set_ylabel("Impedance (Ohms)")
ax2_1.grid(True)
ax2_1.set_title("Corrected: Cable Open-Circuit")
ax2_1.set_xlim(XLIM)
ax2_1.set_ylim(YLIM2T)

src = data['dataset']['atten13dB']['50_term']
# plt.subplot(4, 1, 2)
ax2_2.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_2.set_xlabel("Time (ns)")
ax2_2.set_ylabel("Impedance (Ohms)")
ax2_2.grid(True)
ax2_2.set_title("Corrected: Cable 50-Ohm Termination")
ax2_2.set_xlim(XLIM)
ax2_2.set_ylim(YLIM2T)

src = data['dataset']['atten13dB']['cryostat']['opposite_50_term']
# plt.subplot(4, 1, 3)
ax2_3.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_3.set_xlabel("Time (ns)")
ax2_3.set_ylabel("Impedance (Ohms)")
ax2_3.grid(True)
ax2_3.set_title("Corrected: Cable Cryostat-R")
ax2_3.set_xlim(XLIM)
ax2_3.set_ylim(YLIM2B)

src = data['dataset']['atten13dB']['cryostat_left']['opposite_50_term']
# plt.subplot(4, 1, 4)
ax2_4.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_4.set_xlabel("Time (ns)")
ax2_4.set_ylabel("Impedance (Ohms)")
ax2_4.grid(True)
ax2_4.set_title("Corrected: Cable Cryostat-L")
ax2_4.set_xlim(XLIM)
ax2_4.set_ylim(YLIM2B)

# fig1.suptitle("13 dB Attenuator")
fig1.tight_layout()

##-------------------------------------------
# Plot TDR graphs - 20 dB

display_name = '20-dB'
atten_dB = 20

fig1 = plt.figure(1, figsize=(8, 10))

src = data['dataset']['atten20dB']['open']
# ax2_1.subplot(4, 1, 1)
ax2_1.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_1.set_xlabel("Time (ns)")
ax2_1.set_ylabel("Impedance (Ohms)")
ax2_1.grid(True)
ax2_1.set_title("Corrected: Cable Open-Circuit")
ax2_1.set_xlim(XLIM)
ax2_1.set_ylim(YLIM2T)
ax2_1.legend()

src = data['dataset']['atten20dB']['50_term']
# ax2_2.subplot(4, 1, 2)
ax2_2.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_2.set_xlabel("Time (ns)")
ax2_2.set_ylabel("Impedance (Ohms)")
ax2_2.grid(True)
ax2_2.set_title("Corrected: Cable 50-Ohm Termination")
ax2_2.set_xlim(XLIM)
ax2_2.set_ylim(YLIM2T)
ax2_2.legend()

src = data['dataset']['atten20dB']['cryostat']['opposite_50_term']
# plt.subplot(4, 1, 3)
ax2_3.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_3.set_xlabel("Time (ns)")
ax2_3.set_ylabel("Impedance (Ohms)")
ax2_3.grid(True)
ax2_3.set_title("Corrected: Cable Cryostat-R")
ax2_3.set_xlim(XLIM)
ax2_3.set_ylim(YLIM2B)
ax2_3.legend()

src = data['dataset']['atten20dB']['cryostat_left']['opposite_50_term']
# plt.subplot(4, 1, 4)
ax2_4.plot(np.array(src['x'])*1e9, atten_tdr_transform(src['y'], atten_dB, 50), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_4.set_xlabel("Time (ns)")
ax2_4.set_ylabel("Impedance (Ohms)")
ax2_4.grid(True)
ax2_4.set_title("Corrected: Cable Cryostat-L")
ax2_4.set_xlim(XLIM)
ax2_4.set_ylim(YLIM2B)
ax2_4.legend()

# fig1.suptitle("20 dB Attenuator")
fig1.tight_layout()
fig2.tight_layout()








plt.show()
