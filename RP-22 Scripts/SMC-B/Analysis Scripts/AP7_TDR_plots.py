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

##-------------------------------------------
# Plot TDR graphs - 10 dB

display_name = '10-dB'

XLIM = [42, 62]
YLIM = [45, 55]

fig1 = plt.figure(1, figsize=(8, 10))

src = data['dataset']['atten10dB']['open']
plt.subplot(4, 1, 1)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Open-Circuit")
plt.xlim(XLIM)
plt.ylim(YLIM)

src = data['dataset']['atten10dB']['50_term']
plt.subplot(4, 1, 2)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable 50-Ohm Termination")
plt.xlim(XLIM)
plt.ylim(YLIM)

src = data['dataset']['atten10dB']['cryostat']['opposite_50_term']
plt.subplot(4, 1, 3)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Cryostat-R")
plt.xlim(XLIM)
plt.ylim(YLIM)

src = data['dataset']['atten10dB']['cryostat_left']['opposite_50_term']
plt.subplot(4, 1, 4)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Cryostat-L")
plt.xlim(XLIM)
plt.ylim(YLIM)

# fig1.suptitle("10 dB Attenuator")
fig1.tight_layout()

##-------------------------------------------
# Plot TDR graphs - 13 dB

display_name = '13-dB'

fig1 = plt.figure(1, figsize=(8, 10))

src = data['dataset']['atten13dB']['open']
plt.subplot(4, 1, 1)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Open-Circuit")
plt.xlim(XLIM)
plt.ylim(YLIM)

src = data['dataset']['atten13dB']['50_term']
plt.subplot(4, 1, 2)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable 50-Ohm Termination")
plt.xlim(XLIM)
plt.ylim(YLIM)

src = data['dataset']['atten13dB']['cryostat']['opposite_50_term']
plt.subplot(4, 1, 3)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Cryostat-R")
plt.xlim(XLIM)
plt.ylim(YLIM)

src = data['dataset']['atten13dB']['cryostat_left']['opposite_50_term']
plt.subplot(4, 1, 4)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Cryostat-L")
plt.xlim(XLIM)
plt.ylim(YLIM)

# fig1.suptitle("13 dB Attenuator")
fig1.tight_layout()

##-------------------------------------------
# Plot TDR graphs - 20 dB

display_name = '20-dB'

fig1 = plt.figure(1, figsize=(8, 10))

src = data['dataset']['atten20dB']['open']
plt.subplot(4, 1, 1)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Open-Circuit")
plt.xlim(XLIM)
plt.ylim(YLIM)
plt.legend()

src = data['dataset']['atten20dB']['50_term']
plt.subplot(4, 1, 2)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable 50-Ohm Termination")
plt.xlim(XLIM)
plt.ylim(YLIM)
plt.legend()

src = data['dataset']['atten20dB']['cryostat']['opposite_50_term']
plt.subplot(4, 1, 3)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Cryostat-R")
plt.xlim(XLIM)
plt.ylim(YLIM)
plt.legend()

src = data['dataset']['atten20dB']['cryostat_left']['opposite_50_term']
plt.subplot(4, 1, 4)
plt.plot(np.array(src['x'])*1e9, src['y'], linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
plt.xlabel("Time (ns)")
plt.ylabel("Impedance (Ohms)")
plt.grid(True)
plt.title("Cable Cryostat-L")
plt.xlim(XLIM)
plt.ylim(YLIM)
plt.legend()

# fig1.suptitle("20 dB Attenuator")
fig1.tight_layout()








plt.show()