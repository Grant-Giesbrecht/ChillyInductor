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

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm', 'tdr'])
# if datapath is None:
# 	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
# 	sys.exit()
# else:
# 	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

# filename = "MP4_13Aug2024_R4C4T1_r1.hdf"
# use_old_tdr_file_format = True
# Z0x = 49.5
# # Z0x = 50

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R3C1*D', 'Track 2 43mm', 'tdr'])
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

filename = "RP22B_MP4_24Sept2024_R3C1T2_r1.hdf"
use_old_tdr_file_format = False
Z0x = 49.25
# Z0x = 50

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

##-----1--------------------------------------
# Plot TDR graphs - 10 dB


XLIM = [42, 62]
YLIM1 = [45, 55]
YLIM2T = [0, 400]
YLIM2B = [40, 60]

averaging_cryo_R_x_ns = [57.75, 75]
averaging_cryo_L_x_ns = [57.75, 75]

fig1, (ax1_1, ax1_2, ax1_3, ax1_4) = plt.subplots(4, 1, figsize=(8, 10))
fig2, (ax2_1, ax2_2, ax2_3, ax2_4) = plt.subplots(4, 1, figsize=(8, 10))

##-------------------------------------------
# Plot TDR graphs - 13 dB

display_name = '13-dB'

# fig1 = plt.figure(1, figsize=(8, 10))

if use_old_tdr_file_format:
	srcy = (data['dataset']['atten13dB']['open']['y'])
	src = data['dataset']['atten13dB']['open']
else:
	srcy = (data['dataset']['atten13dB']['open']['Z']['y'])
	src = data['dataset']['atten13dB']['open']['Z']

ax1_1.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_1.set_xlabel("Time (ns)")
ax1_1.set_ylabel("Impedance (Ohms)")
ax1_1.grid(True)
ax1_1.set_title("Cable Open-Circuit")
# ax1_1.set_xlim(XLIM)
# ax1_1.set_ylim(YLIM1)

if use_old_tdr_file_format:
	srcy = (data['dataset']['atten13dB']['50_term']['y'])
	src = data['dataset']['atten13dB']['50_term']
else:
	srcy = (data['dataset']['atten13dB']['match']['Z']['y'])
	src = data['dataset']['atten13dB']['match']['Z']
# plt.subplot(4, 1, 2)
ax1_2.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_2.set_xlabel("Time (ns)")
ax1_2.set_ylabel("Impedance (Ohms)")
ax1_2.grid(True)
ax1_2.set_title("Cable 50-Ohm Termination")
# ax1_2.set_xlim(XLIM)
# ax1_2.set_ylim(YLIM1)

if use_old_tdr_file_format:
	srcy = (data['dataset']['atten13dB']['50_term']['y'])
	src = data['dataset']['atten13dB']['50_term']
else:
	srcy = (data['dataset']['atten13dB']['CryoR_MatchL']['Z']['y'])
	src = data['dataset']['atten13dB']['CryoR_MatchL']['Z']
# plt.subplot(4, 1, 3)
ax1_3.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_3.set_xlabel("Time (ns)")
ax1_3.set_ylabel("Impedance (Ohms)")
ax1_3.grid(True)
ax1_3.set_title("Cable Cryostat-R")
# ax1_3.set_xlim(XLIM)
# ax1_3.set_ylim(YLIM1)

if use_old_tdr_file_format:
	srcy = (data['dataset']['atten13dB']['cryostat_left']['opposite_50_term']['y'])
	src = data['dataset']['atten13dB']['cryostat_left']['opposite_50_term']
else:
	srcy = (data['dataset']['atten13dB']['CryoL_MatchR']['Z']['y'])
	src = data['dataset']['atten13dB']['CryoL_MatchR']['Z']
# plt.subplot(4, 1, 4)
ax1_4.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax1_4.set_xlabel("Time (ns)")
ax1_4.set_ylabel("Impedance (Ohms)")
ax1_4.grid(True)
ax1_4.set_title("Cable Cryostat-L")
# ax1_4.set_xlim(XLIM)
# ax1_4.set_ylim(YLIM1)

# fig1.suptitle("13 dB Attenuator")
fig1.tight_layout()

###################################################################
# Figure 2 - transformed to compensate for attenuator

##-------------------------------------------
# Plot TDR graphs - 13 dB

display_name = '13-dB'
atten_dB = 13



# fig1 = plt.figure(1, figsize=(8, 10))

if use_old_tdr_file_format:
	srcy = (data['dataset']['atten13dB']['open']['y'])
	src = data['dataset']['atten13dB']['open']
else:
	srcy = (data['dataset']['atten13dB']['open']['Z']['y'])
	src = data['dataset']['atten13dB']['open']['Z']
# plt.subplot(4, 1, 1)
ax2_1.plot(np.array(src['x'])*1e9, atten_tdr_transform(srcy, atten_dB, Z0x), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_1.set_xlabel("Time (ns)")
ax2_1.set_ylabel("Impedance (Ohms)")
ax2_1.grid(True)
ax2_1.set_title("Corrected: Cable Open-Circuit")
# ax2_1.set_xlim(XLIM)
ax2_1.set_ylim(YLIM2T)

if use_old_tdr_file_format:
	srcy = (data['dataset']['atten13dB']['50_term']['y'])
	src = data['dataset']['atten13dB']['50_term']
else:
	srcy = (data['dataset']['atten13dB']['match']['Z']['y'])
	src = data['dataset']['atten13dB']['match']['Z']
# plt.subplot(4, 1, 2)
ax2_2.plot(np.array(src['x'])*1e9, atten_tdr_transform(srcy, atten_dB, Z0x), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_2.set_xlabel("Time (ns)")
ax2_2.set_ylabel("Impedance (Ohms)")
ax2_2.grid(True)
ax2_2.set_title("Corrected: Cable 50-Ohm Termination")
# ax2_2.set_xlim(XLIM)
ax2_2.set_ylim(YLIM2T)

if use_old_tdr_file_format:
	srcy = (data['dataset']['atten13dB']['cryostat']['opposite_50_term']['y'])
	src = data['dataset']['atten13dB']['cryostat']['opposite_50_term']
else:
	srcy = (data['dataset']['atten13dB']['CryoR_MatchL']['Z']['y'])
	src = data['dataset']['atten13dB']['CryoR_MatchL']['Z']
# plt.subplot(4, 1, 3)
cryo_R = atten_tdr_transform(srcy, atten_dB, Z0x)
ax2_3.plot(np.array(src['x'])*1e9, cryo_R, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_3.set_xlabel("Time (ns)")
ax2_3.set_ylabel("Impedance (Ohms)")
ax2_3.grid(True)
ax2_3.set_title("Corrected: Cable Cryostat-R")
# ax2_3.set_xlim(XLIM)
ax2_3.set_ylim(YLIM2B)

if use_old_tdr_file_format:
	srcy = (data['dataset']['atten13dB']['cryostat_left']['opposite_50_term']['y'])
	src = data['dataset']['atten13dB']['cryostat_left']['opposite_50_term']
else:
	srcy = (data['dataset']['atten13dB']['CryoL_MatchR']['Z']['y'])
	src = data['dataset']['atten13dB']['CryoL_MatchR']['Z']
# plt.subplot(4, 1, 4)
cryo_L = atten_tdr_transform(srcy, atten_dB, Z0x)
ax2_4.plot(np.array(src['x'])*1e9, cryo_L, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_4.set_xlabel("Time (ns)")
ax2_4.set_ylabel("Impedance (Ohms)")
ax2_4.grid(True)
ax2_4.set_title("Corrected: Cable Cryostat-L")
# ax2_4.set_xlim(XLIM)
ax2_4.set_ylim(YLIM2B)

# fig1.suptitle("13 dB Attenuator")
fig1.tight_layout()

##-------------------------------------------------------------------
## Calculate standard deviations


avging_mask = (np.array(src['x'])*1e9 >= averaging_cryo_L_x_ns[0]) & (np.array(src['x'])*1e9 <= averaging_cryo_L_x_ns[1])
stdev_L = np.std(cryo_L[avging_mask])
mean_L = np.mean(cryo_L[avging_mask])
stdev_R = np.std(cryo_R[avging_mask])
mean_R = np.mean(cryo_R[avging_mask])

print(f"Cryostat from left:")
print(f"\tMean: {mean_L}")
print(f"\tStDev: {stdev_L}")

print(f"Cryostat from right:")
print(f"\tMean: {mean_R}")
print(f"\tStDev: {stdev_R}")

plt.figure(3)
plt.hist(cryo_L[avging_mask], bins=15, alpha=0.4)
plt.hist(cryo_R[avging_mask], bins=15, alpha=0.4)


plt.show()
