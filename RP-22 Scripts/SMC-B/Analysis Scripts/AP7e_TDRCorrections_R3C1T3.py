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

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R3C1*D', 'Track 3 454mm', 'tdr'])
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

filename = "RP22B_MP4_05Oct2024_R3C1T3_r2.hdf"
use_old_tdr_file_format = False
Z0x = 49.2

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

# fig1, (ax1_1, ax1_2, ax1_3, ax1_4) = plt.subplots(4, 1, figsize=(8, 10))
fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(8, 10))
fig3, ((ax3_1, ax3_2), (ax3_3, ax3_4)) = plt.subplots(2, 2, figsize=(8, 10))

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
ax2_1.axhline(y = 50, color=(0.7, 0.7, 0.7), linestyle = '--') 
line1, = ax2_1.plot(np.array(src['x'])*1e9, atten_tdr_transform(srcy, atten_dB, Z0x), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line2, = ax2_1.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_1.set_xlabel("Time (ns)")
ax2_1.set_ylabel("Impedance (Ohms)")
ax2_1.grid(True)
ax2_1.set_title("Cable, Open")
ax2_1.legend([line1, line2], ["Corrected", "Raw"])

# ax2_1.set_xlim(XLIM)
ax2_1.set_ylim([0, 150])

srcy = (data['dataset']['atten13dB']['match']['Z']['y'])
src = data['dataset']['atten13dB']['match']['Z']
# plt.subplot(4, 1, 2)
ax2_2.axhline(y = 50, color=(0.7, 0.7, 0.7), linestyle = '--') 
line1, = ax2_2.plot(np.array(src['x'])*1e9, atten_tdr_transform(srcy, atten_dB, Z0x), linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line2, = ax2_2.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax2_2.set_xlabel("Time (ns)")
ax2_2.set_ylabel("Impedance (Ohms)")
ax2_2.grid(True)
ax2_2.set_title("Cable, Matched")
ax2_2.legend([line1, line2], ["Corrected", "Raw"])
ax2_2.set_ylim([0, 150])

fig2.suptitle("Cable Measurements")

drift_correction = atten_tdr_transform(srcy, atten_dB, Z0x) - 50

##--------------------------------- Cryostat measurements

# srcy = (data['dataset']['atten13dB']['CryoR_MatchL']['Z']['y'])
# drift_correction_R = atten_tdr_transform(srcy, atten_dB, Z0x) - 50
# srcy = (data['dataset']['atten13dB']['CryoL_MatchR']['Z']['y'])
# drift_correction_L = atten_tdr_transform(srcy, atten_dB, Z0x) - 50

# drift_correction = (drift_correction_R + drift_correction_L)/2

srcy = (data['dataset']['atten13dB']['CryoR_MatchL']['Z']['y'])
src = data['dataset']['atten13dB']['CryoR_MatchL']['Z']
cryo_R = atten_tdr_transform(srcy, atten_dB, Z0x)
ax3_1.axhline(y = 50, color=(0.7, 0.7, 0.7), linestyle = '--') 
line1, = ax3_1.plot(np.array(src['x'])*1e9, cryo_R-drift_correction, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line2, = ax3_1.plot(np.array(src['x'])*1e9, cryo_R, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line3, = ax3_1.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax3_1.set_xlabel("Time (ns)")
ax3_1.set_ylabel("Impedance (Ohms)")
ax3_1.grid(True)
srcy = (data['dataset']['atten13dB']['CryoL_MatchR']['Z']['y'])
src = data['dataset']['atten13dB']['CryoL_MatchR']['Z']
cryo_L = atten_tdr_transform(srcy, atten_dB, Z0x)
line4, = ax3_1.plot(np.array(src['x'])*1e9, cryo_L-drift_correction, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line5, = ax3_1.plot(np.array(src['x'])*1e9, cryo_L, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line6, = ax3_1.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax3_1.set_xlabel("Time (ns)")
ax3_1.set_ylabel("Impedance (Ohms)")
ax3_1.grid(True)
ax3_1.set_title("Cryostat, Matched")
ax3_1.legend([line1, line2, line3, line4, line5, line6], ["Corrected v2 (R)", "Corrected (R)", "Raw (R)", "Corrected v2 (L)", "Corrected (L)", "Raw (R)"])
ax3_1.set_ylim([50, 60])

srcy = (data['dataset']['atten13dB']['CryoR_OpenL']['Z']['y'])
src = data['dataset']['atten13dB']['CryoR_OpenL']['Z']
# plt.subplot(4, 1, 4)
cryo_L = atten_tdr_transform(srcy, atten_dB, Z0x)
ax3_3.axhline(y = 50, color=(0.7, 0.7, 0.7), linestyle = '--') 
line1, = ax3_3.plot(np.array(src['x'])*1e9, cryo_L-drift_correction, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line2, = ax3_3.plot(np.array(src['x'])*1e9, cryo_L, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line3, = ax3_3.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax3_3.set_xlabel("Time (ns)")
ax3_3.set_ylabel("Impedance (Ohms)")
ax3_3.grid(True)
srcy = (data['dataset']['atten13dB']['CryoL_OpenR']['Z']['y'])
src = data['dataset']['atten13dB']['CryoL_OpenR']['Z']
# plt.subplot(4, 1, 4)
cryo_L = atten_tdr_transform(srcy, atten_dB, Z0x)
line4, = ax3_3.plot(np.array(src['x'])*1e9, cryo_L-drift_correction, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line5, = ax3_3.plot(np.array(src['x'])*1e9, cryo_L, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line6, = ax3_3.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax3_3.set_xlabel("Time (ns)")
ax3_3.set_ylabel("Impedance (Ohms)")
ax3_3.grid(True)
ax3_3.set_title("Cryostat, Open")
ax3_1.legend([line1, line2, line3, line4, line5, line6], ["Corrected v2 (R)", "Corrected (R)", "Raw (R)", "Corrected v2 (L)", "Corrected (L)", "Raw (R)"])
ax3_1.set_ylim([50, 60])
# ax2_4.set_xlim(XLIM)
# ax2_5.set_ylim(YLIM2B)

srcy = (data['dataset']['atten13dB']['BNC_AB_Open']['Z']['y'])
src = data['dataset']['atten13dB']['BNC_AB_Open']['Z']
cryo_L = atten_tdr_transform(srcy, atten_dB, Z0x)
line1, = ax3_4.plot(np.array(src['x'])*1e9, cryo_L-drift_correction, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
line2, = ax3_4.plot(np.array(src['x'])*1e9, srcy, linestyle=':', marker='.', markersize=2, linewidth=0.5, label=display_name)
ax3_4.set_xlabel("Time (ns)")
ax3_4.set_ylabel("Impedance (Ohms)")
ax3_4.grid(True)
ax3_4.set_title("BNC Cable, Open")
ax3_4.legend([line1, line2], ["Corrected", "Raw"])

# fig1.suptitle("13 dB Attenuator")
fig3.tight_layout()

##-------------------------------------------------------------------
## Calculate standard deviations


avging_mask = (np.array(src['x'])*1e9 >= averaging_cryo_L_x_ns[0]) & (np.array(src['x'])*1e9 <= averaging_cryo_L_x_ns[1])
stdev_L = np.std(cryo_L[avging_mask])
mean_L = np.mean(cryo_L[avging_mask])
stdev_R = np.std(cryo_R[avging_mask])
mean_R = np.mean(cryo_R[avging_mask])

cryo_RLavg = np.concatenate((cryo_R[avging_mask], cryo_L[avging_mask]))
stdev_RLa = np.std(cryo_RLavg)
mean_RLa = np.mean(cryo_RLavg)

print(f"Cryostat from left:")
print(f"\tMean: {mean_L}")
print(f"\tStDev: {stdev_L}")

print(f"Cryostat from right:")
print(f"\tMean: {mean_R}")
print(f"\tStDev: {stdev_R}")

print(f"Combined:")
print(f"\tMean: {mean_RLa}")
print(f"\tStDev: {stdev_RLa}")

plt.figure(3)
plt.hist(cryo_L[avging_mask], bins=15, alpha=0.4)
plt.hist(cryo_R[avging_mask], bins=15, alpha=0.4)

fig4, ax = plt.subplots(figsize=(4.5,3))
ax.hist(cryo_RLavg, bins=15)
ax.set_xlabel("Characteristic Impedance ($\\Omega$)")
ax.set_ylabel("Counts")
fig4.tight_layout()
ax.grid()

fig4.savefig("AP7d_fig4.svg", format='svg')
fig2.savefig("AP7d_fig2.svg", format='svg')

plt.show()
