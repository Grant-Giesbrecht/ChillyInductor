'''
Reads a series of S2P files to approx a cal and system loss.
'''


import matplotlib.pyplot as plt
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
import os
from ganymede import *
from pylogfile.base import *
import sys
import numpy as np
import mplcursors
import pickle

class SParamData:
	
	def __init__(self, filename):
		
		try:
			csv_data = read_rohde_schwarz_csv(filename)
		except Exception as e:
			print(f"Failed to read CSV file. {e}")
			return
		
		self.freq_GHz = csv_data.freq_Hz/1e9
		
		try:
			self.S11 = csv_data.S11_real + complex(0, 1)*csv_data.S11_imag
			self.S11_dB = lin_to_dB(np.abs(self.S11))
		except:
			self.S11 = None
			self.S11_dB = None
		
		try:
			self.S21 = csv_data.S21_real + complex(0, 1)*csv_data.S21_imag
			self.S21_dB = lin_to_dB(np.abs(self.S21))
		except:
			self.S21 = None
			self.S21_dB = None
			
		try:
			self.S12 = csv_data.S12_real + complex(0, 1)*csv_data.S12_imag
			self.S12_dB = lin_to_dB(np.abs(self.S12))
		except:
			self.S12 = None
			self.S12_dB = None
			
		try:
			self.S22 = csv_data.S22_real + complex(0, 1)*csv_data.S22_imag
			self.S22_dB = lin_to_dB(np.abs(self.S22))
		except:
			self.S22 = None
			self.S22_dB = None

#------------------------------------------------------------
# Import Data

datapath_R4C4T1 = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm', 'VNA Traces'])
datapath_R4C4T2_hp = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 2 43mm', 'Uncalibrated SParam', 'Prf 0 dBm'])
datapath_R4C4T2_lp = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 2 43mm', 'Uncalibrated SParam', 'Prf -30 dBm'])

if datapath_R4C4T2_hp is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath_R4C4T2_hp}{Style.RESET_ALL}")

sp_thru = SParamData(os.path.join(datapath_R4C4T2_hp, "26Aug2024_Through.csv"))

sp_loss = SParamData(os.path.join(datapath_R4C4T2_hp, "26Aug2024_Through.csv"))
sp_bias2 = SParamData(os.path.join(datapath_R4C4T2_lp, "MiniCirc_ZFBT4R2GW_Ch1ToRF+DC.csv"))
sp_bias1 = SParamData(os.path.join(datapath_R4C4T2_lp, "PulseLabsWBlueCable_Ch2ToRF+DC.csv"))
sp_cryo = SParamData(os.path.join(datapath_R4C4T2_lp, "26Aug2024_Ch1ToCryoR_Ch2ToCryoL.csv"))

plt.figure(1)
plt.plot(sp_loss.freq_GHz, sp_loss.S11_dB, marker='.', label='S11')
plt.plot(sp_loss.freq_GHz, sp_loss.S21_dB, marker='.', label='S21')
plt.xlabel("Frequency (GHz)")
plt.ylabel("S-Parameters (dB)")
plt.title("0 dBm Through")
plt.grid(True)
plt.legend()

plt.figure(2)
plt.plot(sp_cryo.freq_GHz, sp_cryo.S11_dB, marker='.', label='S11')
plt.plot(sp_cryo.freq_GHz, sp_cryo.S21_dB, marker='.', label='S21')
plt.xlabel("Frequency (GHz)")
plt.ylabel("S-Parameters (dB)")
plt.title("Cryostat")
plt.grid(True)
plt.legend()

## This is dumb - don't do this. It's not even close to valid whoops.
# plt.figure(3)
# plt.plot(sp_cryo.freq_GHz, sp_cryo.S11_dB-sp_loss.S11_dB, marker='.', label='S11')
# plt.plot(sp_cryo.freq_GHz, sp_cryo.S21_dB-sp_loss.S21_dB, marker='.', label='S21')
# plt.xlabel("Frequency (GHz)")
# plt.ylabel("S-Parameters (dB)")
# plt.title("Cryostat -Through")
# plt.grid(True)
# plt.legend()

plt.show()