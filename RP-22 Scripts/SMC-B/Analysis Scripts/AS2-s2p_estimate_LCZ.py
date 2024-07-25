'''
Uses an S2P file to estimate L, C and Z of a chip.
'''


import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
import sys
import os
from ganymede import *
import datetime
from pylogfile.base import *
import pickle

#------------------------------------------------------------
# Import Data
datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R3C4*B', 'Track 1 4mm'])
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
filename = "RP22B_MS1_24Jul2024_r1.hdf"

analysis_file = os.path.join(datapath, filename)

log = LogPile()

##--------------------------------------------
# Read HDF5 File

print("Loading file contents into memory")
# log.info("Loading file contents into memory")

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	
	# Read SC calibration point
	GROUP = 'calibration'
	sc_res = fh[GROUP]['short_circ_res_ohm'][()]
	sc_temp = fh[GROUP]['short_circ_temp_K'][()]
	sc_date_str = fh[GROUP]['short_circ_time'][()].decode()
	try:
		sc_timestamp = datetime.datetime.strptime(sc_date_str, "%Y-%m-%d %H:%M:%S.%f")
	except:
		log.error(f"Failed to interpret short-circuit timestamp.")
		sc_timestamp = None
	
	
	# Read primary dataset
	GROUP = 'dataset'
	resistances = fh[GROUP]['resistances'][()]
	temperatures = fh[GROUP]['temperatures'][()]
	timestamp_strings = fh[GROUP]['times'][()]
	timestamps = []
	for idx, tsstr in enumerate(timestamp_strings):
		# if idx % 100 == 0:
		# print(f"idx = {idx} of {len(timestamp_strings)}")
			# log.debug(f"idx = {idx} of {len(timestamp_strings)}")
		
		try:
			nts = datetime.datetime.strptime(tsstr.decode(), "%Y-%m-%d %H:%M:%S.%f")
			timestamps.append(nts)
		except:
			log.error(f"Failed to interpret dataset timestamp.")
			sc_timestamp = None
	
	# print(resistances)

# log.info("File contents loaded.")

##--------------------------------------------
# Plot Results

fig1 = plt.figure(1)
ax1 = fig1.gca()
ax2 = ax1.twinx()

END_IDX = 2000
ax1.plot(timestamps[:END_IDX], temperatures[:END_IDX], linestyle=':', label="Temp (K)", color=(0, 0, 0.7), marker='.')
ax2.plot(timestamps[:END_IDX], resistances[:END_IDX]-sc_res, linestyle='--', label="Resistance (Ohms)", color=(0.7, 0, 0), marker='.')
plt.grid(True)
plt.legend()
plt.xlabel("Time")
ax1.set_ylabel("Temperature (K)")
ax2.set_ylabel("Resistance (Ohms)")
ax1.set_title("Cooldown over Time")

fig2 = plt.figure(2)
ax1_2 = fig2.gca()
ax1_2.plot(temperatures, resistances-sc_res, linestyle=':', marker='.', color=(0.7, 0, 0))
ax1_2.set_ylabel("Resistance (Ohms)")
ax1_2.set_xlabel("Temperature (K)")
ax1_2.set_title("Resistance over Temperature")
ax1_2.set_xlim([2.8, 4.4])
plt.grid(True)

# Save pickled-figs
pickle.dump(fig1, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig1.pklfig"), 'wb'))
pickle.dump(fig2, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig2.pklfig"), 'wb'))












plt.show()


