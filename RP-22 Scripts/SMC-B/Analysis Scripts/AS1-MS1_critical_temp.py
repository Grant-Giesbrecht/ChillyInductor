'''
Uses data from RP-22/SMC-B/MS1 (temp and resistance versus time) to determine the critical temperature
of the device.
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

plt.rcParams['font.family'] = 'Aptos'

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

# Get seconds
t_start_sec = timestamps[0].timestamp()
timestamps_seconds = np.array([t_.timestamp()-t_start_sec for t_ in timestamps])

##--------------------------------------------
# Plot Results

fig1 = plt.figure(1)
ax1 = fig1.gca()
ax2 = ax1.twinx()

END_IDX = 2000
ax1.plot(timestamps_seconds[:END_IDX], temperatures[:END_IDX], linestyle=':', label="Temp (K)", color=(0, 0, 0.7), marker='.')
ax2.plot(timestamps_seconds[:END_IDX], resistances[:END_IDX]-sc_res, linestyle='--', label="Resistance (Ohms)", color=(0.7, 0, 0), marker='.')
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

fig3 = plt.figure(3)
ax1_3 = fig3.gca()
ax2_3 = ax1_3.twinx()

START_IDX = 1200
END_IDX = 1700
ax1_3.axvline(x = 225.3, color = (0.85, 0.7, 0.4), linewidth=2)
ax1_3.axvline(x = 258.4, color = (0.85, 0.7, 0.4), linewidth=2)
line1_3, = ax1_3.plot(timestamps_seconds[START_IDX:END_IDX]/60, temperatures[START_IDX:END_IDX], linestyle=':', label="Temp (K)", color=(0, 0, 0.7), marker='.')
line2_3, = ax2_3.plot(timestamps_seconds[START_IDX:END_IDX]/60, resistances[START_IDX:END_IDX]-sc_res, linestyle='--', label="Resistance (Ohms)", color=(0.7, 0, 0), marker='.')
ax1_3.grid(True)
ax1_3.set_ylabel("Temperature (K)")
ax2_3.set_ylabel("Resistance (Ohms)")
ax1_3.set_title("Temperature Latency")
ax1_3.set_xlabel("Time (Minutes)")
ax1_3.legend([line1_3, line2_3], ["Temp (K)", "Resistance ($\Omega$)"])
plt.show()

# Save pickled-figs
pickle.dump(fig1, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig1.pklfig"), 'wb'))
pickle.dump(fig2, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig2.pklfig"), 'wb'))












plt.show()


