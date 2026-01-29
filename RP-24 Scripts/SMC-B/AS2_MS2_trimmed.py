
import matplotlib.pyplot as plt
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
import os
from pylogfile.base import *
import sys
import numpy as np
import pickle
import mplcursors

datapath = os.path.join("/", "run", "media", "grant", "M6 T7S", "ARC0 PhD Data", "RP-24 KIFM Series 3", "Data", "SMC-B SiN Chips", "Critical Temperature")
filename = os.path.join(datapath, "RP24B_MS2_15Dec2025_R2C0W7BT3_r1.hdf")



analysis_file = os.path.join(datapath, filename)

log = LogPile()

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
	set_points = fh[GROUP]['temp_set_points_K'][()]
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

indexes = np.linspace(0, len(timestamps)-1, len(timestamps))

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(2, 1)
ax1a = fig1.add_subplot(gs1[0, 0])
ax1b = fig1.add_subplot(gs1[1, 0])

# x_params = timestamps
x_params = indexes

START_IDX = 950
END_IDX = 2000
ax1a.plot(x_params[START_IDX:END_IDX], temperatures[START_IDX:END_IDX], linestyle=':', label="Temp (K)", color=(0, 0, 0.7), marker='.')
ax1b.plot(x_params[START_IDX:END_IDX], resistances[START_IDX:END_IDX]-sc_res, linestyle='--', label="Resistance (Ohms)", color=(0.7, 0, 0), marker='.')
ax1a.grid(True)
ax1b.grid(True)
plt.legend()
plt.xlabel("Time")
ax1a.set_ylabel("Temperature (K)")
ax1b.set_ylabel("Resistance (Ohms)")
ax1a.set_title("Cooldown over Time")

ax1b.set_ylim([0, 1000])

mplcursors.cursor(multiple=True)

plt.show()