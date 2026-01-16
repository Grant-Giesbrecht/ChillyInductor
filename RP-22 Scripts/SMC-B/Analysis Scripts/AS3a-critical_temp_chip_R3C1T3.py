'''
Uses an S2P file to estimate L, C and Z of a chip.
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

def rounding_bin(x:list, y:list, step:float=0.2, tol:float=0.05):

	def round_x(val:float, rounding_thresh:float):
		return round(val/rounding_thresh)*rounding_thresh
	
	# xmin = np.min(x)
	# xmax = np.max(x)
	
	# # Get list of values to round to
	# rounding_points = np.arrange(0, np.ceil(xmax/step)*step, step=step)
	
	x_out = []
	y_out = []
	
	used_x_bins = []
	bin_ys = []
	y_stds = []
	y_means = []
	
	# Scan over x list
	for yv, xv in zip(y, x):
		
		# Find closest rounding point
		rp = round_x(xv, step)
		delta = np.abs(rp - xv)
		
		# If delta is less than tolerance, add point
		if delta <= tol:
			x_out.append(rp)
			y_out.append(yv)
			
			# Add to std lists
			if rp in used_x_bins:
				
				idx = used_x_bins.index(rp)
				bin_ys[idx].append(yv)
			else:
				used_x_bins.append(rp)
				bin_ys.append([yv])
	
	# Calculate standard deviations
	for xv, ylists in zip(used_x_bins, bin_ys):
		y_means.append(np.mean(ylists))
		y_stds.append(np.std(ylists))
		
		print(f"X: {xv}, len(y)={len(ylists)}, std={np.std(ylists)}, mean={np.mean(ylists)}, min={np.min(ylists)}, max={np.max(ylists)}")
	
	return (x_out, y_out, used_x_bins, y_means, y_stds)
			

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R3C1*D', 'Track 3 454mm', 'cooldown'])
# if datapath is None:
# 	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
# 	sys.exit()
# else:
# 	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
# filename = "RP22B_MS2_t1_04Oct2024_R3C1T3_r1.hdf"
#
# analysis_file = os.path.join(datapath, filename)

analysis_file = os.path.join("C:\\", "Users", "gmg3", "Documents", "GitHub", "ChillyInductor", "RP-22 Scripts", "SMC-B", "Measurement Scripts", "data", "RP24B_MS2_15Dec2025_R2C0W7BT3_r1.hdf")

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

##--------------------------------------------
# Plot Results

fig1 = plt.figure(1)
ax1 = fig1.gca()
ax2 = ax1.twinx()

timestamps_seconds = [(ts - timestamps[0]).total_seconds() for ts in timestamps]

# START_IDX = 1350
# END_IDX = 3000
# line1, = ax1.plot(timestamps[START_IDX:END_IDX], temperatures[START_IDX:END_IDX], linestyle=':', label="Temp (K)", color=(0, 0, 0.7), marker='.')
# line2, = ax2.plot(timestamps[START_IDX:END_IDX], np.array(resistances[START_IDX:END_IDX]-sc_res)/1e3, linestyle='--', label="Resistance (Ohms)", color=(0.7, 0, 0), marker='.')
# plt.grid(True)
# plt.legend([line1, line2], ["Temp (K)", "Resistance ($\Omega$)"])
# plt.xlabel("Time")
# ax1.set_ylabel("Temperature (K)")
# ax2.set_ylabel("Resistance (Ohms)")
# ax1.set_title("Cooldown over Time")

START_IDX = 1350
END_IDX = 3000
line1, = ax1.plot(np.array(timestamps_seconds[START_IDX:END_IDX])-timestamps_seconds[START_IDX], temperatures[START_IDX:END_IDX], linestyle=':', label="Temp (K)", color=(0, 0, 0.7), marker='.')
line2, = ax2.plot(np.array(timestamps_seconds[START_IDX:END_IDX])-timestamps_seconds[START_IDX], np.array(resistances[START_IDX:END_IDX]-sc_res)/1e3, linestyle='--', label="Resistance (Ohms)", color=(0.7, 0, 0), marker='.')
plt.grid(True)
plt.legend([line1, line2], ["Temp (K)", "Resistance ($\Omega$)"])
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Temperature (K)")
ax2.set_ylabel("Resistance (k$\Omega$)")
ax1.set_title("Critical Temperature Measurement")
plt.show()

sys.exit()

fig2 = plt.figure(2)
ax1_2 = fig2.gca()
ax1_2.plot(temperatures, resistances-sc_res, linestyle=':', marker='.', color=(0.7, 0, 0))
ax1_2.set_ylabel("Resistance (Ohms)")
ax1_2.set_xlabel("Temperature (K)")
ax1_2.set_title("Resistance over Temperature")
# ax1_2.set_xlim([2.8, 4.4])
plt.grid(True)

xytup = rounding_bin(x=temperatures[START_IDX:END_IDX], y=(resistances[START_IDX:END_IDX]-sc_res), step=0.2, tol=0.05)
temp_rd = xytup[0]
res_rd = np.array(xytup[1])
temp_bins = np.array(xytup[2])
res_means = np.array(xytup[3])
res_std = np.array(xytup[4])

fig3 = plt.figure(3)
ax1_3 = fig3.gca()
ax1_3.scatter(temp_rd, res_rd, marker='.', color=(0.7, 0, 0))
ax1_3.errorbar(temp_bins, res_means, yerr=res_std, color=(0.2, 0.2, 0.6), elinewidth=0.5)
ax1_3.fill_between(temp_bins, res_means - 2*res_std, res_means + 2*res_std, color=(0.2, 0.2, 0.6), alpha=0.2)
ax1_3.fill_between(temp_bins, res_means - res_std, res_means + res_std, color=(0.2, 0.2, 0.6), alpha=0.4)

ax1_3.set_ylabel("Resistance (Ohms)")
ax1_3.set_xlabel("Binned Temperature (K)")
ax1_3.set_title("Resistance over Temperature")
# ax1_2.set_xlim([2.8, 4.4])
plt.grid(True)

plt.rcParams['font.size'] = 14

fig4 = plt.figure(4, figsize=(10, 5))
ax1_4 = fig4.gca()
ax1_4.fill_between(temp_bins, res_means/1e3 - 3*res_std/1e3, res_means/1e3 + 3*res_std/1e3, color=(0.2, 0.2, 0.6), alpha=0.075, linewidth=0)
ax1_4.fill_between(temp_bins, res_means/1e3 - 2*res_std/1e3, res_means/1e3 + 2*res_std/1e3, color=(0.2, 0.2, 0.6), alpha=0.075, linewidth=0)
ax1_4.fill_between(temp_bins, res_means/1e3 - 1.75*res_std/1e3, res_means/1e3 + 1.75*res_std/1e3, color=(0.2, 0.2, 0.6), alpha=0.075, linewidth=0)
ax1_4.fill_between(temp_bins, res_means/1e3 - 1.5*res_std/1e3, res_means/1e3 + 1.5*res_std/1e3, color=(0.2, 0.2, 0.6), alpha=0.075, linewidth=0)
ax1_4.fill_between(temp_bins, res_means/1e3 - 1.25*res_std/1e3, res_means/1e3 + 1.25*res_std/1e3, color=(0.2, 0.2, 0.6), alpha=0.075, linewidth=0)
ax1_4.fill_between(temp_bins, res_means/1e3 - res_std/1e3, res_means/1e3 + res_std/1e3, color=(0.2, 0.2, 0.6), alpha=0.3)
# ax1_4.scatter(temp_rd, res_rd/1e3, marker='.', color=(0.7, 0, 0))
ax1_4.scatter(temp_rd, res_rd/1e3, marker='.', color=(0, 0.7, 0.9), edgecolors=(0, 0, 0.6))

ax1_4.set_ylabel("Resistance (kOhms)")
ax1_4.set_xlabel("Measured Temperature (K)")
ax1_4.set_title("Critical Tempertature Measurement, 3.7 $\mu m$ Device")
# ax1_2.set_xlim([2.8, 4.4])
plt.grid(True)



# Save pickled-figs
pickle.dump(fig1, open(os.path.join("..", "Figures", f"AS3-fig1-{filename}.pklfig"), 'wb'))
pickle.dump(fig2, open(os.path.join("..", "Figures", f"AS3-fig2-{filename}.pklfig"), 'wb'))

fig4.savefig(os.path.join("saved_figures", "AS3a_fig4.png"), dpi=500)

# # Save pickled-figs
# pickle.dump(fig1, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig1.pklfig"), 'wb'))
# pickle.dump(fig2, open(os.path.join("..", "Figures", "RP22B-AS1-MS1 24Jul2024-r1 fig2.pklfig"), 'wb'))












plt.show()



