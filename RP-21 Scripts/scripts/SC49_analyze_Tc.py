import matplotlib.pyplot as plt
import pickle
import os
from colorama import Fore, Style
import numpy as np

V_NORM = 0.0025

# Default file name
# fn = os.path.join("/Volumes/M4 PHD/datafile_transfer/", "Tcrit_04Mar2024_R01.pkl")
fn = os.path.join("/", "Volumes", "M6 T7S", "ARC0 PhD Data", "RP-21 Kinetic Inductance 2023", "Data", "group4_extflash", "2024 Data", "Tcrit_04Mar2024_R03.pkl")

# Read data
with open(fn, 'rb') as f:
	meister_data = pickle.load(f)
data = meister_data['points']

# Unwrap data
temp_K = [x['temp_K'] for x in data]
mfli_V = [x['cryo_voltage_V'] for x in data]

# print(f"{Fore.RED}Warning: Remove these debug points")
# temp_K.insert(0, 0.1)
# mfli_V.insert(0, 0)
# temp_K.insert(0, 0)
# mfli_V.insert(0, 0)

# Find first point above normal voltage
region_sc = []
region_norm = []
for Vm, Tk in zip(mfli_V, temp_K):
	
	if Vm < V_NORM:
		region_sc.append(Tk)
	else:
		region_norm.append(Tk)

# ID critical temperature
if len(region_norm) > 0 and len(region_sc) > 0:
	Tmax = min(region_norm)
	Tmin = max(region_sc)
	print(f"{Fore.WHITE}Tc region: {min([Tmin, Tmax])} - {max([Tmin, Tmax])} K{Style.RESET_ALL}")
	print(f"    -> {Fore.CYAN}{(Tmax+Tmin)/2} +/- {round(abs(Tmax-Tmin)/2*1e4)/10} mK{Style.RESET_ALL}")

# Make plot
plt.figure(1)
plt.plot(temp_K, mfli_V, linestyle=':', marker='.')
plt.grid()
plt.ylim(bottom=0)
plt.xlabel("Temperature (K)")
plt.ylabel("Voltage (V)")
plt.title("Voltage Across Cryostat")

yL, yH = plt.ylim()

if len(region_norm) >= 2:
	xL = min(region_norm)
	xH = max(region_norm)
	plt.fill([xL, xL, xH, xH], [yL, yH, yH, yL], alpha=0.2, color=(0.8, 0, 0))

if len(region_sc) >= 2:
	xL = min(region_sc)
	xH = max(region_sc)
	plt.fill([xL, xL, xH, xH], [yL, yH, yH, yL], alpha=0.2, color=(0, 0, 0.8))

def line_picker(line, mouseevent):
	"""
	Find the points within a certain distance from the mouseclick in
	data coords and attach some extra attributes, pickx and picky
	which are the data points that were picked.
	"""
	if mouseevent.xdata is None:
		return False, dict()
	xdata = line.get_xdata()
	ydata = line.get_ydata()
	maxd = 0.05
	d = np.sqrt(
		(xdata - mouseevent.xdata)**2 + (ydata - mouseevent.ydata)**2)
	
	index_min = min(range(len(d)), key=d.__getitem__)
	pickx = xdata[index_min]
	picky = ydata[index_min]
	props = dict(ind=index_min, pickx=pickx, picky=picky)
	return True, props


def onpick(event):
	
	for idx, l in enumerate(event.canvas.figure.axes[0].lines):
		
		if l.get_url() == "click_marker":
			event.canvas.figure.axes[0].lines[idx].remove()
	
	event.canvas.figure.axes[0].plot([event.pickx], [event.picky], marker='+', color=(1, 0, 0), url="click_marker")
	plt.draw()
	
	print(f"\t{Fore.WHITE}Selected point: {Fore.CYAN}{event.pickx}{Fore.WHITE}, {Fore.CYAN}{event.picky}{Style.RESET_ALL}")


fig = plt.figure(2)
plt.plot(temp_K, mfli_V, linestyle=':', marker='.', picker=line_picker)
plt.grid()
plt.ylim(bottom=0)
plt.xlim(14, 14.75)
plt.xlabel("Temperature (K)")
plt.ylabel("Voltage (V)")
plt.title("Voltage Across Cryostat (Zoomed)")

fig.canvas.callbacks.connect('pick_event', onpick)

if len(region_norm) >= 2:
	xL = min(region_norm)
	xH = max(region_norm)
	plt.fill([xL, xL, xH, xH], [yL, yH, yH, yL], alpha=0.2, color=(0.8, 0, 0))

if len(region_sc) >= 2:
	xL = min(region_sc)
	xH = max(region_sc)
	plt.fill([xL, xL, xH, xH], [yL, yH, yH, yL], alpha=0.2, color=(0, 0, 0.8))

plt.show()

