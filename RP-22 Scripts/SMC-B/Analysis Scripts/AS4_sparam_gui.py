''' Used to interactively match an L and C model to the measured data.

'''

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
import os
from ganymede import *
from pylogfile.base import *
import sys
import numpy as np
import mplcursors
import pickle

DEFAULT_C = 80e-12
DEFAULT_L = 126e-9
DEFAULT_G = 0.48

log = LogPile()
log.set_terminal_level(DEBUG)

#------------------------------------------------------------
# Import Data

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R3C4*B', 'Track 1 4mm', 'VNA Traces'])
# filename = "25June2024_Mid.csv"

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm', 'VNA Traces'])
filename = "Sparam_31July2024_-30dBm_R4C4T1.csv"
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
	
try:
	data = read_rohde_schwarz_csv(os.path.join(datapath, filename))
except Exception as e:
	print(f"Failed to read CSV file. {e}")

#------------------------------------------------------------
# Define Functions

N = 1001
sim_freqs = np.linspace(min(data.freq_Hz), max(data.freq_Hz), N)
conditions = {"C": DEFAULT_C, "L": DEFAULT_L, "G": DEFAULT_G, 'LENGTH_m': 0.0041}

def on_pick(event):
	print("Pick Event")
	artist = event.artist
	xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
	x, y = artist.get_xdata(), artist.get_ydata()
	ind = event.ind
	
	print('Artist picked:', event.artist)
	print('{} vertices picked'.format(len(ind)))
	print('Pick between vertices {} and {}'.format(min(ind), max(ind)+1))
	print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
	print('Data point:', x[ind[0]], y[ind[0]])
	print

def update_model():
	
	# Get constants
	zl = 50
	z0 = np.sqrt(conditions['L']/conditions['C'])
	vp = 1/np.sqrt(conditions['L']*conditions['C'])
	i = complex(0, 1)
	
	# Calcualte electrical length
	elec_length_rad = conditions['LENGTH_m']*sim_freqs/vp*2*np.pi
	
	# Calculate input impedance
	zin = z0 * ( zl + i*z0*np.tan(elec_length_rad) )/( z0 + zl*i*np.tan(elec_length_rad) )
	
	# Calculate reflection coefficient
	gamma = (zin - zl)/(zin + zl)
	
	# Calculate S11 in dB
	s11_dB = lin_to_dB(np.abs(gamma))
	
	# Update axes
	ax.cla()
	ax.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S11)), linestyle='--', marker='.', markersize=2, color=(0.7, 0, 0), label='S_11', picker=10)
	ax.plot(sim_freqs/1e9, s11_dB, linestyle=':', color=(0.7, 0.7, 0.3))
	
	ax.set_xlabel("Frequency (GHz)")
	ax.set_ylabel("S-Parameters (dB)")
	ax.grid(True)
	ax.legend(['Measured', 'Simulated'])
	# plt.title(f"File: {filename}")
	ax.set_ylim([-60, 10])

def update_C(new_C):
	conditions['C'] = new_C*1e-12
	log.debug(f"Chaning C to {new_C} pF")
	update_model()

def update_L(new_L):
	conditions['L'] = new_L*1e-9
	log.debug(f"Chaning L to {new_L} nH")
	update_model()

# def update_C(new_C):
# 	conditions['C'] = new_C
# 	update_model()


#------------------------------------------------------------
# Plot Data

S11 = data.S11_real + complex(0, 1)*data.S11_imag
S21 = data.S21_real + complex(0, 1)*data.S21_imag

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.35)

# ax.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S11)), linestyle='--', marker='.', markersize=2, color=(0.7, 0, 0), label='S_11', picker=10)
# ax.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S21)), linestyle='--', marker='.', markersize=2, color=(0, 0, 0.7), label="S_21", picker=10)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S-Parameters (dB)")
# plt.grid(True)
plt.legend(['Measured', 'Simulated'])
plt.title(f"File: {filename}")
# plt.ylim([-60, 10])

update_model()

mplcursors.cursor(multiple=True)
# fig.canvas.callbacks.connect('pick_event', on_pick)

# Create slider
ax_L = fig.add_axes([0.25, 0.1, 0.65, 0.03])
slider_L = Slider(ax_L, 'L (nH)', 0, 1000, valinit=conditions['L']*1e9, valfmt='%d')
slider_L.on_changed(update_L)

ax_C = fig.add_axes([0.25, 0.2, 0.65, 0.03])
slider_C = Slider(ax_C, 'C (pF)', 0, 1000, valinit=conditions['C']*1e12, valfmt='%d')
slider_C.on_changed(update_C)

# Save pickled-figs
pickle.dump(fig, open(os.path.join("..", "Figures", f"AS2-fig1-{filename}.pklfig"), 'wb'))

plt.show()
