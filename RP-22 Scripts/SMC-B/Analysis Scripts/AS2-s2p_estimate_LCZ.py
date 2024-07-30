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
import mplcursors

#------------------------------------------------------------
# Import Data

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R3C4*B', 'Track 1 4mm', 'VNA Traces'])
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
	
try:
	data = read_rohde_schwarz_csv(os.path.join(datapath, "25June2024_Mid.csv"))
except Exception as e:
	print(f"Failed to read CSV file. {e}")

#------------------------------------------------------------
# Define Functions

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

#------------------------------------------------------------
# Plot Data

S11 = data.S11_real + complex(0, 1)*data.S11_imag
S21 = data.S21_real + complex(0, 1)*data.S21_imag

fig, ax = plt.subplots()

ax.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S11)), linestyle='--', marker='.', markersize=2, color=(0.7, 0, 0), label='S_11', picker=10)
ax.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S21)), linestyle='--', marker='.', markersize=2, color=(0, 0, 0.7), label="S_21", picker=10)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S-Parameters (dB)")
plt.grid(True)
plt.legend()

mplcursors.cursor(multiple=True)
# fig.canvas.callbacks.connect('pick_event', on_pick)

plt.show()



