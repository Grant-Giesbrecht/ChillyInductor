''' Used to interactively match an L and C model to the measured data.

'''

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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
DEFAULT_Z0 = 50
DEFAULT_VP = 0.3
PHYS_LEN = 0.5

log = LogPile()
log.set_terminal_level(DEBUG)

c = 3e8

#------------------------------------------------------------
# Import Data

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C1*E', 'Track 2*', 's_p*', '9Sept*', 'narrow'])
filename = "cryostat.s2p"

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R3C4*B', 'Track 1 4mm', 'VNA Traces'])
# filename = "25June2024_Mid.csv"

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 2 43mm', 'Uncalibrated SParam', 'Prf -30 dBm'])
# filename = "26Aug2024_Ch1ToCryoL_CryoRTerm.csv"

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm', 'VNA Traces'])
# filename = "Sparam_31July2024_-30dBm_R4C4T1_Wide.csv"


if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

sp_filename = os.path.join(datapath, filename)
try:
	if sp_filename[-4:].lower() == '.csv':
		data = read_rohde_schwarz_csv(sp_filename)
	else:
		data = read_s2p(sp_filename)
except Exception as e:
	print(f"Failed to read CSV file. {e}")
	sys.exit()

#------------------------------------------------------------
# Define Functions

N = 3001
FREQ_MAX = max(data.freq_Hz)
FREQ_MIN = min(data.freq_Hz)
sim_freqs = np.linspace(FREQ_MIN, FREQ_MAX, N)
conditions = {"C": DEFAULT_C, "L": DEFAULT_L, "G": DEFAULT_G, 'LENGTH_m': PHYS_LEN, "Z0": DEFAULT_Z0, "Vp": DEFAULT_VP, 'ZOOM':1, 'ZOOM_CENTER':(FREQ_MAX+FREQ_MIN)//2}

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
# ax1 = axes[0]
# ax2 = axes[1]
fig.subplots_adjust(left=0.065, bottom=0.35, top=0.99, right=0.85)

# ax1.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S11)), linestyle='--', marker='.', markersize=2, color=(0.7, 0, 0), label='S_11', picker=10)
# ax1.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S21)), linestyle='--', marker='.', markersize=2, color=(0, 0, 0.7), label="S_21", picker=10)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S-Parameters (dB)")
# plt.grid(True)
plt.legend(['Measured', 'Simulated'])
plt.title(f"File: {filename}")
# plt.ylim([-60, 10])

mplcursors.cursor(multiple=True)
# fig.canvas.callbacks.connect('pick_event', on_pick)

# Create sliders - Vp Z0
ax_Z0 = fig.add_axes([0.1, 0.2, 0.8, 0.03])
slider_Z0 = Slider(ax_Z0, 'Z0 (Ohms)', 0, 250, valinit=conditions['Z0'], valfmt='%d')


ax_Vp = fig.add_axes([0.1, 0.15, 0.8, 0.03])
slider_Vp = Slider(ax_Vp, 'Vp/c (1)', 0, 1, valinit=conditions['Vp']/c, valfmt='%d')


# Create slider - L C
ax_L = fig.add_axes([0.1, 0.05, 0.8, 0.03])
slider_L = Slider(ax_L, 'L (nH)', 0, 1000, valinit=conditions['L']*1e9, valfmt='%d', color='green')


ax_C = fig.add_axes([0.1, 0.1, 0.8, 0.03])
slider_C = Slider(ax_C, 'C (pF)', 0, 1000, valinit=conditions['C']*1e12, valfmt='%d', color='green')

# Create zoom slider
ax_zoom = fig.add_axes([0.87, 0.4, 0.03, 0.55])
slider_zoom = Slider(ax_zoom, 'Zoom', 1, 10, valinit=conditions['ZOOM'], orientation='vertical', color=(0.6, 0, 0.6))

ax_center = fig.add_axes([0.92, 0.4, 0.03, 0.55])
slider_center = Slider(ax_center, 'Center', FREQ_MIN/1e9, FREQ_MAX/1e9, valinit=conditions['ZOOM_CENTER']/1e9, orientation='vertical', color=(0.6, 0, 0.6))

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.85, 0.3, 0.1, 0.04])
reset_button = Button(resetax, 'Reset View', hovercolor=(0.8, 0.6, 0.6))

def update_model_LC():
	
	# Get constants
	zl = 50
	z0 = np.sqrt(conditions['L']/conditions['C'])
	vp = 1/np.sqrt(conditions['L']*conditions['C'])
	i = complex(0, 1)
	
	conditions['Z0'] = z0
	conditions['Vp'] = vp
	
	# Calcualte electrical length
	elec_length_rad = conditions['LENGTH_m']*sim_freqs/vp*2*np.pi
	
	# Calculate input impedance
	zin = z0 * ( zl + i*z0*np.tan(elec_length_rad) )/( z0 + zl*i*np.tan(elec_length_rad) )
	
	# Calculate reflection coefficient
	gamma = (zin - zl)/(zin + zl)
	
	phase_gamma_deg = np.angle(gamma)*180/np.pi
	
	# Calculate S11 in dB
	s11_dB = lin_to_dB(np.abs(gamma), True)
	
	# Update axes
	ax1.cla()
	ax1.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S11)), linestyle='--', marker='.', markersize=2, color=(0.7, 0, 0), label='S_11', picker=10)
	ax1.plot(sim_freqs/1e9, s11_dB, linestyle='-', color=(0.7, 0.7, 0.3), linewidth=0.5, marker='o', markersize=1.5)
	
	ax1.set_xlabel("Frequency (GHz)")
	ax1.set_ylabel("S-Parameters (dB)")
	ax1.grid(True)
	ax1.legend(['Measured', 'Simulated'])
	
	ax2.cla()
	ax2.plot(data.freq_Hz/1e9, np.angle(S11)*180/np.pi, linestyle='--', marker='.', markersize=2, color=(0.7, 0, 0), label='S_11', picker=10)
	ax2.plot(sim_freqs/1e9, phase_gamma_deg, linestyle='-', color=(0.7, 0.7, 0.3), linewidth=0.5, marker='o', markersize=1.5)
	
	ax2.set_xlabel("Frequency (GHz)")
	ax2.set_ylabel("Phase (deg)")
	ax2.grid(True)
	ax2.legend(['Measured', 'Simulated'])
	
	if conditions['ZOOM'] != 1:
		span = (FREQ_MAX - FREQ_MIN)/1e9
		new_span = span/conditions['ZOOM']
		fstart = conditions['ZOOM_CENTER']/1e9-new_span/2
		fend = conditions['ZOOM_CENTER']/1e9+new_span/2
		ax1.set_xlim([fstart, fend])
		ax2.set_xlim([fstart, fend])
	
	ax1.set_ylim([-60, 10])
	ax2.set_ylim([-180, 180])
	
	# Update slider positions
	slider_Vp.eventson = False
	slider_Z0.eventson = False
	slider_Vp.set_val(vp/c)
	slider_Z0.set_val(z0)

	fig.canvas.draw()
	slider_Vp.eventson = True
	slider_Z0.eventson = True

def update_model_ZV():
	
	# Get constants
	zl = 50
	z0 = conditions['Z0']
	vp = conditions['Vp']
	conditions['L'] = z0/vp
	conditions['C'] = 1/vp/z0
	i = complex(0, 1)
	
	# Calcualte electrical length
	elec_length_rad = conditions['LENGTH_m']*sim_freqs/vp*2*np.pi
	
	# Calculate input impedance
	zin = z0 * ( zl + i*z0*np.tan(elec_length_rad) )/( z0 + zl*i*np.tan(elec_length_rad) )
	
	# Calculate reflection coefficient
	gamma = (zin - zl)/(zin + zl)
	
	phase_gamma_deg = np.angle(gamma)*180/np.pi
	
	# Calculate S11 in dB
	s11_dB = lin_to_dB(np.abs(gamma), True)
	
	# Update axes
	ax1.cla()
	ax1.plot(data.freq_Hz/1e9, lin_to_dB(np.abs(S11)), linestyle='--', marker='.', markersize=2, color=(0.7, 0, 0), label='S_11', picker=10)
	ax1.plot(sim_freqs/1e9, s11_dB, linestyle='-', color=(0.7, 0.7, 0.3), linewidth=0.5, marker='o', markersize=1.5)
	
	ax1.set_xlabel("Frequency (GHz)")
	ax1.set_ylabel("S-Parameters (dB)")
	ax1.grid(True)
	ax1.legend(['Measured', 'Simulated'])
	
	ax2.cla()
	ax2.plot(data.freq_Hz/1e9, np.angle(S11)*180/np.pi, linestyle='--', marker='.', markersize=2, color=(0.7, 0, 0), label='S_11', picker=10)
	ax2.plot(sim_freqs/1e9, phase_gamma_deg, linestyle='-', color=(0.7, 0.7, 0.3), linewidth=0.5, marker='o', markersize=1.5)
	
	ax2.set_xlabel("Frequency (GHz)")
	ax2.set_ylabel("Phase (deg)")
	ax2.grid(True)
	ax2.legend(['Measured', 'Simulated'])
	
	if conditions['ZOOM'] != 1:
		span = (FREQ_MAX - FREQ_MIN)/1e9
		new_span = span/conditions['ZOOM']
		fstart = conditions['ZOOM_CENTER']/1e9-new_span/2
		fend = conditions['ZOOM_CENTER']/1e9+new_span/2
		ax1.set_xlim([fstart, fend])
		ax2.set_xlim([fstart, fend])
	
	ax1.set_ylim([-60, 10])
	ax2.set_ylim([-180, 180])
	
	print(f"")
	
	# Update slider positions
	slider_L.eventson = False
	slider_C.eventson = False
	slider_L.set_val(conditions['L']*1e9)
	slider_C.set_val(conditions['C']*1e12)
	fig.canvas.draw()
	slider_L.eventson = True
	slider_C.eventson = True


def update_C(new_C):
	conditions['C'] = new_C*1e-12
	log.debug(f"Chaning C to {new_C} pF")
	update_model_LC()

def update_L(new_L):
	conditions['L'] = new_L*1e-9
	log.debug(f"Chaning L to {new_L} nH")
	update_model_LC()

def update_Z0(new_Z0):
	conditions['Z0'] = new_Z0
	update_model_ZV()

def update_Vp(new_vp):
	conditions['Vp'] = new_vp*c
	update_model_ZV()

def update_zoom(new_zoom):
	conditions['ZOOM'] = new_zoom
	update_model_LC()

def update_zoom_center(new_center):
	conditions['ZOOM_CENTER'] = new_center*1e9
	update_model_LC()

def view_reset(event):
	slider_center.reset()
	slider_zoom.reset()


reset_button.on_clicked(view_reset)

slider_Z0.on_changed(update_Z0)
slider_Vp.on_changed(update_Vp)
slider_L.on_changed(update_L)
slider_C.on_changed(update_C)
slider_zoom.on_changed(update_zoom)
slider_center.on_changed(update_zoom_center)

update_model_LC()

# # Save pickled-figs
# pickle.dump(fig, open(os.path.join("..", "Figures", f"AS2-fig1-{filename}.pklfig"), 'wb'))

plt.show()
