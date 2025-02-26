'''
Modified version of SiP2a. Whereas SiP2a simulated the entire waveform, it was just for visual
coolness. The envelope and thus the phase velocity was calcualted once at the beginning.

SiP3 supports multiple tones by re-calculating the envelope and thus the phase velocity at each
frame of the simualtiosn.
'''

import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import pandas as pd
import numpy as np
import os
import datetime
import hashlib
import matplotlib.animation as animation
import copy
import time
from matplotlib.animation import FFMpegWriter
import argparse

from pylogfile.base import markdown

#======================= Check for command line arguments ==================

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--savemp4', help="Save MP4 of animation.", action='store_true')
args = parser.parse_args()

#================== Set Simulation Parameters =================

f0_Hz = 4.815e9
f1_Hz = 4.815e9*1.5

ampl = 1
b = 0e-9
sigma = 5e-9
phi = 0

# Set some coefficients
L0 = 1
I_star = 1
C0 = 1
nonlinear_region = [40e-9, 60e-9]

tmax = 120e-9
dt = 1e-9

x_start = -30e-9
x_end = 130e-9
x_step = 10e-12

mp4_dpi = 200
mp4_bitrate = 16000
mp4_fps = 40

#=========================== Initialization ===========================

datestr = f"{datetime.datetime.now()}"
cond_hash = hashlib.sha1(datestr.encode("utf-8")).hexdigest()
cond_hash_abb = cond_hash[-6:]

# Calculate wave number
k0 = 2*np.pi*f0_Hz*np.sqrt(L0*C0)
k1 = 2*np.pi*f1_Hz*np.sqrt(L0*C0)

# Create positions of wave points
x_pos0 = np.arange(x_start, x_end, step=x_step)
x_pos = copy.deepcopy(x_pos0)

# Create figure and turn on interactive mode
plt.ion()
fig1 = plt.figure(figsize=(12,6))
fig1.suptitle(f"ID hash: {cond_hash}, ({cond_hash_abb})", fontsize=8)
gs = fig1.add_gridspec(1, 1)
ax0 = fig1.add_subplot(gs[0, 0])

# Initialize axes
ax0.set_xlabel("Position (nm)")
ax0.set_ylabel("Amplitude (1)")
ax0.grid(True)
ax0.set_xlim([x_pos[0]/1e-9, x_pos[-1]/1e-9])
ax0.set_ylim([-ampl*2, ampl*2])

# Initialize artists
pulse_artist, = ax0.plot([], [], linestyle=':', marker='.', color=(0.35, 0.3, 0.65))
region_artist = ax0.fill_between([nonlinear_region[0]/1e-9, nonlinear_region[1]/1e-9], [-100, -100], [100, 100], color=(0.4, 0.4, 0.4), alpha=0.2)

# Calculate number of frames to calcualte
stop_idx = round(tmax/dt)

# Initialize waveform
envelope = ampl*np.exp( -(k0*x_pos-k0*b)**2/(2*(k0*sigma)**2) ) #TODO: Should this be k or something else?
wave = envelope * (np.sin(k0*x_pos) + np.sin(k1*x_pos))

#===================== Primary Frame Generation Logic ===================

def make_ani_frame(loop_idx, t_proceed):
	global x_pos, region_artist, vp, stop_idx, x_pos_f
	
	# envelope = ampl*np.exp( -(k0*x_pos-k0*b)**2/(2*(k0*sigma)**2) )
	
	#===================== Perform Hilbert transform ==========================
	# Goal is to provide tighter bounds on amplitude and slope
	
	# Perform hilbert transform and calculate derivative parameters
	a_signal = hilbert(ampl_mV)
	envelope = np.abs(a_signal)
	
	#===================== Calculate Frame Progression ======================
	
	# Initialize modified phase velocities
	L_prime = L0*(1 + envelope**2/I_star**2) # Calcualte total
	v_phase = 1/np.sqrt(L_prime*C0) # Calculate phase velocity

	# Initialize phase velocity list
	vp = np.zeros(len(x_pos))
	
	# Calculate phase velocity for each point
	for idx, x in enumerate(x_pos):
		if x < nonlinear_region[0] or x > nonlinear_region[1]:
			vp[idx] = 1/np.sqrt(L0*C0)
		else:
			vp[idx] = v_phase[idx]
	
	#TODO: I can no longer just change x-position, I now need to add modify the X and somehow add
	# Ys that are no longer on the same x grids. boooooo.
	#
	# I think the answer is track a third and second harm Y-value for each x point, and their correspoonding envelopes.
	# You look at their relative phases (also tracked) and use trig to determine what the envelope there should be.
	#
	# Voila. I don't want to do that now though, I'm busy!
	
	# Update the x-position of each point
	x_pos = x_pos + vp*dt
	
	# Update the plot
	pulse_artist.set_data(x_pos/1e-9, wave)
	fig1.canvas.draw()
	
	# Pause if neccesary
	while time.time() < t_proceed:
		time.sleep(0.001)
	
	# Flush events (neccesary but I don't know why)
	fig1.canvas.flush_events()
	
	return [x_pos, wave]

#================= Print conditions =============================

mp4_filename = f"SiP2_{cond_hash_abb}.mp4"

nlr0, nlr1 = nonlinear_region[0], nonlinear_region[1]
tab = "    "
print(markdown(f">:aSimulation Conditions Summary:<"))
print(markdown(f"{tab}>:aStimulus:<"))
print(markdown(f"{tab}{tab}ampl: >{ampl}<"))
print(markdown(f"{tab}{tab}b: >{b}<"))
print(markdown(f"{tab}{tab}sigma: >{sigma}<"))
print(markdown(f"{tab}{tab}f0 (GHz): >{f0_Hz/1e9}<"))
print(markdown(f"{tab}{tab}f1 (GHz): >{f1_Hz/1e9}<"))
print(markdown(f"{tab}{tab}phi: >{phi}<"))
print(markdown(f"{tab}>:aNonlinear Region:< "))
print(markdown(f"{tab}{tab}L0: >{L0}<"))
print(markdown(f"{tab}{tab}I_star: >{I_star}<"))
print(markdown(f"{tab}{tab}C0: >{C0}<"))
print(markdown(f"{tab}{tab}x-start (nm): >{nlr0*1e9}<"))
print(markdown(f"{tab}{tab}x-end (nm): >{nlr1*1e9}<"))
print(markdown(f"{tab}>:aSimulation Field:< "))
print(markdown(f"{tab}{tab}x-start (nm): >{x_start*1e9}<"))
print(markdown(f"{tab}{tab}x-end (nm): >{x_end*1e9}<"))
print(markdown(f"{tab}{tab}x-step (nm): >{x_step*1e9}<"))
print(markdown(f"{tab}{tab}time elapsed (ns): >{tmax*1e9}<"))
print(markdown(f"{tab}{tab}delta t (ns): >{dt*1e9}<"))
print(markdown(f"{tab}{tab}No. frames: >{stop_idx}<"))
print(markdown(f"{tab}>:aSaved Files:< "))
print(markdown(f"{tab}{tab}Save MP4?: >{args.savemp4}<"))
if args.savemp4:
	print(markdown(f"{tab}{tab}MP4 name: >{mp4_filename}<"))
	print(markdown(f"{tab}{tab}dpi: >{mp4_dpi}<"))
	print(markdown(f"{tab}{tab}bitrate: >{mp4_bitrate}<"))
print(markdown(f">:aSimulation Name:< >:q{os.path.basename(__file__)}<"))
print(markdown(f"{tab}Start date: >:q{datestr}<"))
print(markdown(f"{tab}ID hash: >:q{cond_hash}<, (>{cond_hash_abb}<)"))
print("")

#===================== Run animation ============================

# Prepare frame rate variables
tlast = time.time()-10 # Initialize to something way back
frame_time = 1/30

# Check if MP4 is to be saved
if args.savemp4:
	print(f"Saving MP4")
	
	# Create writer
	writer = FFMpegWriter(fps=mp4_fps, bitrate=mp4_bitrate)
	
	# Loop over each frame
	with writer.saving(fig1, mp4_filename, dpi=mp4_dpi): #, dpi=300):
		for i in range(stop_idx):
			
			# Update frame
			rval = make_ani_frame(i, tlast+frame_time)
			tlast = time.time()
			
			# Add to MP4
			writer.grab_frame()
else:
	print(f"Not saving MP4")
	
	# Loop over each frame
	for i in range(stop_idx):
		
		# Update frame
		rval = make_ani_frame(i, tlast+frame_time)
		tlast = time.time()

plt.ioff()

fig = plt.figure()
plt.plot(rval[0]/1e-9, rval[1], linestyle='--', marker='.', color=(0, 0, 0.7))
plt.grid(True)
plt.show()