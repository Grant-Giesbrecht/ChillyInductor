import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime
import hashlib
import matplotlib.animation as animation
import copy

from pylogfile.base import markdown


#=========================== Create initial wave ===========================
f0_Hz = 4.815e9

ampl = 1
b = 0e-9
sigma = 5e-9
omega = np.pi*2*f0_Hz
phi = 0

# Set some coefficients
L0 = 1
I_star = 0.25
C0 = 1
nonlinear_region = [15e-9, 25e-9]

k = 2*np.pi*f0_Hz*np.sqrt(L0*C0)

fig1 = plt.figure(figsize=(12,6))
# fig1.suptitle(f"ID hash: {cond_hash}", fontsize=8)
gs = fig1.add_gridspec(1, 1)
ax0 = fig1.add_subplot(gs[0, 0])
# ax1 = fig1.add_subplot(gs[1, 0])

art1, = ax0.plot([], [], linestyle=':', marker='.', color=(0.35, 0.3, 0.65))
# art2, = ax0.plot([0, 1, 2, 3, 4, 5], [.1, .1, .2, .1, .3, .2])
art2 = ax0.fill_between([nonlinear_region[0]/1e-9, nonlinear_region[1]/1e-9], [-100, -100], [100, 100], color=(0.4, 0.4, 0.4), alpha=0.2)

x_pos0 = np.arange(-30e-9, 110e-9, step=0.01e-9)
x_pos = copy.deepcopy(x_pos0)
tmax = 50e-9
dt = 0.25e-9

stop_idx = round(tmax/dt)

# Initialize waveform
envelope = ampl*np.exp( -(k*x_pos-k*b)**2/(2*(k*sigma)**2) )
wave = envelope * np.sin(k*x_pos)

L_prime = L0*(1 + envelope**2/I_star**2) # Calcualte total
v_phase = 1/np.sqrt(L_prime*C0) # Calculate phase velocity

def init_ani():
	global art1, art2
	
	# ax0.cla()
	
	# art1, = ax0.plot([], [], linestyle=':', marker='.', color=(0.35, 0.3, 0.65))
	# art2 = ax0.fill_between([nonlinear_region[0]/1e-9, nonlinear_region[1]/1e-9], [-100, -100], [100, 100], color=(0.4, 0.4, 0.4), alpha=0.2)
	
	ax0.set_xlabel("Position (nm)")
	ax0.set_ylabel("Amplitude (1)")
	ax0.grid(True)
	ax0.set_xlim([x_pos[0]/1e-9, x_pos[-1]/1e-9])
	ax0.set_ylim([-ampl*2, ampl*2])
	
	# ax1.set_xlim([0, len(times_s)])
	# ax1.set_ylim([-100, 100])
	
	return [art1, art2]

vp = np.zeros(len(x_pos))
x_pos_f = []

def make_ani_frame(loop_idx):
	global x_pos, art2, vp, stop_idx, x_pos_f
	
	print(f"{loop_idx}: {np.min(x_pos)} -> {np.max(x_pos)}")
	
	for idx, x in enumerate(x_pos):
		if x < nonlinear_region[0] or x > nonlinear_region[1]:
			vp[idx] = 1/np.sqrt(L0*C0)
		else:
			vp[idx] = v_phase[idx]
	
	x_pos = x_pos + vp*dt
	
	art1.set_data(x_pos/1e-9, wave)
	# art2.set_data(a2x, a2y)
	
	# Reset when at end
	if loop_idx+1 >= stop_idx:
		x_pos_f = copy.deepcopy(x_pos)
		x_pos = copy.deepcopy(x_pos0)
		# art1.set_data([], [])
		print(f"reset (stop_idx = {stop_idx})", flush=True)
		
	return [art1, art2]

ani = animation.FuncAnimation(fig1, make_ani_frame, init_func=init_ani, frames=stop_idx, interval=20, blit=True, repeat=False)

plt.show()

plt.plot(x_pos_f*1e9, wave)
plt.show()




# def guassian_sine (x, t, ampl, b, sigma, omega, phi, k):
# 	return np.sin(omega*x-k*t-phi)*ampl*np.exp( -(x-b-k*t)**2/(2*sigma**2) )


# ims = []
# for i in range(60):
# 	x += np.pi / 15.
# 	y += np.pi / 20.
	
# 	k = 
# 	y = np.sin(k*x - omega*t)
	
# 	im = plt.imshow(f(x, y), animated=True)
# 	ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

FFwriter = animation.FFMpegWriter(fps=50)
ani.save('./SiP2.mp4', writer=FFwriter)



# plt.show()
