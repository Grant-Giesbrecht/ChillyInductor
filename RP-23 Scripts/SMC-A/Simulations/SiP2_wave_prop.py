import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime
import hashlib
import matplotlib.animation as animation


from pylogfile.base import markdown


#=========================== Create initial wave ===========================
f0_Hz = 4.815e9

ampl = 1
b = 10e-9
sigma = 5e-9
omega = np.pi*2*f0_Hz
phi = 0

# Set some coefficients
length = 1
L0 = 1
I_star = 5
C0 = 1

k = 2*np.pi*f0_Hz*np.sqrt(L0*C0)

fig1 = plt.figure(figsize=(12,6))
# fig1.suptitle(f"ID hash: {cond_hash}", fontsize=8)
gs = fig1.add_gridspec(2, 1)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[1, 0])

art1, = ax0.plot([], [], linestyle=':', marker='.', color=(0.35, 0.3, 0.65))
art2, = ax1.plot([], [], linestyle=':', marker='.', color=(1, 0.5, 0.05))

x_pos = np.arange(-30e-9, 100e-9, step=0.01e-9)
times_s = np.arange(0, 80e-9, 1e-9)


def init_ani():
	
	ax0.set_xlabel("Position (nm)")
	ax0.set_ylabel("Amplitude (1)")
	ax0.grid(True)
	ax0.set_xlim([x_pos[0]/1e-9, x_pos[-1]/1e-9])
	ax0.set_ylim([-ampl*2, ampl*2])
	
	ax1.set_xlim([0, len(times_s)])
	ax1.set_ylim([-100, 100])
	
	return [art1, art2]

a2x = []
a2y = []

def make_ani_frame(idx):
	
	# Select time
	t = times_s[idx]
	
	# Create time points when wavefronts occur
	envelope = ampl*np.exp( -(k*x_pos-k*b-k*t)**2/(2*(k*sigma)**2) )
	
	# envelope = 1
	wave = envelope * np.sin(k*x_pos)# - omega*t)
	
	a2x.append(idx)
	a2y.append(omega*t)
	
	art1.set_data(x_pos/1e-9, wave)
	art2.set_data(a2x, a2y)
	
	
	# x = np.linspace(0, 10, 1000)
	# y = np.sin(2 * np.pi * (x - 0.01 * idx))
	# art1.set_data(x, y)
	# art2.set_data(x, y*1.25)
	
	return [art1, art2]
	
	# #================ Plot wave ======================
	
	
	# ax0.plot(x_pos/1e-9, wave, linestyle=':', marker='.', color=(0.35, 0.3, 0.65))

ani = animation.FuncAnimation(fig1, make_ani_frame, init_func=init_ani, frames=len(times_s), interval=20, blit=True)

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

# FFwriter = animation.FFMpegWriter(fps=10)
# ani.save('./dynamic_images.mp4', writer=FFwriter)



# plt.show()
