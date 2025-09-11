
"""
run_nltl_demo.py
----------------
Runs both solvers in nltl_sim.py and saves a couple of quick plots:
 - Midline voltage vs time
 - Space-time voltage colormap for the FDTD

This is intentionally lightweight so you can tweak parameters quickly.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from nltl_sim import demo_ladder, demo_fdtd

# -----------
# Ladder demo
# -----------
ladder_out, ladder_p = demo_ladder()
t = ladder_out.t
v_mid = ladder_out.v_nodes[:, ladder_p.N//2]

plt.figure()
plt.plot(t, v_mid, linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('V_hist(mid node) (V_hist)')
plt.title('Nonlinear Ladder: Midline Voltage vs Time')
plt.tight_layout()
# plt.savefig('./data/ladder_mid_voltage.png', dpi=150)

# ----------
# FDTD demo
# ----------
fdtd_out, fdtd_p = demo_fdtd()
t = fdtd_out.t
x = fdtd_out.x
V_hist = fdtd_out.v_xt

# Plot v at mid-point vs time
v_mid = V_hist[:, V_hist.shape[1]//2]
plt.figure()
plt.plot(t, v_mid, linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('V_hist(x=L/2) (V_hist)')
plt.title('Nonlinear FDTD: Midline Voltage vs Time')
plt.tight_layout()
# plt.savefig('./data/fdtd_mid_voltage.png', dpi=150)

# Space-time colormap (one figure only per tool rules)
plt.figure()
extent = [x[0], x[-1], t[-1], t[0]]  # flip time axis so earlier at top
plt.imshow(V_hist, aspect='auto', extent=extent)  # default colormap
plt.xlabel('Position x (m)')
plt.ylabel('Time t (s)')
plt.title('Nonlinear FDTD: V_hist(x,t) Space-Time')
plt.tight_layout()
# plt.savefig('./data/fdtd_space_time.png', dpi=150)

# print("Saved:")
# print(" - ./data/ladder_mid_voltage.png")
# print(" - ./data/fdtd_mid_voltage.png")
# print(" - ./data/fdtd_space_time.png")

# Create figure and turn on interactive mode
plt.ion()
fig1 = plt.figure(figsize=(12,6))
# fig1.suptitle(f"ID hash: {cond_hash}, ({cond_hash_abb})", fontsize=8)
gs1 = fig1.add_gridspec(2, 1)
ax0a = fig1.add_subplot(gs1[0, 0])
ax0b = fig1.add_subplot(gs1[1, 0])

# Initialize axes
ax0a.set_xlabel("Position (m)")
ax0a.set_ylabel("Amplitude (V)")
ax0a.grid(True)
ax0a.set_xlim([np.min(x), np.max(x)])
ax0a.set_ylim([-1, 1])

# Initialize artists
pulse_artist_0a, = ax0a.plot([], [], linestyle=':', marker='.', color=(0.35, 0.3, 0.65))
pulse_artist_0b, = ax0a.plot([], [], linestyle=':', marker='.', color=(0.35, 0.3, 0.65))

# Decimate V_hist for faster playback
decimation = 2
frame_jump = 10
V_hist_dec = V_hist[::frame_jump, ::decimation]
t_dec = t[::frame_jump]
x_dec = x[::decimation]

# Loop over each frame
t_frame = 1/30
t_last = time.time()
limit_framerate = False
print(f"Dimensions of V_hist_dec: {V_hist_dec.shape}")
idx = 0
for v_frame in V_hist_dec:
	
	# Update frame
	pulse_artist_0a.set_data(x_dec, v_frame)
	fig1.canvas.draw()
	
	# Flush events (neccesary but I don't know why)
	fig1.canvas.flush_events()
	
	# Pause if neccesary
	if limit_framerate:
		while time.time() < (t_last+t_frame):
			time.sleep(0.001)
	t_now = time.time()
	fps = 1/(t_now-t_last)
	print(f"FPS = {fps}")
	t_last = t_now
	
	idx += 1
