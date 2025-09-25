
"""
run_nltl_demo.py (piecewise animation with region annotations)
--------------------------------------------------------------
Top:  FDTD V(x, t)
Bottom: Ladder V(x, t) (nodes mapped to x)
Adds vertical dashed lines at region boundaries and a compact
annotation showing Z0 and alpha for each region.

Notes:
- Uses piecewise demos: demo_fdtd_piecewise() and demo_ladder_piecewise()
- blit=False for backend robustness.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from nltl_sim_implicit import *


def run_fdtd_piecewise():
	Nx = 600
	L = 0.3
	Rs = 50.0
	RL = 50.0
	t0 = 0.7e-9
	sigma = 0.2e-9
	T  = 6e-9
	V0 = 0.6
	
	f0 = 8e9              # carrier frequency (Hz)
	omega0 = 2*np.pi*f0
	
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0) * np.sin(omega0 * t)

	reg = [
		FDTDRegion(x0=0.0,     x1=L/4, L0_per_m=1e-6, C_per_m=130e-12, alpha=1e-3),
		FDTDRegion(x0=L/4,     x1=3*L/4,   L0_per_m=1e-6, C_per_m=130e-12, alpha=10000),
		FDTDRegion(x0=3*L/4,     x1=L,   L0_per_m=1e-6, C_per_m=130e-12, alpha=1e-3),
	]

	dx = L / Nx
	Lmin = min(r.L0_per_m for r in reg)
	Cmax = max(r.C_per_m for r in reg)
	# dt = NLTFDTD_PW.cfl_dt(dx, Lmin, Cmax, safety=0.85)
	dt = NLTFDTD_PW.cfl_dt(dx, Lmin, Cmax, safety=0.5)
	
	print(f"dt = {dt*1e9} ns, 1/dt = {1/dt/1e6} MHz")
	
	
	p = FDTDParamsPW(Nx=Nx, L=L, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs, regions=reg, nonlinear_update="implicit")
	out = NLTFDTD_PW(p).run()
	return out, p

def run_ladder_piecewise():
	N = 240
	L = 0.24
	Rs = 50.0
	RL = 50.0
	t0 = 0.7e-9
	sigma = 0.18e-9
	V0 = 0.6
	
	f0 = 8e9              # carrier frequency (Hz)
	omega0 = 2*np.pi*f0
	
	Vs = lambda t: gaussian_pulse(t, t0, sigma, V0) * np.sin(omega0 * t)
	
	reg = [
		LadderRegion(x0=0.0,     x1=L/4, L0_per_m=1e-6, C_per_m=130e-12, alpha=1e-3),
		LadderRegion(x0=L/4,     x1=3*L/4,   L0_per_m=1e-6, C_per_m=130e-12, alpha=10000),
		LadderRegion(x0=3*L/4,     x1=L,   L0_per_m=1e-6, C_per_m=130e-12, alpha=1e-3),
	]

	dt = 2.0e-12
	T  = 6.0e-9
	p = LadderParamsPW(N=N, L=L, Rs=Rs, RL=RL, dt=dt, T=T, Vs_func=Vs, regions=reg, nonlinear_update="implicit")
	out = NLTLadderPW(p).run()
	return out, p

fdtd_out, fdtd_p = run_fdtd_piecewise()
ladder_out, ladder_p = run_ladder_piecewise()

# Spatial axes
L = fdtd_p.L
x_fdtd = fdtd_out.x
x_lad  = np.linspace(0.0, L, ladder_p.N + 1)

# Time sync (use FDTD time as master)
t_common = fdtd_out.t
V_fdtd = fdtd_out.v_xt

# Interpolate ladder voltages onto t_common
t_lad = ladder_out.t
V_lad_raw = ladder_out.v_nodes
V_lad = np.empty((t_common.size, V_lad_raw.shape[1]), dtype=float)
for k in range(V_lad_raw.shape[1]):
	V_lad[:, k] = np.interp(t_common, t_lad, V_lad_raw[:, k])

# y-limits
vmin = min(V_fdtd.min(), V_lad.min())
vmax = max(V_fdtd.max(), V_lad.max())
margin = 0.05*(vmax - vmin + 1e-12)
ylims = (vmin - margin, vmax + margin)

# ------------------------- Extract region info -----------------------------

def region_boundaries_from_fdtd_params(pw_params: FDTDParamsPW):
	# Collect unique interior boundaries from regions, excluding 0 and L duplicates
	xs = sorted(set([r.x0 for r in pw_params.regions] + [pw_params.regions[-1].x1]))
	# interior boundaries (exclude 0 and L)
	return [x for x in xs if (x > 0.0 + 1e-15 and x < pw_params.L - 1e-15)]

def region_labels_fdtd(pw_params: FDTDParamsPW):
	labels = []
	for r in pw_params.regions:
		Z0 = np.sqrt(r.L0_per_m / r.C_per_m)
		labels.append((r.x0, r.x1, Z0, r.alpha))
	return labels

fdtd_bounds = region_boundaries_from_fdtd_params(fdtd_p)
fdtd_labels = region_labels_fdtd(fdtd_p)

# ------------------------- Plot Results -----------------------------------

plt.ioff()
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.28)
ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0])

ax_top.set_title("FDTD: V(x, t) — piecewise line")
ax_bot.set_title("Ladder: V(x, t) — piecewise line (nodes mapped to x)")

for ax in (ax_top, ax_bot):
	ax.set_xlim(0.0, L)
	ax.set_ylim(*ylims)
	ax.set_xlabel("Position x (m)")
	ax.set_ylabel("Voltage (V)")

# Plot initial lines
(line_top,) = ax_top.plot(x_fdtd, V_fdtd[0, :], linewidth=1.5)
(line_bot,) = ax_bot.plot(x_lad,  V_lad[0,  :], linewidth=1.5)

# Add vertical dashed lines at boundaries on both axes
boundary_lines = []
for xb in fdtd_bounds:
	lt = ax_top.axvline(xb, linestyle='--', linewidth=1.0)
	lb = ax_bot.axvline(xb, linestyle='--', linewidth=1.0)
	boundary_lines.extend([lt, lb])

# Compose compact annotation text (region interval, Z0, alpha)
# Example: [0.000–0.150 m] Z0≈50.0 Ω, α=2.0e-03
label_lines = []
for (x0, x1, Z0, alpha) in fdtd_labels:
	label_lines.append(f"[{x0:.3f}–{x1:.3f} m]  Z0≈{Z0:.1f} $\\Omega$,  $\\alpha$={alpha:.2e}")
label_text = "\n".join(label_lines)

# Put the annotation inside the top axes, upper-right corner
annot = ax_top.text(0.98, 0.05, label_text, transform=ax_top.transAxes,	ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round', alpha=0.2, lw=0.0) )

# Time stamp
time_txt = ax_top.text(0.02, 0.93, f"t = {t_common[0]:.3e} s", ha='left', va='center', transform=ax_top.transAxes)

def init():
	line_top.set_data(x_fdtd, V_fdtd[0, :])
	line_bot.set_data(x_lad,  V_lad[0,  :])
	time_txt.set_text(f"t = {t_common[0]:.3e} s")
	return (line_top, line_bot, time_txt, annot, *boundary_lines)

decimate = 5

def update(frame):
	v_top = V_fdtd[frame*decimate, :]
	v_bot = V_lad[frame*decimate,  :]
	line_top.set_data(x_fdtd, v_top)
	line_bot.set_data(x_lad,  v_bot)
	time_txt.set_text(f"t = {t_common[frame*decimate]:.3e} s")
	return (line_top, line_bot, time_txt, annot, *boundary_lines)

# Keep a strong reference to the animation at module scope
ani = FuncAnimation( fig, update, frames=len(t_common)//decimate, init_func=init, interval=30, blit=False, cache_frame_data=False)

# Optional save (GIF). Comment out if not needed.
try:
    out_path = "./data/nltl_compare_piecewise.gif"
    ani.save(out_path, writer=PillowWriter(fps=30))
    print(f"Saved animation: {out_path}")
except Exception as e:
    print("Animation save skipped or failed:", e)

if __name__ == "__main__":
	plt.show()
