
"""
run_nltl_demo.py (synchronized two-panel animation)
---------------------------------------------------
Top:  FDTD V(x, t)
Bottom: Ladder V(node->x, t)
Both panels advance in the SAME loop, with matched time stamps and common x-axis.

Requires: nltl_sim.py in the same directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from nltl_sim import demo_ladder, demo_fdtd

# -------------------------
# Run simulations (once)
# -------------------------
fdtd_out, fdtd_p = demo_fdtd()
ladder_out, ladder_p = demo_ladder()

# Common spatial axis: map ladder nodes to the same physical length as FDTD
L = fdtd_p.L
x_fdtd = fdtd_out.x                 # Nx+1
x_lad  = np.linspace(0.0, L, ladder_p.N + 1)

# Choose common time base = FDTD times; interpolate ladder to these times
t_common = fdtd_out.t               # Nt_f
V_fdtd = fdtd_out.v_xt              # (Nt_f, Nx+1)

# Interpolate ladder node voltages to t_common
t_lad = ladder_out.t
V_lad_raw = ladder_out.v_nodes      # (Nt_l, N+1)

# Precompute interpolation (loop over nodes)
V_lad = np.empty((t_common.size, V_lad_raw.shape[1]), dtype=float)
for k in range(V_lad_raw.shape[1]):
    V_lad[:, k] = np.interp(t_common, t_lad, V_lad_raw[:, k])

# Compute common y-limits for fair visual comparison
vmin = min(V_fdtd.min(), V_lad.min())
vmax = max(V_fdtd.max(), V_lad.max())
margin = 0.05*(vmax - vmin + 1e-12)
ylims = (vmin - margin, vmax + margin)

# -------------------------
# Build figure + animation
# -------------------------
plt.ioff()  # animation controls refresh
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0])

ax_top.set_title("FDTD: V(x, t)")
ax_bot.set_title("Ladder: V(x, t) (nodes mapped to x)")

for ax in (ax_top, ax_bot):
    ax.set_xlim(0.0, L)
    ax.set_ylim(*ylims)
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Voltage (V)")

# Initialize line objects
(line_top,) = ax_top.plot(x_fdtd, V_fdtd[0, :], linewidth=1.5)
(line_bot,) = ax_bot.plot(x_lad,  V_lad[0,  :], linewidth=1.5)
time_txt = fig.text(0.5, 0.95, f"t = {t_common[0]:.3e} s", ha='center', va='center')

def init():
    line_top.set_data(x_fdtd, V_fdtd[0, :])
    line_bot.set_data(x_lad,  V_lad[0,  :])
    time_txt.set_text(f"t = {t_common[0]:.3e} s")
    return (line_top, line_bot, time_txt)

def update(frame):
    v_top = V_fdtd[frame, :]
    v_bot = V_lad[frame,  :]
    line_top.set_data(x_fdtd, v_top)
    line_bot.set_data(x_lad,  v_bot)
    time_txt.set_text(f"t = {t_common[frame]:.3e} s")
    return (line_top, line_bot, time_txt)

ani = FuncAnimation(
    fig, update, frames=len(t_common), init_func=init, interval=30, blit=True
)

# Save a GIF for quick sharing (optional). Comment out to skip saving.
try:
    out_path = "/mnt/data/nltl_compare.gif"
    ani.save(out_path, writer=PillowWriter(fps=30))
    print(f"Saved animation: {out_path}")
except Exception as e:
    print("Animation save skipped or failed:", e)

# Show interactively if running as a script
if __name__ == "__main__":
    plt.show()
