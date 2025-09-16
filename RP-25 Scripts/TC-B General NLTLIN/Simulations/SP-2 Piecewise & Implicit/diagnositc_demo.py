from nltl_sim import demo_fdtd_piecewise
from nltl_analysis import probe_fdtd_voltage, plot_signal_diagnostics

# Run simulation and probe near the load
fdtd_out, _ = demo_fdtd_piecewise()
t, v_t = probe_fdtd_voltage(fdtd_out, x=fdtd_out.x[-1])

# Plot diagnostics (assume ~8 GHz carrier)
fig, axes = plot_signal_diagnostics(t, v_t, dt=t[1]-t[0], center_freq_hz=8e9, smooth_points=11, fmax=20e9)
fig.show()