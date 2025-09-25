
# example_piecewise_run.py
import numpy as np
from nltl_sim import demo_fdtd_piecewise, demo_ladder_piecewise

fdtd_out, fdtd_p = demo_fdtd_piecewise()
lad_out, lad_p   = demo_ladder_piecewise()

print("FDTD x-grid length:", fdtd_out.x.size, "time steps:", fdtd_out.t.size)
print("Ladder nodes:", lad_out.v_nodes.shape[1], "time steps:", lad_out.t.size)
