import matplotlib as mpl
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from ganymede import extract_visible_xy
import json
import mplcursors

subharm_sigma_presquare = np.array([10, 12, 15, 20, 25, 30, 35.35, 40, 70.7, 141, 282])
subharm_sigma = subharm_sigma_presquare/np.sqrt(2)
subharm_err = np.array([8.19E-03, 5.62e-3, 3.32E-03, 4.64E-03, 5.02E-03, 5.64E-03, 6.72e-3 , 8.04E-03, 1.37e-2, 2.78e-2, 5.1e-2])


subharm_max_sigma = [5, 10, 15, 20, 25, 30, 35, 40, 50, 75]
subharm_max_err = [5.3573e-3, 3.614e-3, 4.387e-3, 5.46e-3, 6.987e-3, 8.391e-3, 9.5975e-3, 1.097e-2, 1.405e-2, 2.8396e-2]

trad_max_sigma = [5, 10, 15, 20, 25, 35, 40, 50, 100, 200]
trad_max_err = [2.412e-3, 3.08e-3, 3.7913e-3, 5.0167e-3, 6.637e-3, 1.01e-2, 1.09e-2, 1.38e-2, 2.497e-2, 4.169e-2]

tri_max_sigma = [15, 20, 25, 30, 35, 40, 50, 75]
tri_max_err = [7.32e-3, 7.63e-3, 8.822e-3, 1.0268e-2, 1.1255e-2, 1.256e-2, 1.492e-2, 3.725e-2]
# tri_max_err = [8.822e-3, 1.0268e-2, 1.1255e-2, 1.256e-2, 1.492e-2, 2.48e-2] # Here i'm using the lower error 75ns point which has lower SNR but seems more reasonable. This is just a test, dont publish this one!

color_trad = (0.3, 0.3, 0.3)
# color_trad = (0.45, 0.45, 0.45)
# color_trad = (0, 179/255, 146/255)
color_doub = (0, 119/255, 179/255) # From TQE template section header color
color_direct = (179/255, 119/255, 0) # From TQE template section header color
color_subh = (119/255, 179/255, 0) # From TQE template section header color
color_subhmax = (0, 179/255, 0) # From TQE template section header color
color_tradhmax = (0.7, 0, 0) # From TQE template section header color
color_trihmax = (0, 0, 0.7) # From TQE template section header color


# Prepare figure 1
fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])


ax1a.plot(subharm_max_sigma, subharm_max_err, linestyle=':', marker='o', color=color_subhmax, label="Subharmonic Drive, Trace-2, Long Sigma", markersize=10, markeredgewidth=2)
ax1a.plot(trad_max_sigma, trad_max_err, linestyle=':', marker='+', color=color_tradhmax, label="Traditional Drive, Trace-2, Long Sigma", markersize=10, markeredgewidth=2)

ax1a.plot(tri_max_sigma, tri_max_err, linestyle=':', marker='x', color=color_trihmax, label="Third harmonic Drive, Trace-2, Long Sigma", markersize=10, markeredgewidth=2)


ax1a.grid(True)
ax1a.set_xlabel("Pulse Width Parameter ($\\sigma$) (ns)")

ax1a.set_ylabel(f"Error per Gate")
ax1a.legend()


mplcursors.cursor(multiple=True)


standard_sigmas = np.array([10, 15, 20, 25, 30, 40])

mod_sigmas = standard_sigmas * 2 * 2 *np.sqrt(2)
print(f"Standard sigmas -> 2*2*root(2)")
print(mod_sigmas)
print(mod_sigmas*np.sqrt(2)/8)

plt.show()