from AP2UNIV_data_source import *
import matplotlib.pyplot as plt

#========== Get expected error ===========

expected_err_x_us = np.linspace(0, 0.2, 101)
T1_us = 36
T2_us = 20
tau = 1.875*expected_err_x_us*8
expected_err_y_F = 1/6*(3+np.exp(-tau/T1_us)+2*np.exp(-tau/T2_us))
expected_err_y_err = 1-expected_err_y_F

#========= Plot results ===========

ylim_common = [0, 0.006]
xlim_common = [0, 60]
CSIZE = 6
GUIDE_COLOR = (0.5, 0.5, 0.5)
GUIDE_ALPHA = 0.7
COLOR_TRAD = (0.75, 0, 0)
COLOR_DOUB = (0, 0, 0.75)
COLOR_TRIPLE = (0, 0.75, 0)
ERROR_ALPHA = 0.7

fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

ax2a.plot(trad_sigmas, trad_std_uncs, linestyle=':', marker='o', color=COLOR_TRAD, label="Traditional")
ax2a.plot(doubler_sigmas, doubler_std_uncs, linestyle=':', marker='o', color=COLOR_DOUB, label="Doubler")
ax2a.plot(tri_sigmas, tri_std_uncs, linestyle=':', marker='o', color=COLOR_TRIPLE, label="Tripler")

ax2a.grid(True)
ax2a.set_xlabel("Sigma (ns)")
ax2a.set_ylabel("Standard deviation(Error per gate)")
ax2a.set_title("Error Comparison")
ax2a.set_ylim(ylim_common)
ax2a.set_xlim(xlim_common)
ax2a.legend()


fig2.tight_layout()

plt.show()