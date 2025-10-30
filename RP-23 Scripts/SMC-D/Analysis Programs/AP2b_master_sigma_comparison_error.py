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

ylim_common = [0, 0.02]
xlim_common = [0, 60]
CSIZE = 6
GUIDE_COLOR = (0.5, 0.5, 0.5)
GUIDE_ALPHA = 0.7
COLOR_TRAD = (0.75, 0, 0)
COLOR_DOUB = (0, 0, 0.75)
COLOR_TRIPLE = (0, 0.75, 0)
ERROR_ALPHA = 0.7

fig1 = plt.figure(1, figsize=(15, 5))
gs1 = fig1.add_gridspec(1, 3)
ax1a = fig1.add_subplot(gs1[0, 0])
ax1b = fig1.add_subplot(gs1[0, 1])
ax1c = fig1.add_subplot(gs1[0, 2])

fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

# ax1a.plot(sigmas, err_uncs, linestyle=':', marker='+', color=(0.75, 0, 0))
ax1a.plot(expected_err_x_us*1e3, expected_err_y_err, color=GUIDE_COLOR, alpha=GUIDE_ALPHA, label="Expected error")
ax1a.errorbar(trad_sigmas, trad_err_uncs, yerr=trad_std_uncs, linestyle=':', marker='o', color=COLOR_TRAD, label="Traditional", capsize=CSIZE)
ax1a.grid(True)
ax1a.set_xlabel("Sigma (ns)")
ax1a.set_ylabel("Error per gate")
ax1a.set_title("Traditional")
ax1a.set_ylim(ylim_common)
ax1a.set_xlim(xlim_common)
# ax1a.legend()

ax1b.plot(expected_err_x_us*1e3, expected_err_y_err, color=GUIDE_COLOR, alpha=GUIDE_ALPHA, label="Expected error")
ax1b.errorbar(doubler_sigmas, doubler_err_uncs, yerr=doubler_std_uncs, linestyle=':', marker='o', color=COLOR_DOUB, label="Doubler", capsize=CSIZE)
ax1b.grid(True)
ax1b.set_xlabel("Sigma (ns)")
ax1b.set_ylabel("Error per gate")
ax1b.set_title("Doubler")
ax1b.set_ylim(ylim_common)
ax1b.set_xlim(xlim_common)

ax1c.plot(expected_err_x_us*1e3, expected_err_y_err, color=GUIDE_COLOR, alpha=GUIDE_ALPHA, label="Expected error")
ax1c.errorbar(tri_sigmas, tri_err_uncs, yerr=tri_std_uncs, linestyle=':', marker='o', color=COLOR_TRIPLE, label="Tripler", capsize=CSIZE)
ax1c.grid(True)
ax1c.set_xlabel("Sigma (ns)")
ax1c.set_ylabel("Error per gate")
ax1c.set_title("Tripler")
ax1c.set_ylim(ylim_common)
ax1c.set_xlim(xlim_common)

ax2a.plot(expected_err_x_us*1e3, expected_err_y_err, color=GUIDE_COLOR, alpha=GUIDE_ALPHA, label="Expected error")
ax2a.errorbar(trad_sigmas, trad_err_uncs, yerr=trad_std_uncs, linestyle=':', marker='o', color=COLOR_TRAD, label="Traditional", capsize=CSIZE, alpha=ERROR_ALPHA)
ax2a.errorbar(doubler_sigmas[1:], doubler_err_uncs[1:], yerr=doubler_std_uncs[1:], linestyle=':', marker='o', color=COLOR_DOUB, label="Doubler", capsize=CSIZE, alpha=ERROR_ALPHA)
ax2a.errorbar(tri_sigmas, tri_err_uncs, yerr=tri_std_uncs, linestyle=':', marker='o', color=COLOR_TRIPLE, label="Tripler", capsize=CSIZE, alpha=ERROR_ALPHA)
ax2a.grid(True)
ax2a.set_xlabel("Sigma (ns)")
ax2a.set_ylabel("Error per gate")
ax2a.set_title("Error Comparison")
ax2a.set_ylim(ylim_common)
ax2a.set_xlim(xlim_common)
ax2a.legend()



fig1.tight_layout()
fig2.tight_layout()

plt.show()