import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class datapoint:
	sigma: float #ns
	err:float 
	err_unc:float
	std_unc:float
	notes:str = ""

# List of datapoints
traditional_points = []
doubler_points = []
tripler_points = []

#=========== Add doubler points ==============
doubler_points.append(datapoint(sigma=5, err=5.36e-3, err_unc=8.64e-3, std_unc=1.18e-2))
doubler_points.append(datapoint(sigma=10, err=3.614e-3, err_unc=3.47e-3, std_unc=1.56e-3))
doubler_points.append(datapoint(sigma=15, err=4.387e-3, err_unc=4.38e-3, std_unc=1.23e-3))
doubler_points.append(datapoint(sigma=20, err=5.46e-3, err_unc=5.5e-3, std_unc=1.41e-3))
doubler_points.append(datapoint(sigma=25, err=6.987e-3, err_unc=7.03e-3, std_unc=1.53e-3 ))
doubler_points.append(datapoint(sigma=30 , err=8.391e-3 , err_unc=8.66e-3 , std_unc=2.26e-3 ))
doubler_points.append(datapoint(sigma=35, err=9.598e-3 , err_unc=9.82e-3 , std_unc=2.227e-3 ))
doubler_points.append(datapoint(sigma=40 , err=1.09e-2 , err_unc=1.13e-2 , std_unc=2.41e-3 ))
doubler_points.append(datapoint(sigma=50 , err=1.405e-2 , err_unc=1.44e-2 , std_unc=3.46e-3 ))

traditional_points.append(datapoint(sigma=5 , err=2.412e-2 , err_unc=2.33e-3 , std_unc=1.5e-3 ))
traditional_points.append(datapoint(sigma=10, err=3.08e-3 , err_unc=3.02e-3 , std_unc=9.01e-4 ))
traditional_points.append(datapoint(sigma=15, err=3.791e-3 , err_unc=3.793e-3 , std_unc=8.99e-4 ))
traditional_points.append(datapoint(sigma=20 , err=5.0167e-3 , err_unc=5.03e-3 , std_unc=9.53e-4 ))
traditional_points.append(datapoint(sigma=25 , err=6.63e-3 , err_unc=6.8e-3 , std_unc=1.7e-3 ))
traditional_points.append(datapoint(sigma=30 , err=7.785e-3 , err_unc=7.88e-3 , std_unc=1.46e-3 ))
traditional_points.append(datapoint(sigma=35 , err=1.0078e-2 , err_unc=1.01e-2 , std_unc=1.87e-3 ))
traditional_points.append(datapoint(sigma=40 , err=1.09e-2 , err_unc=1.11e-2 , std_unc=2.3e-3 ))
traditional_points.append(datapoint(sigma=50 , err=1.379e-2 , err_unc=1.42e-2 , std_unc=3.71e-3 ))

# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))

# tripler_points.append(datapoint(sigma=10 , err=2.41e-2 , err_unc=7.59e-2 , std_unc=0.125 ))
tripler_points.append(datapoint(sigma=15 , err=7.3153e-3 , err_unc=7.61e-3 , std_unc=3.12e-3 ))
tripler_points.append(datapoint(sigma=20 , err=7.63e-3 , err_unc=7.65e-3 , std_unc=2.61e-3 ))
tripler_points.append(datapoint(sigma=25 , err=8.822e-3 , err_unc=9.24e-3 , std_unc=3.26e-3 ))
tripler_points.append(datapoint(sigma=30 , err=1.0268e-2 , err_unc=1.06e-2 , std_unc=3.62e-3 ))
tripler_points.append(datapoint(sigma=35 , err=1.255e-2 , err_unc=1.29e-2 , std_unc=4.97e-3 ))
tripler_points.append(datapoint(sigma=40 , err=1.257e-2 , err_unc=1.3e-2 , std_unc=3.93e-3 ))
tripler_points.append(datapoint(sigma=50 , err=1.492e-2 , err_unc=1.56e-2 , std_unc=4.79e-3 ))
# tripler_points.append(datapoint(sigma=75 , err= , err_unc= , std_unc= ))

# tripler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# tripler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# tripler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))

# datapoint(sigma=20, )

trad_sigmas = []
trad_errs = []
trad_err_uncs = []
trad_std_uncs = []
for pt in traditional_points:
	trad_sigmas.append(pt.sigma)
	trad_errs.append(pt.err)
	trad_err_uncs.append(pt.err_unc)
	trad_std_uncs.append(pt.std_unc)

doubler_sigmas = []
doubler_errs = []
doubler_err_uncs = []
doubler_std_uncs = []
for pt in doubler_points:
	doubler_sigmas.append(pt.sigma)
	doubler_errs.append(pt.err)
	doubler_err_uncs.append(pt.err_unc)
	doubler_std_uncs.append(pt.std_unc)

tri_sigmas = []
tri_errs = []
tri_err_uncs = []
tri_std_uncs = []
for pt in tripler_points:
	tri_sigmas.append(pt.sigma)
	tri_errs.append(pt.err)
	tri_err_uncs.append(pt.err_unc)
	tri_std_uncs.append(pt.std_unc)

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