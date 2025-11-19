import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
from stardust.algorithm import linstep

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pub', help="Publication version.", action='store_true')
parser.add_argument('-s', '--save', help="Save figure to PDF.", action='store_true')
args = parser.parse_args()

from AP2UNIV_data_source import *

if args.pub:
	
	fsz = 20
	mpl.rcParams['font.family'] = 'sans-serif'
	mpl.rcParams['font.size'] = fsz
	mpl.rcParams['font.sans-serif'] = ['Arial']

	plt.rc('font', size=fsz)          # Controls default text sizes
	plt.rc('axes', titlesize=fsz)     # Fontsize of the axes title
	plt.rc('axes', labelsize=fsz)     # Fontsize of the x and y labels
	plt.rc('xtick', labelsize=fsz)    # Fontsize of the x-axis tick labels
	plt.rc('ytick', labelsize=fsz)    # Fontsize of the y-axis tick labels
	plt.rc('legend', fontsize=fsz)    # Fontsize of the legend
	plt.rc('figure', titlesize=fsz)   # Fontsize of the figure title

#========== Get expected error ===========

expected_err_x_us = np.linspace(0, 0.06, 101)
# T1_us = 36
# T2_us = 20

T1_us = 33
T2_us = 40

T1_us = 24
T2_us = 39

T1_us_list = [36.7, 23.45, 25.05, 29.45]
T2_us_list = [21, 29.21, 27.85, 30.41, 26.75, 22.8]

tau = 1.875*expected_err_x_us*8
expected_err_y_F = 1/6*(3+np.exp(-tau/T1_us)+2*np.exp(-tau/T2_us))
expected_err_y_err = 1-expected_err_y_F

#========= Plot results ===========

ylim_common = [0, 0.017]
ylim_common = [0, 17]
xlim_common = [0, 60]

CSIZE = 6
GUIDE_COLOR = (0.7, 0, 0)
GUIDE_ALPHA = 0.5

COLOR_TRAD = (0.3, 0.3, 0.3)
COLOR_DOUB = (0, 119/255, 179/255) # From TQE template section header color
COLOR_TRIPLE = (0, 128/255, 35/255) # From RB schema figure
ERROR_ALPHA = 1

marker_trad = ('s', 8)
marker_doub = ('o', 8)
marker_tri = ('v', 8)

# # fig1 = plt.figure(1, figsize=(15, 5))
# fig1 = plt.figure(1, figsize=(10, 3))
# gs1 = fig1.add_gridspec(1, 3)
# ax1a = fig1.add_subplot(gs1[0, 0])
# ax1b = fig1.add_subplot(gs1[0, 1])
# ax1c = fig1.add_subplot(gs1[0, 2])

# fig2 = plt.figure(2)
fig2 = plt.figure(2, figsize=(10, 8))
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

# # ax1a.plot(sigmas, err_uncs, linestyle=':', marker='+', color=(0.75, 0, 0))
# ax1a.plot(expected_err_x_us*1e3, expected_err_y_err*1e3, color=GUIDE_COLOR, alpha=GUIDE_ALPHA, label="Expected error")
# ax1a.errorbar(trad_sigmas, np.array(trad_err_uncs)*1e3, yerr=np.array(trad_std_uncs)*1e3, linestyle='--', marker=marker_trad[0], color=COLOR_TRAD, label="Traditional", capsize=CSIZE, markersize=marker_trad[1])
# ax1a.grid(True)
# ax1a.set_xlabel("Sigma (ns)")
# ax1a.set_ylabel("Error per gate (x1e3)")
# ax1a.set_title("Traditional")
# ax1a.set_ylim(ylim_common)
# ax1a.set_xlim(xlim_common)
# # ax1a.legend()
# 
# ax1b.plot(expected_err_x_us*1e3, np.array(expected_err_y_err)*1e3, color=GUIDE_COLOR, alpha=GUIDE_ALPHA, label="Expected error")
# ax1b.errorbar(doubler_sigmas[1:], np.array(doubler_err_uncs[1:])*1e3, yerr=np.array(doubler_std_uncs[1:])*1e3, linestyle='--', marker=marker_doub[0], color=COLOR_DOUB, label="Doubler", capsize=CSIZE, markersize=marker_doub[1])
# ax1b.grid(True)
# ax1b.set_xlabel("Sigma (ns)")
# ax1b.set_ylabel("Error per gate (x1e3)")
# ax1b.set_title("Doubler")
# ax1b.set_ylim(ylim_common)
# ax1b.set_xlim(xlim_common)
# 
# ax1c.plot(expected_err_x_us*1e3, np.array(expected_err_y_err)*1e3, color=GUIDE_COLOR, alpha=GUIDE_ALPHA, label="Expected error")
# ax1c.errorbar(tri_sigmas, np.array(tri_err_uncs)*1e3, yerr=np.array(tri_std_uncs)*1e3, linestyle='--', marker=marker_tri[0], color=COLOR_TRIPLE, label="Tripler", capsize=CSIZE, markersize=marker_tri[1])
# ax1c.grid(True)
# ax1c.set_xlabel("Sigma (ns)")
# ax1c.set_ylabel("Error per gate (x1e3)")
# ax1c.set_title("Tripler")
# ax1c.set_ylim(ylim_common)
# ax1c.set_xlim(xlim_common)

for t1_ in T1_us_list:
	for t2_ in T2_us_list:
		
		tau = 1.875*expected_err_x_us*8
		expected_err_y_F = 1/6*(3+np.exp(-tau/t1_)+2*np.exp(-tau/t2_))
		expected_err_y_err = 1-expected_err_y_F
		
		ax2a.plot(expected_err_x_us*1e3, np.array(expected_err_y_err)*1e3, color=GUIDE_COLOR, alpha=GUIDE_ALPHA, label=f"Expected error, T1={t1_}, T2={t2_}")

T1_us_list_mcb = linstep(30, 35, 1)
T2_us_list_mcb = linstep(39, 44, 1)

for t1_ in T1_us_list_mcb:
	for t2_ in T2_us_list_mcb:
		
		tau = 1.875*expected_err_x_us*8
		expected_err_y_F = 1/6*(3+np.exp(-tau/t1_)+2*np.exp(-tau/t2_))
		expected_err_y_err = 1-expected_err_y_F
		
		ax2a.plot(expected_err_x_us*1e3, np.array(expected_err_y_err)*1e3, color=(0.5, 0, 0), alpha=GUIDE_ALPHA, label=f"Expected error, T1={t1_}, T2={t2_}")

ax2a.errorbar(trad_sigmas, np.array(trad_err_uncs)*1e3, yerr=np.array(trad_std_uncs)*1e3, linestyle='--', marker=marker_trad[0], color=COLOR_TRAD, label="Traditional", capsize=CSIZE, alpha=ERROR_ALPHA, markersize=marker_trad[1])
ax2a.errorbar(doubler_sigmas[1:], np.array(doubler_err_uncs[1:])*1e3, yerr=np.array(doubler_std_uncs[1:])*1e3, linestyle='--', marker=marker_doub[0], color=COLOR_DOUB, label="Doubler", capsize=CSIZE, alpha=ERROR_ALPHA, markersize=marker_doub[1])
ax2a.errorbar(tri_sigmas, np.array(tri_err_uncs)*1e3, yerr=np.array(tri_std_uncs)*1e3, linestyle='--', marker=marker_tri[0], color=COLOR_TRIPLE, label="Tripler", capsize=CSIZE, alpha=ERROR_ALPHA, markersize=marker_tri[1])

ax2a.grid(True)
ax2a.set_xlabel("Sigma (ns)")
ax2a.set_ylabel("Error per gate (x1e3)")
# ax2a.set_title("Error Comparison")
ax2a.set_ylim(ylim_common)
ax2a.set_xlim(xlim_common)
# ax2a.legend()

# fig1.tight_layout()
fig2.tight_layout()

if args.save:
	# fig1.savefig(os.path.join("figures", "AP2bPub_v2_fig1.pdf"))
	fig2.savefig(os.path.join("figures", "AP2bPub_v3_fig2.pdf"))
	
	# fig1.savefig(os.path.join("figures", "AP2bPub_v2_fig1.eps"))
	fig2.savefig(os.path.join("figures", "AP2bPub_v3_fig2.eps"))

plt.show()