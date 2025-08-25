import json
import matplotlib.pyplot as plt
import mplcursors

# 50 ns
file_trad = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF50ns\Trad\RB\SF50ns_trad_fit_I.json"
file_doub = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF50ns\doubler\RB\SF50ns_doub_fit_Q.json"
title_str = f"50ns Sigma Randomized Benchmark Results"
outfile = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF50ns\doubler\RB\AP4_fig1_50ns.pdf"

# 10 ns
file_trad = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF10ns\\trad\RB\SF10ns_trad_data_Q.json"
file_doub = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF10ns\doubler\RB\SF10ns_doubler_data_Q.json"
title_str = f"10ns Sigma Randomized Benchmark Results"
outfile = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF10ns\doubler\RB\AP4_fig1_10ns.pdf"

# Load JSON file
with open(file_trad, 'r') as f:
	trad_data = json.load(f)
# trad_x_pts = trad_data['I_points_x']
# trad_y_pts = trad_data['I_points_y']
# trad_x_fit = trad_data['I_fit_x']
# trad_y_fit = trad_data['I_fit_y']
trad_x_pts = trad_data['Q_points_x']
trad_y_pts = trad_data['Q_points_y']
trad_x_fit = trad_data['Q_fit_x']
trad_y_fit = trad_data['Q_fit_y']


# Load JSON file
with open(file_doub, 'r') as f:
	doub_data = json.load(f)
# doub_x_pts = doub_data['Q_points_X']
# doub_y_pts = doub_data['Q_points_Y']
# doub_x_fit = doub_data['Q_fit_X']
# doub_y_fit = doub_data['Q_fit_y']
doub_x_pts = doub_data['Q_points_x']
doub_y_pts = doub_data['Q_points_y']
doub_x_fit = doub_data['Q_fit_x']
doub_y_fit = doub_data['Q_fit_y']

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

ax1a.plot(trad_x_fit, trad_y_fit, linestyle=':', color='red', label="Traditional, fit")
ax1a.scatter(trad_x_pts, trad_y_pts, marker='o', color='red', label="Traditional")

ax1a.plot(doub_x_fit, doub_y_fit, linestyle=':', color='blue', label="Doubler, fit")
ax1a.scatter(doub_x_pts, doub_y_pts, marker='o', color='blue', label="Doubler")

ax1a.grid(True)
ax1a.legend()

ax1a.set_title(title_str)
ax1a.set_ylabel(f"Visibility")
ax1a.set_xlabel(f"Sequence Length")

mplcursors.cursor(multiple=True)

fig1.tight_layout()

plt.savefig(outfile)

plt.show()

