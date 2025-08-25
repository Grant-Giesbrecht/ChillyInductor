import matplotlib as mpl
import os
import json
import matplotlib.pyplot as plt
import mplcursors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pub', help="Publication version.", action='store_true')
parser.add_argument('-s', '--save', help="Save figure to PDF.", action='store_true')
args = parser.parse_args()

# 50 ns
# file_trad = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF50ns\Trad\RB\SF50ns_trad_fit_I.json"
# file_doub = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF50ns\doubler\RB\SF50ns_doub_fit_Q.json"
# title_str = f"50ns Sigma Randomized Benchmark Results"
# outfile = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF50ns\doubler\RB\AP4_fig1_50ns.pdf"

# # 10 ns
# file_trad = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF10ns\\trad\RB\SF10ns_trad_data_Q.json"
# file_doub = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF10ns\doubler\RB\SF10ns_doubler_data_Q.json"
# title_str = f"10ns Sigma Randomized Benchmark Results"
# outfile = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF10ns\doubler\RB\AP4_fig1_10ns.pdf"

# 10 ns - MacOS
base_path = os.path.join("/", "Volumes", "M6 T7S", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Misc", "Mai_Exodus", "grant_data_mai", "SF10ns")
file_trad = os.path.join(base_path, "trad", "RB", "SF10ns_trad_data_Q.json")
file_doub = os.path.join(base_path, "doubler", "RB", "SF10ns_doubler_data_Q.json")
title_str = f""
outfile = os.path.join(base_path, "doubler", "RB", "AP4_fig1_10ns.pdf")

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

if args.pub:
	plt.rcParams['font.family'] = 'sans-serif'
	plt.rcParams['font.sans-serif'] = ['Arial']
	plt.rcParams['font.size'] = 16

color_trad = (0.3, 0.3, 0.3)
color_doub = (0, 119/255, 179/255) # From TQE template section header color

fig1 = plt.figure(1, figsize=(8, 5))
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

if args.pub:
	
	ax1a.plot(trad_x_fit, trad_y_fit, linestyle='--', color=color_trad, label="Direct Drive, fit")
	ax1a.plot(trad_x_pts, trad_y_pts, marker='+', color=color_trad, label="Direct Drive", markersize=10, markeredgewidth=2, linewidth=0)

	ax1a.plot(doub_x_fit, doub_y_fit, linestyle=':', color=color_doub, label="Subharmonic Drive, fit")
	ax1a.plot(doub_x_pts, doub_y_pts, marker='o', color=color_doub, label="Subharmonic Drive", markersize=7, linewidth=0)

	ax1a.grid(True)
	ax1a.legend()

	ax1a.set_ylabel(f"Visibility")
	ax1a.set_xlabel(f"Sequence Length")

	mplcursors.cursor(multiple=True)

	fig1.tight_layout()
else:
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

if args.save:
	plt.savefig(outfile)

plt.show()

