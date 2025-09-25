import matplotlib as mpl
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import mplcursors
from ganymede import locate_drive

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pub', help="Publication version.", action='store_true')
parser.add_argument('-s', '--save', help="Save figure to PDF.", action='store_true')
args = parser.parse_args()

m6_dir = locate_drive("M6 T7S")

FILE_T0 = os.path.join(m6_dir, "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-C July Campaign", "1-July-2025 TD Measurements", "C1RP23C_f105_AVG5k_00000.txt")
FILE_D0 = os.path.join(m6_dir, "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-C July Campaign", "1-July-2025 TD Measurements", "C1RP23C_f112_AVG5k_00000.txt")
OFFSET_D0 = 97.44+0.78-3.16
T_MIN0A = -460
T_MAX0A = -380
T_MIN0B = -380
T_MAX0B = -280


FILE_T1 = os.path.join(m6_dir, "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-C July Campaign", "1-July-2025 TD Measurements", "C1RP23C_f102_AVG5k_00000.txt")
FILE_D1 = os.path.join(m6_dir, "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-C July Campaign", "1-July-2025 TD Measurements", "C1RP23C_f107_AVG5k_00000.txt")
OFFSET_D1 = 97.7
T_MIN1A = -460
T_MAX1A = -380

FIGSIZE = (5, 5)
# FIGSIZE = (10, 7)

#========================== READ ALL FIELS =======================

# Read file
try:
	fn = FILE_T0
	trad_df_ovdr = pd.read_csv(fn, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

# Read file
try:
	fn = FILE_D0
	doub_df_ovdr = pd.read_csv(fn, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()


# Read file
try:
	fn = FILE_T1
	trad_df_nom = pd.read_csv(fn, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file. Aborting.")
	sys.exit()

# Read file
try:
	fn = FILE_D1
	doub_df_nom = pd.read_csv(fn, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file. Aborting.")
	sys.exit()

if args.pub:
	mpl.rcParams['font.family'] = 'sans-serif'
	mpl.rcParams['font.sans-serif'] = ['Arial']
	mpl.rcParams['font.size'] = 16

color_trad = (0.3, 0.3, 0.3)
# color_trad = (0.45, 0.45, 0.45)
# color_trad = (0, 179/255, 146/255)
color_doub = (0, 119/255, 179/255) # From TQE template section header color

labels = []
labels.append("Direct Drive (Bias=0 $\\mu A$), +17 dB")
labels.append("Subharmonic Drive (Bias=265 $\\mu A$), +16 dB")

labels_nom = []
labels_nom.append("Direct Drive (Bias=0 $\\mu A$)")
labels_nom.append("Subharmonic Drive (Bias=265 $\\mu A$)")
#========================== FIG1: OVERDRIVE, PI =======================

def plot_sub_wave(trad_df, doub_df, doub_t_shift, t_scale_shift, fig_num, label_list, x_limits, y_limits):
	
	t_trad = np.array(trad_df['Time']*1e9) - t_scale_shift
	v_trad = np.array(trad_df['Ampl']*1e3)
	
	t_doub = np.array(doub_df['Time']*1e9) - t_scale_shift + doub_t_shift
	v_doub = np.array(doub_df['Ampl']*1e3)

	# Prepare figure 1
	fig1 = plt.figure(fig_num, figsize=FIGSIZE)
	gs1 = fig1.add_gridspec(1, 1)
	ax1a = fig1.add_subplot(gs1[0, 0])

	LW = 2
	ALP=0.6
	MKSZ= 2
	# ax1a.plot(t_d0, v_d0, linestyle=':', marker='.', color=color_doub, label="Subharmonic Drive", markersize=7)
	# ax1a.plot(t_t0, v_t0, linestyle='--', marker='.', color=color_trad, label="Direct Drive", markersize=10, markeredgewidth=2)
	ax1a.plot(t_trad, v_trad, linestyle='-', marker='o', markersize=MKSZ, color=color_trad, label=label_list[0], linewidth=LW, alpha=ALP)
	ax1a.plot(t_doub, v_doub, linestyle='-', marker='o', markersize=MKSZ, color=color_doub, label=label_list[1], linewidth=LW, alpha=ALP)

	ax1a.grid(True)
	ax1a.set_xlabel("Time (ns)")
	ax1a.set_ylabel(f"Voltage (mV)")
	ax1a.legend(fontsize=12)
	ax1a.set_xlim(x_limits)
	if y_limits is not None:
		ax1a.set_ylim(y_limits)

	fig1.tight_layout()
	
	return fig1

fig1 = plot_sub_wave(trad_df_ovdr, doub_df_ovdr, doub_t_shift=OFFSET_D0, t_scale_shift=-460+5, fig_num=1, label_list=labels, x_limits=(0, 70), y_limits=(-150, 150) )
fig2 = plot_sub_wave(trad_df_ovdr, doub_df_ovdr, doub_t_shift=OFFSET_D0+1.67, t_scale_shift=-460+5, fig_num=2, label_list=labels, x_limits=(97.5, 157.5), y_limits=(-150, 150) )

fig3 = plot_sub_wave(trad_df_nom, doub_df_nom, doub_t_shift=OFFSET_D1, t_scale_shift=-445, fig_num=3, label_list=labels_nom, x_limits=(0, 55), y_limits=(-25, 25) )
fig4 = plot_sub_wave(trad_df_nom, doub_df_nom, doub_t_shift=OFFSET_D1+1.61, t_scale_shift=-445, fig_num=4, label_list=labels_nom, x_limits=(90, 140), y_limits=(-25, 25) )


if args.save:
	fig1.savefig(os.path.join(".", "figures", "AP5a_ovdrPi_fig1.pdf"))
	fig2.savefig(os.path.join(".", "figures", "AP5a_ovdrPi2_fig2.pdf"))
	fig3.savefig(os.path.join(".", "figures", "AP5a_nomPi_fig3.pdf"))
	fig4.savefig(os.path.join(".", "figures", "AP5a_nomPi2_fig4.pdf"))

mplcursors.cursor(multiple=True)
plt.show()
