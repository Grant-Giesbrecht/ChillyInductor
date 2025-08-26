import matplotlib as mpl
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from ganymede import locate_drive

parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--pub', help="Publication version.", action='store_true')
parser.add_argument('-s', '--save', help="Save figure to PDF.", action='store_true')
args = parser.parse_args()

m6_dir = locate_drive("M6 T7S")

FILE_T0 = os.path.join(m6_dir, "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-C July Campaign", "1-July-2025 TD Measurements", "C1RP23C_f105_AVG5k_00000.txt")
FILE_D0 = os.path.join(m6_dir, "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-C July Campaign", "1-July-2025 TD Measurements", "C1RP23C_f112_AVG5k_00000.txt")
OFFSET_D0 = 96.2
T_MIN0 = -460
T_MAX0 = -280

FILE_T1 = os.path.join(m6_dir, "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-C July Campaign", "1-July-2025 TD Measurements", "C1RP23C_f102_AVG5k_00000.txt")
FILE_D1 = os.path.join(m6_dir, "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-C July Campaign", "1-July-2025 TD Measurements", "C1RP23C_f107_AVG5k_00000.txt")
OFFSET_D1 = 97.7
T_MIN1 = -460
T_MAX1 = -300


# Read file
try:
	fn = FILE_T0
	df_t0 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_t0 = np.array(df_t0['Time']*1e9) - T_MIN0
	v_t0 = np.array(df_t0['Ampl']*1e3)
except:
	print(f"Failed to find file. Aborting.")
	sys.exit()

# Read file
try:
	fn = FILE_D0
	df_d0 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_d0 = np.array(df_d0['Time']*1e9) - T_MIN0
	v_d0 = np.array(df_d0['Ampl']*1e3)	
	
	t_d0 = t_d0 + OFFSET_D0
except:
	print(f"Failed to find file. Aborting.")
	sys.exit()


# Read file
try:
	fn = FILE_T1
	df_t1 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_t1 = np.array(df_t1['Time']*1e9) - T_MIN1
	v_t1 = np.array(df_t1['Ampl']*1e3)
except:
	print(f"Failed to find file. Aborting.")
	sys.exit()

# Read file
try:
	fn = FILE_D1
	df_d1 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_d1 = np.array(df_d1['Time']*1e9) - T_MIN1
	v_d1 = np.array(df_d1['Ampl']*1e3)	
	
	t_d1 = t_d1 + OFFSET_D1
except:
	print(f"Failed to find file. Aborting.")
	sys.exit()

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 16

color_trad = (0.3, 0.3, 0.3)
# color_trad = (0.45, 0.45, 0.45)
# color_trad = (0, 179/255, 146/255)
color_doub = (0, 119/255, 179/255) # From TQE template section header color



# Prepare figure 1
fig1 = plt.figure(1, figsize=(8, 5))
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

LW = 2
ALP=0.6
MKSZ= 2
# ax1a.plot(t_d0, v_d0, linestyle=':', marker='.', color=color_doub, label="Subharmonic Drive", markersize=7)
# ax1a.plot(t_t0, v_t0, linestyle='--', marker='.', color=color_trad, label="Direct Drive", markersize=10, markeredgewidth=2)
ax1a.plot(t_t0, v_t0, linestyle='-', marker='o', markersize=MKSZ, color=color_trad, label="Direct Drive (Bias=0 $\\mu A$), +17 dB", linewidth=LW, alpha=ALP)
ax1a.plot(t_d0, v_d0, linestyle='-', marker='o', markersize=MKSZ, color=color_doub, label="Subharmonic Drive (Bias=265 $\\mu A$), +16 dB", linewidth=LW, alpha=ALP)

ax1a.grid(True)
ax1a.set_xlabel("Time (ns)")
ax1a.set_ylabel(f"Voltage (mV)")
ax1a.legend(fontsize=12)
ax1a.set_xlim([T_MIN0-T_MIN0, T_MAX0-T_MIN0])

fig1.tight_layout()

if args.save:
	fig1.savefig(os.path.join(".", "figures", "AP5_fig1.pdf"))

# Prepare figure 1
fig2 = plt.figure(2, figsize=(8, 5))
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

LW = 2
ALP=0.6
MKSZ= 2
# ax1a.plot(t_d0, v_d0, linestyle=':', marker='.', color=color_doub, label="Subharmonic Drive", markersize=7)
# ax1a.plot(t_t0, v_t0, linestyle='--', marker='.', color=color_trad, label="Direct Drive", markersize=10, markeredgewidth=2)
ax2a.plot(t_t1, v_t1, linestyle='-', marker='o', markersize=MKSZ, color=color_trad, label="Direct Drive (Bias=0 $\\mu A$)", linewidth=LW, alpha=ALP)
ax2a.plot(t_d1, v_d1, linestyle='-', marker='o', markersize=MKSZ, color=color_doub, label="Subharmonic Drive (Bias=265 $\\mu A$)", linewidth=LW, alpha=ALP)

ax2a.grid(True)
ax2a.set_xlabel("Time (ns)")
ax2a.set_ylabel(f"Voltage (mV)")
ax2a.legend()
ax2a.set_xlim([T_MIN0-T_MIN0, T_MAX0-T_MIN0])

fig2.tight_layout()

if args.save:
	fig2.savefig(os.path.join(".", "figures", "AP5_fig2.pdf"))

plt.show()