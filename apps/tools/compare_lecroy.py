#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from colorama import Fore, Style
import time
import os
import sys
import argparse
import mplcursors

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('comparefile')
# parser.add_argument('--trimdc', help='Removes DC from FFT data', action='store_true')
parser.add_argument('--toffns', help='Add this horizontal offset (in ns) to the comparison file', type=float, default=0)
parser.add_argument('--voffmv', help='Add this vertical offset (in mV) to the comparison file', type=float, default=0)
parser.add_argument('--tmin', help='Set min time (ns)', type=float)
parser.add_argument('--tmax', help='Set max time (ns)', type=float)
parser.add_argument('--shortlegend', help='Abbreviate legend', action='store_true')
args = parser.parse_args()



# Read file
try:
	df = pd.read_csv(args.filename, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file {args.filename}. Aborting.")
	sys.exit()

# Read file
try:
	dfc = pd.read_csv(args.comparefile, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file {args.comparefile}. Aborting.")
	sys.exit()

# Create local variables
t = np.array(df['Time']*1e9)
v = np.array(df['Ampl']*1e3)
tc = np.array(dfc['Time']*1e9)
vc = np.array(dfc['Ampl']*1e3)

t_si = np.array(df['Time'])
v_si = np.array(df['Ampl'])
tc_si = np.array(dfc['Time'])
vc_si = np.array(dfc['Ampl'])

# Apply offsets
tc = tc + args.toffns
vc = vc + args.voffmv

# Create figure
fig1 = plt.figure(1, figsize=(8, 8))

gs = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs[0, 0])
ax1a.set_title(f"{os.path.basename(args.filename)} vs {os.path.basename(args.comparefile)}")

if args.shortlegend:
	label_0 = "orig"
	label_c = "comp"
else:
	label_0 = os.path.basename(args.filename)
	label_c = os.path.basename(args.comparefile)


ax1a.plot(t, v, linestyle='--', marker='.', color=(0, 0, 0.65), alpha=0.4, label=label_0)
ax1a.plot(tc, vc, linestyle=':', marker='.', color=(0.65, 0, 0), alpha=0.4, label=label_c)
ax1a.set_xlabel("Time (ns))")
ax1a.set_ylabel("Voltage (mV)")
ax1a.grid(True)

if args.tmin is not None:
	xl = ax1a.get_xlim()
	ax1a.set_xlim([args.tmin, xl[1]])
if args.tmax is not None:
	xl = ax1a.get_xlim()
	ax1a.set_xlim([xl[0], args.tmax])

mplcursors.cursor(multiple=True)

ax1a.legend()
fig1.tight_layout()

plt.show()