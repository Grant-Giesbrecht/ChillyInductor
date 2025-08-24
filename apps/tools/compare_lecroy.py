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
parser.add_argument('--file2', help='Add a second comparison file')
parser.add_argument('--file3', help='Add a third comparison file')
parser.add_argument('--file4', help='Add a fourth comparison file')
parser.add_argument('--toffns', help='Add this horizontal offset (in ns) to the comparison file', type=float, default=0)
parser.add_argument('--toffns2', help='Add this horizontal offset (in ns) to comparison file 2', type=float, default=0)
parser.add_argument('--toffns3', help='Add this horizontal offset (in ns) to comparison file 3', type=float, default=0)
parser.add_argument('--toffns4', help='Add this horizontal offset (in ns) to comparison file 4', type=float, default=0)
parser.add_argument('--voffmv', help='Add this vertical offset (in mV) to the comparison file', type=float, default=0)
parser.add_argument('--voffmv2', help='Add this vertical offset (in mV) to comparison file 2', type=float, default=0)
parser.add_argument('--voffmv3', help='Add this vertical offset (in mV) to comparison file 3', type=float, default=0)
parser.add_argument('--voffmv4', help='Add this vertical offset (in mV) to comparison file 4', type=float, default=0)
parser.add_argument('--legend0', help='Label for original file')
parser.add_argument('--legend1', help='Label for comparison file 1')
parser.add_argument('--legend2', help='Label for comparison file 2')
parser.add_argument('--legend3', help='Label for comparison file 3')
parser.add_argument('--legend4', help='Label for comparison file 4')
parser.add_argument('--scale', help='Scale factor for comparison file.', type=float, default=1)

parser.add_argument('--tmin', help='Set min time (ns)', type=float)
parser.add_argument('--tmax', help='Set max time (ns)', type=float)
parser.add_argument('--shortlegend', help='Abbreviate legend', action='store_true')
parser.add_argument('--flip', help='Flip plot order', action='store_true')
parser.add_argument('--path', help='Specify path to append to all provided filenames, both for reading and saving.')
parser.add_argument('--savepdf', help='Save the figure as a PDF to the provided filename.')
parser.add_argument('--title', help='Specify graph title')
parser.add_argument('--figsizex', help='Specify figure x-size. Default=8.', type=float, default=8)
parser.add_argument('--figsizey', help='Specify figure y-size. Default=8.', type=float, default=8)
args = parser.parse_args()



# Read file
try:
	fn = args.filename
	if args.path is not None:
		fn = os.path.join(args.path, fn)
	df = pd.read_csv(fn, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file {args.filename}. Aborting.")
	sys.exit()

# Read file
try:
	fn = args.comparefile
	if args.path is not None:
		fn = os.path.join(args.path, fn)
	dfc = pd.read_csv(fn, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file {args.comparefile}. Aborting.")
	sys.exit()

# Read file
if args.file2 is not None:
	try:
		fn = args.file2
		if args.path is not None:
			fn = os.path.join(args.path, fn)
		df2 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
		
		t2 = np.array(df2['Time']*1e9)
		v2 = np.array(df2['Ampl']*1e3)
		
		t2 = t2 + args.toffns2
		v2 = v2 + args.voffmv2
	except:
		print(f"Failed to find file {args.file2}. Skipping file.")
		df2 = None
	label_2 = os.path.basename(args.file2)
else:
	df2 = None

# Read file
if args.file3 is not None:
	try:
		fn = args.file3
		if args.path is not None:
			fn = os.path.join(args.path, fn)
		df3 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
		
		t3 = np.array(df3['Time']*1e9)
		v3 = np.array(df3['Ampl']*1e3)
		
		t3 = t3 + args.toffns3
		v3 = v3 + args.voffmv3
	except:
		print(f"Failed to find file {args.file3}. Skipping file.")
		df3 = None
	label_3 = os.path.basename(args.file3)
else:
	df3 = None

# Read file
if args.file4 is not None:
	try:
		fn = args.file4
		if args.path is not None:
			fn = os.path.join(args.path, fn)
		df4 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
		
		t4 = np.array(df4['Time']*1e9)
		v4 = np.array(df4['Ampl']*1e3)
		
		t4 = t4 + args.toffns4
		v4 = v4 + args.voffmv4
	except:
		print(f"Failed to find file {args.file4}. Skipping file.")
		df4 = None
	label_4 = os.path.basename(args.file4)
else:
	df4 = None

# Create local variables
t = np.array(df['Time']*1e9)
v = np.array(df['Ampl']*1e3)
tc = np.array(dfc['Time']*1e9)
vc = np.array(dfc['Ampl']*1e3)

# Scale comparison file
vc = vc*args.scale

t_si = np.array(df['Time'])
v_si = np.array(df['Ampl'])
tc_si = np.array(dfc['Time'])
vc_si = np.array(dfc['Ampl'])

# Apply offsets
tc = tc + args.toffns
vc = vc + args.voffmv

# Create figure
fig1 = plt.figure(1, figsize=(args.figsizex, args.figsizey))

gs = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs[0, 0])
if args.title is not None:
	ax1a.set_title(args.title)
else:
	ax1a.set_title(f"{os.path.basename(args.filename)} vs {os.path.basename(args.comparefile)}")

if args.shortlegend:
	label_0 = "orig"
	label_c = "comp"
	label_2 = "comp2"
	label_3 = "comp3"
	label_4 = "comp4"
else:
	label_0 = os.path.basename(args.filename)
	label_c = os.path.basename(args.comparefile)

ALPHA = 0.4
color2 = (0, 0.65, 0)
color3 = (0.45, 0, 0.6)
color4 = (0, 0.45, 0.6)

# Override legends
if args.legend0 is not None:
	label_0 = args.legend0
if args.legend1 is not None:
	label_c = args.legend1
if args.legend2 is not None:
	label_2 = args.legend2
if args.legend3 is not None:
	label_3 = args.legend3
if args.legend4 is not None:
	label_4 = args.legend4
	
if args.flip:
	if df4 is not None:
		ax1a.plot(t4, v4, linestyle=':', marker='.', color=color4, alpha=ALPHA, label=label_4)
	if df3 is not None:
		ax1a.plot(t3, v3, linestyle=':', marker='.', color=color3, alpha=ALPHA, label=label_3)
	if df2 is not None:
		ax1a.plot(t2, v2, linestyle=':', marker='.', color=color2, alpha=ALPHA, label=label_2)
	ax1a.plot(tc, vc, linestyle=':', marker='.', color=(0.65, 0, 0), alpha=0.4, label=label_c)
	ax1a.plot(t, v, linestyle='--', marker='.', color=(0, 0, 0.65), alpha=0.4, label=label_0)
else:
	ax1a.plot(t, v, linestyle='--', marker='.', color=(0, 0, 0.65), alpha=0.4, label=label_0)
	ax1a.plot(tc, vc, linestyle=':', marker='.', color=(0.65, 0, 0), alpha=0.4, label=label_c)
	if df2 is not None:
		ax1a.plot(t2, v2, linestyle=':', marker='.', color=color2, alpha=ALPHA, label=label_2)
	if df3 is not None:
		ax1a.plot(t3, v3, linestyle=':', marker='.', color=color3, alpha=ALPHA, label=label_3)
	if df4 is not None:
		ax1a.plot(t4, v4, linestyle=':', marker='.', color=color4, alpha=ALPHA, label=label_4)
		
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

# Save pdf if requested
if args.savepdf is not None:
	
	# Get filename
	fn = args.savepdf
	if args.path is not None:
		fn = os.path.join(args.path, fn)
	
	# Save figure
	fig1.savefig(fn)
	
	print(f"Saved figure to: {fn}")

plt.show()