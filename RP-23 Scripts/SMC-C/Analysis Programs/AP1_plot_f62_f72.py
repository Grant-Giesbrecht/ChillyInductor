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
from graf.base import sample_colormap
from scipy.signal import hilbert

path = "/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-C July Campaign/1-July-2025 TD Measurements/"

try:
	fn = "C1RP23C_f62_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_62 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_62 = np.array(df_62['Time']*1e9)
	v_62 = np.array(df_62['Ampl']*1e3)
	
	a_62 = hilbert(v_62)
	env_62 = np.abs(a_62)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f63_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_63 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_63 = np.array(df_63['Time']*1e9)
	v_63 = np.array(df_63['Ampl']*1e3)
	
	a_63 = hilbert(v_63)
	env_63 = np.abs(a_63)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f64_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_64 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_64 = np.array(df_64['Time']*1e9)
	v_64 = np.array(df_64['Ampl']*1e3)
	
	a_64 = hilbert(v_64)
	env_64 = np.abs(a_64)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f65_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_65 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_65 = np.array(df_65['Time']*1e9)
	v_65 = np.array(df_65['Ampl']*1e3)
	
	a_65 = hilbert(v_65)
	env_65 = np.abs(a_65)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f66_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_66 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_66 = np.array(df_66['Time']*1e9)
	v_66 = np.array(df_66['Ampl']*1e3)
	
	a_66 = hilbert(v_66)
	env_66 = np.abs(a_66)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f67_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_67 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_67 = np.array(df_67['Time']*1e9)
	v_67 = np.array(df_67['Ampl']*1e3)
	
	a_67 = hilbert(v_67)
	env_67 = np.abs(a_67)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f68_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_68 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_68 = np.array(df_68['Time']*1e9)
	v_68 = np.array(df_68['Ampl']*1e3)
	
	a_68 = hilbert(v_68)
	env_68 = np.abs(a_68)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f69_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_69 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_69 = np.array(df_69['Time']*1e9)
	v_69 = np.array(df_69['Ampl']*1e3)
	
	a_69 = hilbert(v_69)
	env_69 = np.abs(a_69)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f70_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_70 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_70 = np.array(df_70['Time']*1e9)
	v_70 = np.array(df_70['Ampl']*1e3)
	
	a_70 = hilbert(v_70)
	env_70 = np.abs(a_70)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f71_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_71 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	
	t_71 = np.array(df_71['Time']*1e9)
	v_71 = np.array(df_71['Ampl']*1e3)
	
	a_71 = hilbert(v_71)
	env_71 = np.abs(a_71)
except:
	print(f"Failed to find file {fn}. Aborting.")
	sys.exit()

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

ax1a.plot(t_62, v_62+1.6, linestyle=':', marker='.', label="No Bias")
ax1a.plot(t_63, v_63+1.6, linestyle=':', marker='.', label="V_{bias} = 0.275")
ax1a.plot(t_63, -1*(v_63+1.6), linestyle=':', marker='.', label="(V_{bias} = 0.275) Inverted")
ax1a.legend()
ax1a.grid(True)


cm = sample_colormap('viridis', N=8)

fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

ALP = 0.2
ax2a.plot(t_64, v_64, linestyle=":", marker='.', label="0.275 V", alpha=ALP, color=cm[0])
ax2a.plot(t_65, v_65, linestyle=":", marker='.', label="0.35 V", alpha=ALP, color=cm[1])
ax2a.plot(t_66, v_66, linestyle=":", marker='.', label="0.45 V", alpha=ALP, color=cm[2])
ax2a.plot(t_67, v_67, linestyle=":", marker='.', label="0.55 V", alpha=ALP, color=cm[3])
ax2a.plot(t_68, v_68, linestyle=":", marker='.', label="0.65 V", alpha=ALP, color=cm[4])
ax2a.plot(t_69, v_69, linestyle=":", marker='.', label="0.75 V", alpha=ALP, color=cm[5])
ax2a.plot(t_70, v_70, linestyle=":", marker='.', label="0.85 V", alpha=ALP, color=cm[6])
ax2a.plot(t_71, v_71, linestyle=":", marker='.', label="0.95 V", alpha=ALP, color=cm[7])

ax2a.legend()
ax2a.grid(True)



fig3 = plt.figure(3)
gs3 = fig3.add_gridspec(1, 1)
ax3a = fig3.add_subplot(gs3[0, 0])

ALP = 0.2
ax3a.plot(t_64, env_64, linestyle=":", marker='.', label="0.275 V", alpha=ALP, color=cm[0])
ax3a.plot(t_65, env_65, linestyle=":", marker='.', label="0.35 V", alpha=ALP, color=cm[1])
ax3a.plot(t_66, env_66, linestyle=":", marker='.', label="0.45 V", alpha=ALP, color=cm[2])
ax3a.plot(t_67, env_67, linestyle=":", marker='.', label="0.55 V", alpha=ALP, color=cm[3])
ax3a.plot(t_68, env_68, linestyle=":", marker='.', label="0.65 V", alpha=ALP, color=cm[4])
ax3a.plot(t_69, env_69, linestyle=":", marker='.', label="0.75 V", alpha=ALP, color=cm[5])
ax3a.plot(t_70, env_70, linestyle=":", marker='.', label="0.85 V", alpha=ALP, color=cm[6])
ax3a.plot(t_71, env_71, linestyle=":", marker='.', label="0.95 V", alpha=ALP, color=cm[7])

ax3a.legend()
ax3a.grid(True)


mplcursors.cursor(multiple=True)

plt.show()
