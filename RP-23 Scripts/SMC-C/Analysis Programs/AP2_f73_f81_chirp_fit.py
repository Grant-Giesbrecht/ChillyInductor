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
import pylogfile.base as plf
from rp23c_helper import *

log = plf.LogPile()

path = "/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-C July Campaign/1-July-2025 TD Measurements/"

trim_time = True
time_bounds = [[-1300, -800]]

#=====================================================================#
#           Load Doubler Data

try:
	fn = "C1RP23C_f73_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_73 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data73 = AP2Analyzer(df_73, -13.52, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f74_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_74 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data74 = AP2Analyzer(df_74, -6, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f75_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_75 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data75 = AP2Analyzer(df_75, 0, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f76_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_76 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data76 = AP2Analyzer(df_76, 6, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f77_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_77 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data77 = AP2Analyzer(df_77, 12, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f78_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_78 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data78 = AP2Analyzer(df_78, -3, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f79_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_79 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data79 = AP2Analyzer(df_79, 3, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f80_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_80 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data80 = AP2Analyzer(df_80, 9, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f81_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_81 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data81 = AP2Analyzer(df_81, 15, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

#=====================================================================#
#           Load Traditional Data

try:
	fn = "C1RP23C_f82_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_82 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data82 = AP2Analyzer(df_82, -24.74, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f83_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_83 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data83 = AP2Analyzer(df_83, -18, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f84_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_84 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data84 = AP2Analyzer(df_84, -12, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f85_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_85 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data85 = AP2Analyzer(df_85, -6, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f86_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_86 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data86 = AP2Analyzer(df_86, 0, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f87_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_87 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data87 = AP2Analyzer(df_87, 6, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f88_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_88 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data88 = AP2Analyzer(df_88, 12, log, trim_time, time_bounds)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

class AutoColorMap:
	
	def __init__(self, color_data:list, N, log:plf.LogPile):
		
		self.N = N
		self.color_data = color_data
		
		self.idx = 0
		
		self.log = log
	
	def __call__(self):
		try:
			cd = self.color_data[self.idx]
			self.idx += 1
			if self.idx >= self.N:
				self.idx = 0
				self.log.debug(f"Colormap auto-reset.")
		except:
			self.idx = 0
			cd = (0, 0, 0)
		
		return cd
	
	def reset(self):
		self.idx = 0

# Prepare data list
data_list = [data73, data74, data78, data75, data79, data76, data80, data77, data81]
data_list_trad = [data82, data83, data84, data85, data86, data87, data88]

#=====================================================================#
#           Plot Doubler Data

# Prepare color map
N = 9
cmdata = sample_colormap('viridis', N=N)
cmap = AutoColorMap(cmdata, N, log)

# Prepare alpha setting
ALP = 0.4

#------------------------------
# Figure 1

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

for dat in data_list:
	c = cmap()
	ax1a.plot(dat.time_ns, dat.wave_mV, linestyle=':', marker='.', label=f"P = {dat.power_dBm} dBm", color=c, alpha=ALP)
	ax1a.plot(dat.time_ns, dat.envelope, linestyle='-', label=f"Envelope, P = {dat.power_dBm} dBm", color=c, alpha=ALP, linewidth=0.5)
cmap.reset()
ax1a.legend()
ax1a.grid(True)


#------------------------------
# Figure 2

fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

ALP=0.8
for dat in data_list:
	c = cmap()
	ax2a.plot(dat.zcf_time, dat.zcf_freq, linestyle=':', marker='.', label=f"P = {dat.power_dBm} dBm", color=c, alpha=ALP, markersize=10)
cmap.reset()
# ax2a.legend()
ax2a.grid(True)
ax2a.set_ylim([0.085, 0.115])

ax2a.set_title("Doubler, 50 ns, IF=100 MHz, Power Sweep")
ax2a.set_xlabel("Time (ns)")
ax2a.set_ylabel("Frequency (GHz)")
ax2a.legend()

#------------------------------
# Figure 5

fig5 = plt.figure(5)
gs5 = fig5.add_gridspec(2, 2)
ax5a = fig5.add_subplot(gs5[0, 0])
ax5b = fig5.add_subplot(gs5[1, 0])
ax5c = fig5.add_subplot(gs5[0, 1])
ax5d = fig5.add_subplot(gs5[1, 1])

ALP=0.8
powers = []
K1s = []
lambdas = []
sigmas = []
for dat in data_list:
	powers.append(dat.power_dBm)
	K1s.append(dat.fit_results[2])
	lambdas.append(dat.fit_results[3])
	sigmas.append(dat.fit_results[4])

ax5a.plot(powers, K1s, linestyle=':', marker='+', color=(0, 0.6, 0.2))
ax5a.grid(True)
ax5a.set_xlabel("Power (dBm)")
ax5a.set_ylabel(f"L n2 I0")

ax5b.plot(powers, lambdas, linestyle=':', marker='+', color=(0, 0.6, 0.2))
ax5b.grid(True)
ax5b.set_xlabel("Power (dBm)")
ax5b.set_ylabel(f"Lambda")

ax5c.plot(powers, sigmas, linestyle=':', marker='+', color=(0, 0.6, 0.2))
ax5c.grid(True)
ax5c.set_xlabel("Power (dBm)")
ax5c.set_ylabel(f"Sigma")

fig5.tight_layout()

#=====================================================================#
#           Plot Traditional Data

# Prepare color map
N = 7
cmdata = sample_colormap('magma', N=N)
cmap = AutoColorMap(cmdata, N, log)

# Prepare alpha setting
ALP = 0.4

#------------------------------
# Figure 3

fig3 = plt.figure(3)
gs3 = fig3.add_gridspec(1, 1)
ax3a = fig3.add_subplot(gs3[0, 0])

for dat in data_list_trad:
	ax3a.plot(dat.time_ns, dat.wave_mV, linestyle=':', marker='x', label=f"P = {dat.power_dBm} dBm", color=cmap(), alpha=ALP)
cmap.reset()
ax3a.legend()
ax3a.grid(True)


#------------------------------
# Figure 4

fig4 = plt.figure(4)
gs4 = fig4.add_gridspec(1, 1)
ax4a = fig4.add_subplot(gs4[0, 0])

ALP=0.8
for dat in data_list_trad:
	ax4a.plot(dat.zcf_time, dat.zcf_freq, linestyle=':', marker='.', label=f"P = {dat.power_dBm} dBm", color=cmap(), alpha=ALP, markersize=10)
cmap.reset()
# ax2a.legend()
ax4a.grid(True)
ax4a.set_ylim([0.085, 0.115])







mplcursors.cursor(multiple=True)

plt.show()
