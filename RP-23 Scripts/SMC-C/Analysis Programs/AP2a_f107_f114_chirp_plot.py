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
from graf.base import sample_colormap, AutoColorMap
import pylogfile.base as plf
from rp23c_helper import *

log = plf.LogPile()

path = "/Volumes/M6 T7S/ARC0 PhD Data/RP-23 Qubit Readout/Data/SMC-C July Campaign/1-July-2025 TD Measurements/"

trim_time = True
time_bounds = [[-280, -210]]

IF = 0.3
IF_tol = 0.03
do_val_trim = False
do_fit = False

#=====================================================================#
#           Load Doubler Data

try:
	fn = "C1RP23C_f107_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_107 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data107 = AP2Analyzer(df_107, -6.62, log, trim_time, time_bounds, IF_GHz=IF, IF_chirp_tol_GHz=IF_tol, do_val_trim=do_val_trim, do_fit=do_fit)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f108_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_108 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data108 = AP2Analyzer(df_108, -3, log, trim_time, time_bounds, IF_GHz=IF, IF_chirp_tol_GHz=IF_tol, do_val_trim=do_val_trim, do_fit=do_fit)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f109_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_109 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data109 = AP2Analyzer(df_109, 0, log, trim_time, time_bounds, IF_GHz=IF, IF_chirp_tol_GHz=IF_tol, do_val_trim=do_val_trim, do_fit=do_fit)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f110_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_110 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data110 = AP2Analyzer(df_110, 3, log, trim_time, time_bounds, IF_GHz=IF, IF_chirp_tol_GHz=IF_tol, do_val_trim=do_val_trim, do_fit=do_fit)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f111_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_111 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data111 = AP2Analyzer(df_111, 6, log, trim_time, time_bounds, IF_GHz=IF, IF_chirp_tol_GHz=IF_tol, do_val_trim=do_val_trim, do_fit=do_fit)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f112_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_112 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data112 = AP2Analyzer(df_112, 9, log, trim_time, time_bounds, IF_GHz=IF, IF_chirp_tol_GHz=IF_tol, do_val_trim=do_val_trim, do_fit=do_fit)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f113_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_113 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data113 = AP2Analyzer(df_113, 12, log, trim_time, time_bounds, IF_GHz=IF, IF_chirp_tol_GHz=IF_tol, do_val_trim=do_val_trim, do_fit=do_fit)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()

try:
	fn = "C1RP23C_f114_AVG5k_00000.txt"
	fn = os.path.join(path, fn)
	
	df_114 = pd.read_csv(fn, skiprows=4, encoding='utf-8')
	data114 = AP2Analyzer(df_114, 15, log, trim_time, time_bounds, IF_GHz=IF, IF_chirp_tol_GHz=IF_tol, do_val_trim=do_val_trim, do_fit=do_fit)
	
except Exception as e:
	print(f"Failed to load file {fn} ({e}). Aborting.")
	sys.exit()


#=====================================================================#
#           Load Traditional Data




# Prepare data list
data_list = [data107, data108, data109, data110, data111, data112, data113, data114]
# data_list_trad = [data82, data83, data84, data85, data86, data87, data88]

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
	ax2a.plot(dat.zcf_time, dat.zcf_freq, linestyle=':', marker='.', label=f"Waveform, P = {dat.power_dBm} dBm", color=c, alpha=ALP, markersize=10)
cmap.reset()
# ax2a.legend()
ax2a.grid(True)
ax2a.set_ylim([0.27, 0.33])

#------------------------------
# Figure 5

# fig5 = plt.figure(5)
# gs5 = fig5.add_gridspec(2, 2)
# ax5a = fig5.add_subplot(gs5[0, 0])
# ax5b = fig5.add_subplot(gs5[1, 0])
# ax5c = fig5.add_subplot(gs5[0, 1])
# ax5d = fig5.add_subplot(gs5[1, 1])

# ALP=0.8
# powers = []
# K1s = []
# lambdas = []
# sigmas = []
# for dat in data_list:
# 	powers.append(dat.power_dBm)
# 	K1s.append(dat.fit_results[2])
# 	lambdas.append(dat.fit_results[3])
# 	sigmas.append(dat.fit_results[4])

# ax5a.plot(powers, K1s, linestyle=':', marker='+', color=(0, 0.6, 0.2))
# ax5a.grid(True)
# ax5a.set_xlabel("Power (dBm)")
# ax5a.set_ylabel(f"L n2 I0")

# ax5b.plot(powers, lambdas, linestyle=':', marker='+', color=(0, 0.6, 0.2))
# ax5b.grid(True)
# ax5b.set_xlabel("Power (dBm)")
# ax5b.set_ylabel(f"Lambda")

# ax5c.plot(powers, sigmas, linestyle=':', marker='+', color=(0, 0.6, 0.2))
# ax5c.grid(True)
# ax5c.set_xlabel("Power (dBm)")
# ax5c.set_ylabel(f"Sigma")

# fig5.tight_layout()

#=====================================================================#
#           Plot Traditional Data

# # Prepare color map
# N = 7
# cmdata = sample_colormap('magma', N=N)
# cmap = AutoColorMap(cmdata, N, log)

# # Prepare alpha setting
# ALP = 0.4

# #------------------------------
# # Figure 3

# fig3 = plt.figure(3)
# gs3 = fig3.add_gridspec(1, 1)
# ax3a = fig3.add_subplot(gs3[0, 0])

# for dat in data_list_trad:
# 	ax3a.plot(dat.time_ns, dat.wave_mV, linestyle=':', marker='x', label=f"P = {dat.power_dBm} dBm", color=cmap(), alpha=ALP)
# cmap.reset()
# ax3a.legend()
# ax3a.grid(True)


# #------------------------------
# # Figure 4

# fig4 = plt.figure(4)
# gs4 = fig4.add_gridspec(1, 1)
# ax4a = fig4.add_subplot(gs4[0, 0])

# ALP=0.8
# for dat in data_list_trad:
# 	ax4a.plot(dat.zcf_time, dat.zcf_freq, linestyle=':', marker='.', label=f"P = {dat.power_dBm} dBm", color=cmap(), alpha=ALP, markersize=10)
# cmap.reset()
# # ax2a.legend()
# ax4a.grid(True)
# ax4a.set_ylim([0.085, 0.115])







mplcursors.cursor(multiple=True)

plt.show()
