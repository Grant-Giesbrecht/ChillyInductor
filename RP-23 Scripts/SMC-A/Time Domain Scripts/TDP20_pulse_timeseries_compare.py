''' Goal is to compare multiple pulses across one time series. The question is, do 
sequential pulses even look the same? If not, we have a bigger problem at hand!

NOTE: In this example, the "data manager" class and tool is circumvented. We can
get away with this because of how simple the program is. However, to add sliders
or make it so it can hot swap data files, we'd need to introduce the data manager.
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz
import time
import os
import sys
import argparse
from graf.base import sample_colormap

import blackhole.widgets as bhw
import blackhole.base as bh
import pandas as pd

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction, QActionGroup, QDoubleValidator, QIcon, QFontDatabase, QFont, QPixmap
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QWidget, QTabWidget, QLabel, QGridLayout, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QMainWindow, QSlider, QPushButton, QGroupBox, QListWidget, QFileDialog, QProgressBar, QStatusBar

import pylogfile.base as plf
import numpy as np
import sys
import time
from scipy.signal import hilbert
from colorama import Fore, Style
from scipy.optimize import curve_fit
import matplotlib.patches as patches
import copy

import argparse

#-------------------------------------------------------------
# Define things

log = plf.LogPile()

def trim_time_series(t, y, t_start, t_end):
	
	"""
	Trims a time series to keep only values where t is within [t_start, t_end].

	Parameters:
		t : np.array - Time values
		y : np.array - Corresponding data values
		t_start : float - Start time
		t_end : float - End time

	Returns:
		t_trimmed, y_trimmed : np.array - Trimmed time and data arrays
	"""
	mask = (t >= t_start) & (t <= t_end)  # Create a boolean mask
	return t[mask], y[mask]  # Apply mask to both arrays

class TDP20MainWindow(bh.BHMainWindow):
	
	def __init__(self, log, app, data_manager, ts:list, vs:list, envs:list, lpf_env:list, offsets:list):
		super().__init__(log, app, data_manager, window_title="TDP20 Time Series Analyzer")
		
		self.ts = ts
		self.vs = vs
		self.offsets = offsets
		self.envelopes = envs
		self.lpf_env = lpf_env
		
		self.cmap = sample_colormap(cmap_name='viridis', N=len(ts))
		
		self.main_grid = QGridLayout()
		
		self.overlay_plot = bhw.BHMultiPlotWidget(self, grid_dim=[1, 1], plot_locations=[[0, 0]], custom_render_func=lambda pw:self.render_overlay(pw, False))
		
		self.overlay_plot_lpf = bhw.BHMultiPlotWidget(self, grid_dim=[1, 1], plot_locations=[[0, 0]], custom_render_func=lambda pw:self.render_overlay(pw, True))
		
		self.pulses_tab_widget = bh.BHTabWidget(self)
		
		# Create super tab
		self.super_tab_widget = bh.BHTabWidget(self)
		self.super_tab_widget.addTab(self.overlay_plot, "Envelope Overlay")
		self.super_tab_widget.addTab(self.overlay_plot_lpf, "Low-Pass Overlay")
		self.super_tab_widget.addTab(self.pulses_tab_widget, "Individual Pulses")
		
		self.add_basic_menu_bar()
		
		self.main_grid.addWidget(self.super_tab_widget, 0, 0)
		
		self.central_widget = QWidget()
		self.central_widget.setLayout(self.main_grid)
		self.setCentralWidget(self.central_widget)
		
		self.sub_widgets = []
		
		self.make_pulse_tabs()
		
		self.show()
	
	def render_overlay(self, pw, use_lpf:bool):
		
		pw.axes[0].cla()
		
		for idx, t_ in enumerate(self.ts):
			
			if use_lpf:
				pw.axes[0].plot(t_ + self.offsets[idx]-t_[0], self.lpf_env[idx], linestyle='-', alpha=0.6, color=self.cmap[idx], label=f"Pulse {idx}")
				pw.axes[0].set_title("Low-Pass Envelope Overlay")
			else:
				pw.axes[0].plot(t_ + self.offsets[idx]-t_[0], self.envelopes[idx], linestyle='-', alpha=0.6, color=self.cmap[idx], label=f"Pulse {idx}")
				pw.axes[0].set_title("Envelope Overlay")
				
		pw.axes[0].set_xlabel("Normalized Time (ns)")
		pw.axes[0].set_ylabel("Amplitude (mV)")
		
		pw.axes[0].legend()
		
		pw.axes[0].grid(True)
		
	def render_pulse(self, pw, idx:int):
		
		# Clear data
		pw.axes[0].cla()
		
		pw.axes[0].plot(self.ts[idx]+self.offsets[idx], self.vs[idx], linestyle=':', marker='.', color=self.cmap[idx])
		pw.axes[0].plot(self.ts[idx]+self.offsets[idx], self.envelopes[idx], linestyle='-', color=(1, 0.55, 0), alpha=0.6)
		
		pw.axes[0].set_xlabel("Time (ns)")
		pw.axes[0].set_xlabel("Amplitude (mV)")
		
		pw.axes[0].grid(True)
		
	
	def make_pulse_tabs(self):
		
		for idx, t in enumerate(self.ts):
			
			# Create plot
			self.sub_widgets.append(bhw.BHMultiPlotWidget(self, grid_dim=[1, 1], plot_locations=[[0, 0]], custom_render_func=lambda pw:self.render_pulse(pw, idx)))
			
			# Add to tab
			self.pulses_tab_widget.addTab(self.sub_widgets[-1], f"Pulse {idx}")

##--------------------------------------------------------
# Select data source

DATADIR = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-A", "Time Domain Measurements", "2025-03-25")

#NOTE: Comparing two direct-drive pulses to see how good subtraction can look.
filename = f"{DATADIR}/C1Medwav_0,275V_-9dBm_2,3679GHz_15Pi_sig35ns_r22_00000.txt"
voffset = 3.775
trim_times = []
trim_times.append([-6044.5, -5927.8])
trim_times.append([-5789.9, -5683.3])
trim_times.append([-5537.2, -5424.2])
# trim_times.append([])

hoffsets = []
hoffsets.append(0)
hoffsets.append(4.92)
hoffsets.append(7.76)

##--------------------------------------------------------
# Import data
df = pd.read_csv(filename, skiprows=4, encoding='utf-8')

t_si = np.array(df['Time'])
v_si = np.array(df['Ampl'])+(voffset*1e-3)

# Create local variables
t = np.array(t_si*1e9)
v = np.array(v_si*1e3)

an_sig = hilbert(v)
env = np.abs(an_sig)

# Lowpass filter
b, a = butter(3, 0.1, fs=1/(t[1]-t[0]), btype='low', analog=False)
lpf_env = lfilter(b, a, env)

#-------------------------------------------------------------------
## Perform trimming and envelope detection

ts = []
vs = []
envs = []
lpfenvs = []
for tt in trim_times:
	log.info(f"Trimming times: >{tt}<.")
	
	# Trim
	t_, v_ = trim_time_series(t, v, tt[0], tt[1])
	_, e_ = trim_time_series(t, env, tt[0], tt[1])
	_, lpfe_ = trim_time_series(t, lpf_env, tt[0], tt[1])
	
	# Add to list
	ts.append(np.array(t_))
	vs.append(np.array(v_))
	envs.append(np.array(e_))
	lpfenvs.append(np.array(lpfe_))

#-------------------------------------------------------------------
## Plot

# Create app object
app = QtWidgets.QApplication(sys.argv)
app.setStyle(f"Fusion")
# app.setWindowIcon

def void():
	pass

# Create Data Manager
data_manager = bh.BHDatasetManager(log, load_function=void)

window = TDP20MainWindow(log, app, data_manager, ts, vs, envs, lpfenvs, hoffsets)

app.exec()