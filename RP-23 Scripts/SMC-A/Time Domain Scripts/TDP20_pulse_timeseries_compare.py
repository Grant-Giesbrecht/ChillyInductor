''' Goal is to compare multiple pulses across one time series. The question is, do 
sequential pulses even look the same? If not, we have a bigger problem at hand!
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
	
	def __init__(self, log, app, data_manager, ts:list, vs:list, offsets:list):
		super().__init__(log, app, data_manager, window_title="Time Series Analyzer")
		
		self.main_grid = QGridLayout()
		
		self.pulses_tab_widget = bh.BHTabWidget(self)
		# self.tab_widget.addTab()
		
		self.add_basic_menu_bar()
		
		self.main_grid.addWidget(self.pulses_tab_widget, 0, 0)
		
		self.central_widget = QWidget()
		self.central_widget.setLayout(self.main_grid)
		self.setCentralWidget(self.central_widget)
		
		self.sub_widgets = []
		
		self.ts = ts
		self.vs = vs
		self.offsets = offsets
		
		self.cmap = sample_colormap(cmap_name='plasma', N=len(ts))
		
		self.make_pulse_tabs()
		
		self.show()
	
	def render_pulse(self, pw, idx:int):
		
		# Clear data
		pw.axes[0].cla()
		
		print(self.ts[idx])
		print(self.offsets[idx])
		
		pw.axes[0].plot(self.ts[idx]+self.offsets[idx], self.vs[idx], linestyle=':', marker='.', color=self.cmap[idx])
		
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
trim_times = []
trim_times.append([-6044.5, -5927.8])
trim_times.append([-5789.9, -5683.3])
trim_times.append([-5537.2, -5424.2])
# trim_times.append([])

hoffsets = []
hoffsets.append(0)
hoffsets.append(0)
hoffsets.append(0)

##--------------------------------------------------------
# Import data
df = pd.read_csv(filename, skiprows=4, encoding='utf-8')

# Create local variables
t = np.array(df['Time']*1e9)
v = np.array(df['Ampl']*1e3)

t_si = np.array(df['Time'])
v_si = np.array(df['Ampl'])

#-------------------------------------------------------------------
## Perform trimming and envelope detection

ts = []
vs = []
for tt in trim_times:
	log.info(f"Trimming times: >{tt}<.")
	# Trim
	t_, v_ = trim_time_series(t, v, tt[0], tt[1])
	
	# Add to list
	ts.append(np.array(t_))
	vs.append(np.array(v_))

#-------------------------------------------------------------------
## Plot

print("Did things")

# Create app object
app = QtWidgets.QApplication(sys.argv)
app.setStyle(f"Fusion")
# app.setWindowIcon

def void():
	pass

# Create Data Manager
data_manager = bh.BHDatasetManager(log, load_function=void)

window = TDP20MainWindow(log, app, data_manager, ts, vs, hoffsets)

app.exec()