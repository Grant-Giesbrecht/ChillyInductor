import sys
import matplotlib
matplotlib.use('qtagg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from pylogfile.base import *

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QTabWidget, QLabel, QSlider

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from chillyinductor.rp22_helper import *
from colorama import Fore, Style
from ganymede import *
from pylogfile.base import *
import sys
import numpy as np
import pickle
from matplotlib.widgets import Slider, Button
from abc import abstractmethod, ABC
# import qdarktheme

# TODO: Important
#
# * Tests show if only the presently displayed graph is rendered, when the sliders are adjusted, the update speed is far better (no kidding).
# * Tests also show pre-allocating a mask isn't super useful. It's mostly the matplotlib rendering that slows it down.

# TODO:
# 1. Init graph properly (says void and stuff)
# 2. Add labels to values on freq and power sliders
# 3. Add save log, save graph, zoom controls, etc.
# 4. Frequency-domain plots!
# 5. Datatips
# 6. Buttons to zoom to max efficiency
# 7. Data panel at bottom showing file stats:
#		- collection date, operator notes, file size, num points, open log in lumberjack button
# 8. Better way to load any data file

# TODO: Graphs to add
# 1. Applied voltage, measured current, target current, estimated additional impedance.
# 2. Temperature vs time
#

#------------------------------------------------------------
# Import Data

sp_datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm', "VNA Traces"])
if sp_datapath is None:
	print(f"{Fore.RED}Failed to find s-parameter data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located s-parameter data directory at: {Fore.LIGHTBLACK_EX}{sp_datapath}{Style.RESET_ALL}")

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm'])
# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 2 43mm'])
# datapath = '/Volumes/M5 PERSONAL/data_transfer'
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

# filename = "RP22B_MP3_t1_31July2024_R4C4T1_r1_autosave.hdf"
# filename = "RP22B_MP3_t1_1Aug2024_R4C4T1_r1.hdf"
filename = "RP22B_MP3_t2_8Aug2024_R4C4T1_r1.hdf"
# filename = "RP22B_MP3a_t3_19Aug2024_R4C4T2_r1.hdf"
# filename ="RP22B_MP3a_t2_20Aug2024_R4C4T2_r1_autosave.hdf"

sp_filename = "Sparam_31July2024_-30dBm_R4C4T1.csv"


sp_analysis_file = os.path.join(sp_datapath, sp_filename)#"Sparam_31July2024_-30dBm_R4C4T1.csv")

analysis_file = os.path.join(datapath, filename)

log = LogPile()

##--------------------------------------------
# Read S-Parameters

try:
	sparam_data = read_rohde_schwarz_csv(sp_analysis_file)
except Exception as e:
	print(f"Failed to read S-parameter CSV file. {e}")
	sys.exit()

S11 = sparam_data.S11_real + complex(0, 1)*sparam_data.S11_imag
S21 = sparam_data.S21_real + complex(0, 1)*sparam_data.S21_imag
S11_dB = lin_to_dB(np.abs(S11))
S21_dB = lin_to_dB(np.abs(S21))
S_freq_GHz = sparam_data.freq_Hz/1e9

##--------------------------------------------
# Read HDF5 File


print("Loading file contents into memory")
# log.info("Loading file contents into memory")

t_hdfr_0 = time.time()
with h5py.File(analysis_file, 'r') as fh:
	
	# Read primary dataset
	GROUP = 'dataset'
	freq_rf_GHz = fh[GROUP]['freq_rf_GHz'][()]
	power_rf_dBm = fh[GROUP]['power_rf_dBm'][()]
	
	waveform_f_Hz = fh[GROUP]['waveform_f_Hz'][()]
	waveform_s_dBm = fh[GROUP]['waveform_s_dBm'][()]
	waveform_rbw_Hz = fh[GROUP]['waveform_rbw_Hz'][()]
	
	MFLI_V_offset_V = fh[GROUP]['MFLI_V_offset_V'][()]
	requested_Idc_mA = fh[GROUP]['requested_Idc_mA'][()]
	raw_meas_Vdc_V = fh[GROUP]['raw_meas_Vdc_V'][()]
	Idc_mA = fh[GROUP]['Idc_mA'][()]
	detect_normal = fh[GROUP]['detect_normal'][()]
	
	temperature_K = fh[GROUP]['temperature_K'][()]

##--------------------------------------------
# Generate Mixing Products lists

rf1 = spectrum_peak_list(waveform_f_Hz, waveform_s_dBm, freq_rf_GHz*1e9)
rf2 = spectrum_peak_list(waveform_f_Hz, waveform_s_dBm, freq_rf_GHz*2e9)
rf3 = spectrum_peak_list(waveform_f_Hz, waveform_s_dBm, freq_rf_GHz*3e9)

rf1W = dBm2W(rf1)
rf2W = dBm2W(rf2)
rf3W = dBm2W(rf3)

#-------------------------------------------

req_bias_list = np.unique(requested_Idc_mA)
unique_bias = req_bias_list
unique_pwr = np.unique(power_rf_dBm)
unique_freqs = np.unique(freq_rf_GHz)

##--------------------------------------------
# Create GUI

def get_graph_lims(data:list, step=None):
	
	umin = np.min(data)
	umax = np.max(data)
	
	return [np.floor(umin/step)*step, np.ceil(umax/step)*step]

# class MplCanvas(FigureCanvasQTAgg):
	
# 	def __init__(self, parent=None, width=5, height=4, dpi=100):
# 		fig = Figure(figsize=(width, height), dpi=dpi)
# 		self.axes = fig.add_subplot(111)
# 		super(MplCanvas, self).__init__(fig)
		
class TabPlotWidget(QWidget):
	
	def __init__(self):
		super().__init__()
		self._is_active = False # Indicates if the current tab is displayed
		self.plot_is_current = False # Does the plot need to be re-rendered?
	
	def is_active(self):
		return self._is_active
	
	def set_active(self, b:bool):
		self._is_active = b
		self.plot_data()
	
	def plot_data(self):
		''' If the plot needs to be updated (active and out of date) re-renders the graph and displays it. '''
		
		if self.is_active() and (not self.plot_is_current):
			self.render_plot()
	
	def update_plot(self):
		''' Marks the plot to be updated eventually. '''
		
		# Indicate need to replot
		self.plot_is_current = False
		
		self.plot_data()
		
	@abstractmethod
	def render_plot(self):
		pass

class HarmGenFreqDomainPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01}
		self.global_conditions = global_conditions
		
		self.manual_init()
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1)
		
		# Estimate system Z
		expected_Z = MFLI_V_offset_V[1]/(requested_Idc_mA[1]/1e3) #TODO: Do something more general than index 1
		system_Z = MFLI_V_offset_V/(Idc_mA/1e3)
		self.extra_z = system_Z - expected_Z
		
		
		self.render_plot()
		
		# Create widgets
		self.fig1c = FigureCanvas(self.fig1)
		self.toolbar1 = NavigationToolbar2QT(self.fig1c, self)
		
		# Add widgets to parent-widget and set layout
		self.grid = QtWidgets.QGridLayout()
		self.grid.addWidget(self.toolbar1, 0, 0)
		self.grid.addWidget(self.fig1c, 1, 0)
		self.setLayout(self.grid)
	
	def manual_init(self):
		
		self.ylims1 = get_graph_lims(np.concatenate((rf1, rf2, rf3)), step=10)
	
	def get_condition(self, c:str):
		
		if c in self.conditions:
			return self.conditions[c]
		elif c in self.global_conditions:
			return self.global_conditions[c]
		else:
			return None
	
	def render_plot(self):
		b = self.get_condition('sel_bias_mA')
		p = self.get_condition('sel_power_dBm')
		use_fund = self.get_condition('freqxaxis_isfund')
		
		# Filter relevant data
		mask_bias = (requested_Idc_mA == b)
		mask_pwr = (power_rf_dBm == p)
		mask = (mask_bias & mask_pwr)
		
		# Plot results
		self.ax1.cla()
		
		# Check correct number of points
		mask_len = np.sum(mask)
		if len(unique_freqs) != mask_len:
			log.warning(f"Cannot display data: Mismatched number of points (bias = {b} mA, pwr = {p} dBm, mask: {mask_len}, freq: {len(unique_freqs)})")
			self.fig1.canvas.draw_idle()
			return
		
		if use_fund:
			self.ax1.plot(freq_rf_GHz[mask], rf1[mask], linestyle=':', marker='o', markersize=1.5, color=(0, 0.7, 0))
			self.ax1.plot(freq_rf_GHz[mask], rf2[mask], linestyle=':', marker='o', markersize=1.5, color=(0, 0, 0.7))
			self.ax1.plot(freq_rf_GHz[mask], rf3[mask], linestyle=':', marker='o', markersize=1.5, color=(0.7, 0, 0))
			self.ax1.set_xlabel("Fundamental Frequency (GHz)")
		else:
			self.ax1.plot(freq_rf_GHz[mask], rf1[mask], linestyle=':', marker='o', markersize=1.5, color=(0, 0.7, 0))
			self.ax1.plot(freq_rf_GHz[mask]*2, rf2[mask], linestyle=':', marker='o', markersize=1.5, color=(0, 0, 0.7))
			self.ax1.plot(freq_rf_GHz[mask]*3, rf3[mask], linestyle=':', marker='o', markersize=1.5, color=(0.7, 0, 0))
			self.ax1.set_xlabel("Fundamental/Harmonic Tone Frequency (GHz)")
			
		self.ax1.set_title(f"Bias = {b} mA, p = {p} dBm")
		self.ax1.set_ylabel("Power (dBm)")
		self.ax1.legend(["Fundamental", "2nd Harm.", "3rd Harm."])
		self.ax1.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax1.set_ylim(self.ylims1)
		
		self.fig1.tight_layout()
		
		self.fig1.canvas.draw_idle()

class CE23FreqDomainPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01}
		self.global_conditions = global_conditions
		
		self.manual_init()
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1)
		self.fig2, self.ax2 = plt.subplots(1, 1)
		
		# Estimate system Z
		expected_Z = MFLI_V_offset_V[1]/(requested_Idc_mA[1]/1e3) #TODO: Do something more general than index 1
		system_Z = MFLI_V_offset_V/(Idc_mA/1e3)
		self.extra_z = system_Z - expected_Z
		
		
		self.render_plot()
		
		# Create widgets
		self.fig1c = FigureCanvas(self.fig1)
		self.toolbar1 = NavigationToolbar2QT(self.fig1c, self)
		self.fig2c = FigureCanvas(self.fig2)
		self.toolbar2 = NavigationToolbar2QT(self.fig2c, self)
		
		# Add widgets to parent-widget and set layout
		self.grid = QtWidgets.QGridLayout()
		self.grid.addWidget(self.toolbar1, 0, 0)
		self.grid.addWidget(self.fig1c, 1, 0)
		self.grid.addWidget(self.toolbar2, 0, 1)
		self.grid.addWidget(self.fig2c, 1, 1)
		self.setLayout(self.grid)
	
	def manual_init(self):
		
		# Calculate total power and CE
		self.total_power = rf1W + rf2W + rf3W
		self.ce2 = rf2W/self.total_power*100
		self.ce3 = rf3W/self.total_power*100
		
		# Get autoscale choices
		umax1 = np.max(self.ce2)
		umin1 = np.min(self.ce2)
		umax2 = np.max(self.ce3)
		umin2 = np.min(self.ce3)
		
		rstep1 = self.get_condition('rounding_step1')
		rstep2 = self.get_condition('rounding_step2')
		if rstep1 is None:
			rstep1 = 10
		if rstep2 is None:
			rstep2 = 10
		
		self.ylims1 = [np.floor(umin1/rstep1)*rstep1, np.ceil(umax1/rstep1)*rstep1]
		self.ylims2 = [np.floor(umin2/rstep2)*rstep2, np.ceil(umax2/rstep2)*rstep2]
	
	def get_condition(self, c:str):
		
		if c in self.conditions:
			return self.conditions[c]
		elif c in self.global_conditions:
			return self.global_conditions[c]
		else:
			return None
	
	def render_plot(self):
		b = self.get_condition('sel_bias_mA')
		p = self.get_condition('sel_power_dBm')
		use_fund = self.get_condition('freqxaxis_isfund')
		
		# Filter relevant data
		mask_bias = (requested_Idc_mA == b)
		mask_pwr = (power_rf_dBm == p)
		mask = (mask_bias & mask_pwr)
		
		# Plot results
		self.ax1.cla()
		self.ax2.cla()
		
		# Check correct number of points
		mask_len = np.sum(mask)
		if len(unique_freqs) != mask_len:
			log.warning(f"Cannot display data: Mismatched number of points (bias = {b} mA, pwr = {p} dBm, mask: {mask_len}, freq: {len(unique_freqs)})")
			self.fig1.canvas.draw_idle()
			return
		
		if use_fund:
			self.ax1.plot(freq_rf_GHz[mask], self.ce2[mask], linestyle=':', marker='o', markersize=2, color=(0.6, 0, 0.7))
			self.ax2.plot(freq_rf_GHz[mask], self.ce3[mask], linestyle=':', marker='o', markersize=2, color=(0.45, 0.05, 0.1))
			self.ax1.set_xlabel("Fundamental Frequency (GHz)")
			self.ax2.set_xlabel("Fundamental Frequency (GHz)")
		else:
			self.ax1.plot(freq_rf_GHz[mask]*2, self.ce2[mask], linestyle=':', marker='o', markersize=2, color=(0.6, 0, 0.7))
			self.ax2.plot(freq_rf_GHz[mask]*3, self.ce3[mask], linestyle=':', marker='o', markersize=2, color=(0.45, 0.05, 0.1))
			self.ax1.set_xlabel("2nd Harmonic Frequency (GHz)")
			self.ax2.set_xlabel("3rd Harmonic Frequency (GHz)")
			
		self.ax1.set_title(f"Bias = {b} mA, p = {p} dBm")
		self.ax1.set_ylabel("2nd Harm. Conversion Efficiency (%)")
		self.ax1.grid(True)
		
		self.ax2.set_title(f"Bias = {b} mA, p = {p} dBm")
		self.ax2.set_ylabel("3rd Harm. Conversion Efficiency (%)")
		self.ax2.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax1.set_ylim(self.ylims1)
			self.ax2.set_ylim(self.ylims2)
		
		self.fig1.tight_layout()
		self.fig2.tight_layout()
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class CE23BiasDomainPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01}
		self.global_conditions = global_conditions
		
		self.manual_init()
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1)
		self.fig2, self.ax2 = plt.subplots(1, 1)
		
		# Estimate system Z
		expected_Z = MFLI_V_offset_V[1]/(requested_Idc_mA[1]/1e3) #TODO: Do something more general than index 1
		system_Z = MFLI_V_offset_V/(Idc_mA/1e3)
		self.extra_z = system_Z - expected_Z
		
		
		self.render_plot()
		
		# Create widgets
		self.fig1c = FigureCanvas(self.fig1)
		self.toolbar1 = NavigationToolbar2QT(self.fig1c, self)
		self.fig2c = FigureCanvas(self.fig2)
		self.toolbar2 = NavigationToolbar2QT(self.fig2c, self)
		
		# Add widgets to parent-widget and set layout
		self.grid = QtWidgets.QGridLayout()
		self.grid.addWidget(self.toolbar1, 0, 0)
		self.grid.addWidget(self.fig1c, 1, 0)
		self.grid.addWidget(self.toolbar2, 0, 1)
		self.grid.addWidget(self.fig2c, 1, 1)
		self.setLayout(self.grid)
	
	def manual_init(self):
		
		# Calculate total power and CE
		self.total_power = rf1W + rf2W + rf3W
		self.ce2 = rf2W/self.total_power*100
		self.ce3 = rf3W/self.total_power*100
		
		# Get autoscale choices
		umax1 = np.max(self.ce2)
		umin1 = np.min(self.ce2)
		umax2 = np.max(self.ce3)
		umin2 = np.min(self.ce3)
		
		rstep1 = self.get_condition('rounding_step1')
		rstep2 = self.get_condition('rounding_step2')
		if rstep1 is None:
			rstep1 = 10
		if rstep2 is None:
			rstep2 = 10
		
		self.ylims1 = [np.floor(umin1/rstep1)*rstep1, np.ceil(umax1/rstep1)*rstep1]
		self.ylims2 = [np.floor(umin2/rstep2)*rstep2, np.ceil(umax2/rstep2)*rstep2]
	
	def get_condition(self, c:str):
		
		if c in self.conditions:
			return self.conditions[c]
		elif c in self.global_conditions:
			return self.global_conditions[c]
		else:
			return None
	
	def render_plot(self):
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask_freq = (freq_rf_GHz == f)
		mask_pwr = (power_rf_dBm == p)
		mask = (mask_freq & mask_pwr)
		
		# Plot results
		self.ax1.cla()
		self.ax2.cla()
		
		# Check correct number of points
		mask_len = np.sum(mask)
		if len(req_bias_list) != mask_len:
			log.warning(f"Cannot display data: Mismatched number of points (freq = {f} GHz, pwr = {p} dBm, mask: {mask_len}, bias: {len(self.req_bias_list)})")
			self.fig1.canvas.draw_idle()
			return
		
		
		self.ax1.plot(requested_Idc_mA[mask], self.ce2[mask], linestyle=':', marker='o', markersize=2, color=(0.6, 0, 0.7))
		self.ax2.plot(requested_Idc_mA[mask], self.ce3[mask], linestyle=':', marker='o', markersize=2, color=(0.45, 0.05, 0.1))
		
		self.ax1.set_title(f"f-fund = {f} GHz, f-harm2 = {rd(2*f)} GHz, p = {p} dBm")
		self.ax1.set_xlabel("Requested DC Bias (mA)")
		self.ax1.set_ylabel("2nd Harm. Conversion Efficiency (%)")
		self.ax1.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax1.set_ylim(self.ylims1)
		
		self.ax2.set_title(f"f-fund = {f} GHz, f-harm3 = {rd(3*f)} GHz, p = {p} dBm")
		self.ax2.set_xlabel("Requested DC Bias (mA)")
		self.ax2.set_ylabel("3rd Harm. Conversion Efficiency (%)")
		self.ax2.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax2.set_ylim(self.ylims2)
		
		self.fig1.tight_layout()
		self.fig2.tight_layout()
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class IVPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01, 'rounding_step_x1b':0.05}
		self.global_conditions = global_conditions
		
		# Create figure
		self.fig1, ax_arr1 = plt.subplots(2, 1)
		self.fig2, ax_arr2 = plt.subplots(2, 1)
		self.ax1t = ax_arr1[0]
		self.ax1b = ax_arr1[1]
		self.ax2t = ax_arr2[0]
		self.ax2b = ax_arr2[1]
		
		self.manual_init()
		
		self.render_plot()
		
		# Create widgets
		self.fig1c = FigureCanvas(self.fig1)
		self.toolbar1 = NavigationToolbar2QT(self.fig1c, self)
		self.fig2c = FigureCanvas(self.fig2)
		self.toolbar2 = NavigationToolbar2QT(self.fig2c, self)
		
		# Add widgets to parent-widget and set layout
		self.grid = QtWidgets.QGridLayout()
		self.grid.addWidget(self.toolbar1, 0, 0)
		self.grid.addWidget(self.fig1c, 1, 0)
		self.grid.addWidget(self.toolbar2, 0, 1)
		self.grid.addWidget(self.fig2c, 1, 1)
		self.setLayout(self.grid)
		
	def manual_init(self):
		
		# Estimate system Z
		expected_Z = MFLI_V_offset_V[1]/(requested_Idc_mA[1]/1e3) #TODO: Do something more general than index 1
		system_Z = MFLI_V_offset_V/(Idc_mA/1e3)
		self.extra_z = system_Z - expected_Z
		
		# Get autoscale choices
		umax1 = np.max(Idc_mA)
		umin1 = np.min(Idc_mA)
		umax2 = np.max(self.extra_z)
		umin2 = np.min(self.extra_z)
		
		rstep1 = self.get_condition('rounding_step1')
		rstep2 = self.get_condition('rounding_step2')
		if rstep1 is None:
			rstep1 = 10
		if rstep2 is None:
			rstep2 = 10
		
		self.ylims1 = [np.floor(umin1/rstep1)*rstep1, np.ceil(umax1/rstep1)*rstep1]
		self.ylims2 = [np.floor(umin2/rstep2)*rstep2, np.ceil(umax2/rstep2)*rstep2]
		self.xlims2b = self.ylims1
		self.xlims1b = get_graph_lims(MFLI_V_offset_V, self.get_condition('rounding_step_x1b'))
	
	def get_condition(self, c:str):
		
		if c in self.conditions:
			return self.conditions[c]
		elif c in self.global_conditions:
			return self.global_conditions[c]
		else:
			return None
	
	def render_plot(self):
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask_freq = (freq_rf_GHz == f)
		mask_pwr = (power_rf_dBm == p)
		mask = (mask_freq & mask_pwr)
		
		# Plot results
		self.ax1t.cla()
		self.ax1b.cla()
		self.ax2t.cla()
		self.ax2b.cla()
		
		# Check correct number of points
		mask_len = np.sum(mask)
		if len(req_bias_list) != mask_len:
			log.warning(f"Cannot display data: Mismatched number of points (freq = {f} GHz, pwr = {p} dBm, mask: {mask_len}, bias: {len(self.req_bias_list)})")
			self.fig1.canvas.draw_idle()
			return
		
		self.ax1t.plot(requested_Idc_mA[mask], Idc_mA[mask], linestyle=':', marker='o', markersize=4, color=(0.6, 0, 0.7))
		self.ax1b.plot(MFLI_V_offset_V[mask], Idc_mA[mask], linestyle=':', marker='s', markersize=4, color=(0.2, 0, 0.8))
		
		self.ax2t.plot(requested_Idc_mA[mask], self.extra_z[mask], linestyle=':', marker='o', markersize=4, color=(0.45, 0.5, 0.1))
		self.ax2b.plot(Idc_mA[mask], self.extra_z[mask], linestyle=':', marker='o', markersize=4, color=(0, 0.5, 0.8))
		
		self.ax1t.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax1t.set_xlabel("Requested DC Bias (mA)")
		self.ax1t.set_ylabel("Measured DC Bias (mA)")
		self.ax1t.grid(True)
		
			
		self.ax1b.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax1b.set_xlabel("Applied DC Voltage (V)")
		self.ax1b.set_ylabel("Measured DC Bias (mA)")
		self.ax1b.grid(True)
			
		
		self.ax2t.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax2t.set_xlabel("Requested DC Bias (mA)")
		self.ax2t.set_ylabel("Additional Impedance (Ohms)")
		self.ax2t.grid(True)
		
		self.ax2b.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax2b.set_xlabel("Measured DC Bias (mA)")
		self.ax2b.set_ylabel("Additional Impedance (Ohms)")
		self.ax2b.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax1t.set_ylim(self.ylims1)
			self.ax1b.set_ylim(self.ylims1)
			
			self.ax1b.set_xlim(self.xlims1b)
			
			self.ax2t.set_ylim(self.ylims2)
			self.ax2b.set_ylim(self.ylims2)
			
			self.ax2b.set_xlim(self.xlims2b)
		
		self.fig1.tight_layout()
		self.fig2.tight_layout()
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class SParamSPDPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = {'rounding_step':10}
		self.global_conditions = global_conditions
		
		self.manual_init()
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1, figsize=(12, 7))
		self.fig1.subplots_adjust(left=0.065, bottom=0.065, top=0.95, right=0.8)
		
		self.render_plot()
		
		# Create widgets
		self.fig1c = FigureCanvas(self.fig1)
		self.toolbar = NavigationToolbar2QT(self.fig1c, self)
		
		# Add widgets to parent-widget and set layout
		self.grid = QtWidgets.QGridLayout()
		self.grid.addWidget(self.toolbar, 0, 0)
		self.grid.addWidget(self.fig1c, 1, 0)
		self.setLayout(self.grid)
	
	def manual_init(self):
		pass
		# # Get autoscale choices
		# umax = np.max([np.max(rf1), np.max(rf2), np.max(rf3)])
		# umin = np.min([np.min(rf1), np.min(rf2), np.min(rf3)])
		
		# rstep = self.get_condition('rounding_step')
		# if rstep is None:
		# 	rstep = 10
		
		# self.ylims = [np.floor(umin/rstep)*rstep, np.ceil(umax/rstep)*rstep]
	
	def get_condition(self, c:str):
		
		if c in self.conditions:
			return self.conditions[c]
		elif c in self.global_conditions:
			return self.global_conditions[c]
		else:
			return None
	
	def render_plot(self):
		# f = self.get_condition('sel_freq_GHz')
		# p = self.get_condition('sel_power_dBm')
		
		# # Filter relevant data
		# mask_freq = (freq_rf_GHz == f)
		# mask_pwr = (power_rf_dBm == p)
		# mask = (mask_freq & mask_pwr)
		
		# Plot results
		self.ax1.cla()
		
		# Check correct number of points
		# mask_len = np.sum(mask)
		# if len(self.req_bias_list) != mask_len:
		# 	log.warning(f"Cannot display data: Mismatched number of points (freq = {f} GHz, pwr = {p} dBm, mask: {mask_len}, bias: {len(self.req_bias_list)})")
		# 	self.fig1.canvas.draw_idle()
		# 	return
		
		
		self.ax1.plot(S_freq_GHz, S11_dB, linestyle=':', marker='o', markersize=1, color=(0.7, 0, 0))
		self.ax1.plot(S_freq_GHz, S21_dB, linestyle=':', marker='o', markersize=1, color=(0, 0.7, 0))
		# self.ax1.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax1.set_xlabel("Frequency (GHz)")
		self.ax1.set_ylabel("Power (dBm)")
		self.ax1.legend(["S11", "S21"])
		self.ax1.grid(True)
		
		# if self.get_condition('fix_scale'):
		# 	self.ax1.set_ylim(self.ylims)
		
		self.fig1.tight_layout()
		
		self.fig1.canvas.draw_idle()


class HarmGenBiasDomainPlotWidget(TabPlotWidget):
	
	def __init__(self, f_GHz:list, p_dBm:list, ridc:list, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = {'rounding_step':10}
		self.global_conditions = global_conditions
		
		# Save primary data
		self.freq_rf_GHz = f_GHz
		self.power_rf_dBm = p_dBm
		self.requested_Idc_mA = ridc
		
		self.manual_init()
		
		# Save lists of freq/power options
		self.freq_list = np.unique(self.freq_rf_GHz)
		self.pwr_list = np.unique(self.power_rf_dBm)
		self.req_bias_list = np.unique(self.requested_Idc_mA)
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1, figsize=(12, 7))
		self.fig1.subplots_adjust(left=0.065, bottom=0.065, top=0.95, right=0.8)
		
		self.render_plot()
		
		# Create widgets
		self.fig1c = FigureCanvas(self.fig1)
		self.toolbar = NavigationToolbar2QT(self.fig1c, self)
		
		# Add widgets to parent-widget and set layout
		self.grid = QtWidgets.QGridLayout()
		self.grid.addWidget(self.toolbar, 0, 0)
		self.grid.addWidget(self.fig1c, 1, 0)
		self.setLayout(self.grid)
	
	def manual_init(self):
		
		# Get autoscale choices
		umax = np.max([np.max(rf1), np.max(rf2), np.max(rf3)])
		umin = np.min([np.min(rf1), np.min(rf2), np.min(rf3)])
		
		rstep = self.get_condition('rounding_step')
		if rstep is None:
			rstep = 10
		
		self.ylims = [np.floor(umin/rstep)*rstep, np.ceil(umax/rstep)*rstep]
	
	def get_condition(self, c:str):
		
		if c in self.conditions:
			return self.conditions[c]
		elif c in self.global_conditions:
			return self.global_conditions[c]
		else:
			return None
	
	def render_plot(self):
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask_freq = (freq_rf_GHz == f)
		mask_pwr = (power_rf_dBm == p)
		mask = (mask_freq & mask_pwr)
		
		# Plot results
		self.ax1.cla()
		
		# Check correct number of points
		mask_len = np.sum(mask)
		if len(self.req_bias_list) != mask_len:
			log.warning(f"Cannot display data: Mismatched number of points (freq = {f} GHz, pwr = {p} dBm, mask: {mask_len}, bias: {len(self.req_bias_list)})")
			self.fig1.canvas.draw_idle()
			return
		
		
		self.ax1.plot(Idc_mA[mask], rf1[mask], linestyle=':', marker='o', markersize=1.5, color=(0, 0.7, 0))
		self.ax1.plot(Idc_mA[mask], rf2[mask], linestyle=':', marker='o', markersize=1.5, color=(0, 0, 0.7))
		self.ax1.plot(Idc_mA[mask], rf3[mask], linestyle=':', marker='o', markersize=1.5, color=(0.7, 0, 0))
		self.ax1.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax1.set_xlabel("DC Bias (mA)")
		self.ax1.set_ylabel("Power (dBm)")
		self.ax1.legend(["Fundamental", "2nd Harm.", "3rd Harm."])
		self.ax1.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax1.set_ylim(self.ylims)
		
		self.fig1.tight_layout()
		
		self.fig1.canvas.draw_idle()

class BiasDomainTabWidget(QTabWidget):
	
	def __init__(self, global_conditions:dict, main_window):
		super().__init__()
		
		self.gcond = global_conditions
		self.main_window = main_window
		self.object_list = []
		self._is_active = False
		
		#------------ Harmonics widget
		
		self.object_list.append(HarmGenBiasDomainPlotWidget(freq_rf_GHz, power_rf_dBm, requested_Idc_mA, global_conditions=self.gcond))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "Harmonic Generation")
		
		#------------ CE widget
		
		self.object_list.append(CE23BiasDomainPlotWidget(global_conditions=self.gcond))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "Efficiency")
		
		#------------ Harmonics widget
		
		self.object_list.append(IVPlotWidget(global_conditions=self.gcond))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "Bias Current")
		
		self.currentChanged.connect(self.update_active_tab)
		
	def set_active(self, b:bool):
		self._is_active = b
		self.update_active_tab()
	
	def update_active_tab(self):
		
		# Set all objects to inactive
		for obj in self.object_list:
			obj.set_active(False)
		
		# Set only the active widget to active
		if self._is_active:
			self.object_list[self.currentIndex()].set_active(True)

class FrequencyDomainTabWidget(QTabWidget):
	
	def __init__(self, global_conditions:dict, main_window):
		super().__init__()
		
		self._is_active = False
		
		self.gcond = global_conditions
		self.main_window = main_window
		self.object_list = []
		
		#------------ Harmonics widget
		
		self.object_list.append(HarmGenFreqDomainPlotWidget(global_conditions=self.gcond))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "Harmonic Generation")
		
		#------------ CE widget
		
		self.object_list.append(CE23FreqDomainPlotWidget(global_conditions=self.gcond))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "Efficiency")
		
		self.currentChanged.connect(self.update_active_tab)
	
	def set_active(self, b:bool):
		self._is_active = b
		self.update_active_tab()
	
	def update_active_tab(self):
		
		# Set all objects to inactive
		for obj in self.object_list:
			obj.set_active(False)
		
		# Set only the active widget to active
		if self._is_active:
			self.object_list[self.currentIndex()].set_active(True)
	
class SPDTabWidget(QTabWidget):
	''' S-Parameter Domain Tab Widget'''
	
	def __init__(self, global_conditions:dict, main_window):
		super().__init__()
		
		self._is_active = False
		
		self.gcond = global_conditions
		self.main_window = main_window
		self.object_list = []
		
		#------------ Harmonics widget
		
		self.object_list.append(SParamSPDPlotWidget(global_conditions=self.gcond))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "S-Parameters")
		
		# #------------ CE widget
		
		# self.object_list.append(CE23FreqDomainPlotWidget(global_conditions=self.gcond))
		# self.main_window.gcond_subscribers.append(self.object_list[-1])
		# self.addTab(self.object_list[-1], "Efficiency")
		
		# self.currentChanged.connect(self.update_active_tab)
	
	def set_active(self, b:bool):
		self._is_active = b
		self.update_active_tab()
	
	def update_active_tab(self):
		
		# Set all objects to inactive
		for obj in self.object_list:
			obj.set_active(False)
		
		# Set only the active widget to active
		if self._is_active:
			self.object_list[self.currentIndex()].set_active(True)
	
class HGA1Window(QtWidgets.QMainWindow):

	def __init__(self, log, freqs, powers, app, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Save local variables
		self.log = log
		self.app = app
		
		# Master Data
		self.freq_list = freqs
		self.pwr_list = powers
		
		self.unique_freq_list = np.unique(freqs)
		self.unique_pwr_list = np.unique(powers)
		
		# Initialize global conditions
		self.gcond = {'sel_freq_GHz': self.freq_list[len(self.freq_list)//2], 'sel_power_dBm': self.pwr_list[len(self.pwr_list)//2], 'sel_bias_mA': req_bias_list[len(req_bias_list)//2], 'fix_scale':False, 'freqxaxis_isfund':False}
		self.gcond_subscribers = []
		
		# Basic setup
		self.setWindowTitle("PyQt Cryo Analyzer")
		self.grid = QtWidgets.QGridLayout() # Create the primary layout
		self.add_menu()
		
		# Create tab widget
		self.tab_widget_widgets = []
		self.tab_widget = QtWidgets.QTabWidget()
		self.tab_widget.currentChanged.connect(self.update_active_tab)
		self.make_tabs() # Make tabs
		
		# Make sliders
		self.slider_box = QtWidgets.QWidget()
		self.populate_slider_box()
		
		# Active main sweep file label
		self.active_file_label = QLabel()
		self.active_file_label.setText(f"Active Main Sweep File: {analysis_file}")
		self.active_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
		
		# Active s-param file label
		self.active_spfile_label = QLabel()
		self.active_spfile_label.setText(f"Active S-Parameter File: {sp_analysis_file}")
		self.active_spfile_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
		
		# Place each widget
		self.grid.addWidget(self.tab_widget, 0, 0)
		self.grid.addWidget(self.slider_box, 0, 1)
		self.grid.addWidget(self.active_file_label, 1, 0, 1, 2)
		self.grid.addWidget(self.active_spfile_label, 2, 0, 1, 2)
		
		# Set the central widget
		central_widget = QtWidgets.QWidget()
		central_widget.setLayout(self.grid)
		self.setCentralWidget(central_widget)
		
		
		
		self.show()
	
	def close(self):
		''' This will be called before the window closes. Save any stuff etc here.'''
		
		pass
	
	def set_gcond(self, key, value):
		
		self.gcond[key] = value
		
		for sub in self.gcond_subscribers:
			sub.global_conditions[key] = value
	
	def plot_all(self):
		
		for sub in self.gcond_subscribers:
			sub.update_plot()
	
	def update_freq(self, x):
		try:
			new_freq = self.unique_freq_list[x]
			self.set_gcond('sel_freq_GHz', new_freq)
			self.plot_all()
		except Exception as e:
			log.warning(f"Index out of bounds! ({e})")
		
		self.freq_slider_vallabel.setText(f"{new_freq} GHz")
	
	def update_active_tab(self):
		
		# Set all objects to inactive
		for obj in self.tab_widget_widgets:
			obj.set_active(False)
		
		self.tab_widget_widgets[self.tab_widget.currentIndex()].set_active(True)
	
	def update_pwr(self, x):
		try:
			new_pwr = self.unique_pwr_list[x]
			self.set_gcond('sel_power_dBm', new_pwr)
			self.plot_all()
		except Exception as e:
			log.warning(f"Index out of bounds! ({e})")
		
		self.pwr_slider_vallabel.setText(f"{new_pwr} dBm")
		
	def update_bias(self, x):
		try:
			new_b = req_bias_list[x]
			self.set_gcond('sel_bias_mA', new_b)
			self.plot_all()
		except Exception as e:
			log.warning(f"Index out of bounds! ({e})")
		
		self.bias_slider_vallabel.setText(f"{new_b} dBm")
	
	def populate_slider_box(self):
		
		
		ng = QtWidgets.QGridLayout()
		
		
		self.freq_slider_hdrlabel = QtWidgets.QLabel()
		self.freq_slider_hdrlabel.setText("Frequency (GHz)")
		
		self.freq_slider_vallabel = QtWidgets.QLabel()
		self.freq_slider_vallabel.setText("VOID (GHz)")
		
		self.freq_slider = QtWidgets.QSlider(Qt.Orientation.Vertical)
		self.freq_slider.valueChanged.connect(self.update_freq)
		self.freq_slider.setSingleStep(1)
		self.freq_slider.setMinimum(0)
		self.freq_slider.setMaximum(len(np.unique(self.freq_list))-1)
		self.freq_slider.setTickInterval(1)
		self.freq_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksLeft)
		self.freq_slider.setSliderPosition(0)
		
		self.pwr_slider_hdrlabel = QtWidgets.QLabel()
		self.pwr_slider_hdrlabel.setText("Power (dBm)")
		
		self.pwr_slider_vallabel = QtWidgets.QLabel()
		self.pwr_slider_vallabel.setText("VOID (dBm)")
		
		self.pwr_slider = QtWidgets.QSlider(Qt.Orientation.Vertical)
		self.pwr_slider.valueChanged.connect(self.update_pwr)
		self.pwr_slider.setSingleStep(1)
		self.pwr_slider.setMinimum(0)
		self.pwr_slider.setMaximum(len(np.unique(self.pwr_list))-1)
		self.pwr_slider.setTickInterval(1)
		self.pwr_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksLeft)
		self.pwr_slider.setSliderPosition(0)
		
		self.bias_slider_hdrlabel = QtWidgets.QLabel()
		self.bias_slider_hdrlabel.setText("Bias (mA)")
		
		self.bias_slider_vallabel = QtWidgets.QLabel()
		self.bias_slider_vallabel.setText("VOID (mA)")
		
		self.bias_slider = QtWidgets.QSlider(Qt.Orientation.Vertical)
		self.bias_slider.valueChanged.connect(self.update_bias)
		self.bias_slider.setSingleStep(1)
		self.bias_slider.setMinimum(0)
		self.bias_slider.setMaximum(len(req_bias_list)-1)
		self.bias_slider.setTickInterval(1)
		self.bias_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksLeft)
		self.bias_slider.setSliderPosition(0)
		
		ng.addWidget(self.freq_slider_hdrlabel, 0, 0)
		ng.addWidget(self.freq_slider, 1, 0, alignment=Qt.AlignmentFlag.AlignHCenter)
		ng.addWidget(self.freq_slider_vallabel, 2, 0)
		
		ng.addWidget(self.pwr_slider_hdrlabel, 0, 1)
		ng.addWidget(self.pwr_slider, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
		ng.addWidget(self.pwr_slider_vallabel, 2, 1)
		
		ng.addWidget(self.bias_slider_hdrlabel, 0, 2)
		ng.addWidget(self.bias_slider, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
		ng.addWidget(self.bias_slider_vallabel, 2, 2)
		
		self.slider_box.setLayout(ng)
		
		# Trigger all slider callbacks
		self.update_bias(self.bias_slider.value())
		self.update_freq(self.freq_slider.value())
		self.update_pwr(self.pwr_slider.value())
	
	def make_tabs(self):
		
		self.tab_widget_widgets.append(BiasDomainTabWidget(self.gcond, self))
		self.tab_widget.addTab(self.tab_widget_widgets[-1], "Main Sweep - Bias Domain")
		
		self.tab_widget_widgets.append(FrequencyDomainTabWidget(self.gcond, self))
		self.tab_widget.addTab(self.tab_widget_widgets[-1], "Main Sweep - Frequency Domain")
		
		self.tab_widget_widgets.append(SPDTabWidget(self.gcond, self))
		self.tab_widget.addTab(self.tab_widget_widgets[-1], "S-Parameters")
	
	def add_menu(self):
		''' Adds menus to the window'''
		
		self.bar = self.menuBar()
		
		# File Menu --------------------------------------
		
		self.file_menu = self.bar.addMenu("File")
		self.file_menu.triggered[QAction].connect(self._process_file_menu)
		
		self.save_graph_act = QAction("Save Graph", self)
		self.save_graph_act.setShortcut("Ctrl+Shift+G")
		self.file_menu.addAction(self.save_graph_act)
		
		self.close_window_act = QAction("Close Window", self)
		self.close_window_act.setShortcut("Ctrl+W")
		self.file_menu.addAction(self.close_window_act)
		
		# Graph Menu --------------------------------------
		
		self.graph_menu = self.bar.addMenu("Graph")
		self.graph_menu.triggered[QAction].connect(self._process_graph_menu)
		
		self.fix_scales_act = QAction("Fix Scales", self, checkable=True)
		self.fix_scales_act.setShortcut("Ctrl+F")
		self.fix_scales_act.setChecked(True)
		self.set_gcond('fix_scale', self.fix_scales_act.isChecked())
		self.graph_menu.addAction(self.fix_scales_act)
		
			# Graph Menu: Freq-axis sub menu -------------
		
		self.freqxaxis_graph_menu = self.graph_menu.addMenu("Frequency X-Axis")
		
		self.freqxaxis_group = QActionGroup(self)
		
		self.freqxaxis_fund_act = QAction("Show Fundamental", self, checkable=True)
		self.freqxaxis_harm_act = QAction("Show Harmonics", self, checkable=True)
		self.freqxaxis_harm_act.setChecked(True)
		self.freqxaxis_harm_act.setShortcut("Shift+X")
		self.freqxaxis_fund_act.setShortcut("Ctrl+Shift+X")
		self.set_gcond('freqxaxis_isfund', self.freqxaxis_fund_act.isChecked())
		self.freqxaxis_graph_menu.addAction(self.freqxaxis_fund_act)
		self.freqxaxis_graph_menu.addAction(self.freqxaxis_harm_act)
		self.freqxaxis_group.addAction(self.freqxaxis_fund_act)
		self.freqxaxis_group.addAction(self.freqxaxis_harm_act)
		
	def _process_file_menu(self, q):
		
		if q.text() == "Save Graph":
			self.log.warning("TODO: Implement save graph")
		if q.text() == "Close Window":
			self.close()
			sys.exit(0)
	
	def _process_graph_menu(self, q):
		
		if q.text() == "Fix Scales":
			self.set_gcond('fix_scale', self.fix_scales_act.isChecked())
			self.plot_all()
		elif q.text() == "Show Fundamental" or q.text() == "Show Harmonics":
			self.set_gcond('freqxaxis_isfund', self.freqxaxis_fund_act.isChecked())
			self.plot_all()
			
app = QtWidgets.QApplication(sys.argv)

w = HGA1Window(log, freq_rf_GHz, power_rf_dBm, app)
app.exec()
