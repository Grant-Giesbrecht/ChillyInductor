import sys
import matplotlib
matplotlib.use('qtagg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from pylogfile.base import *

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

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
# import qdarktheme

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

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm'])

if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

# filename = "RP22B_MP3_t1_31July2024_R4C4T1_r1_autosave.hdf"
# filename = "RP22B_MP3_t1_1Aug2024_R4C4T1_r1.hdf"
filename = "RP22B_MP3_t2_8Aug2024_R4C4T1_r1.hdf"

analysis_file = os.path.join(datapath, filename)

log = LogPile()

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

##--------------------------------------------
# Create GUI

class MplCanvas(FigureCanvasQTAgg):

	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = fig.add_subplot(111)
		super(MplCanvas, self).__init__(fig)

class TabPlot():
	def __init__(self, fig, toolbar):
		self.fig = fig
		self.toolbar = toolbar

class CE23BiasDomainPlotWidget(QtWidgets.QWidget):
	
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
		
		
		self.plot_data()
		
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
	
	def plot_data(self):
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
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class IVPlotWidget(QtWidgets.QWidget):
	
	def __init__(self, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01}
		self.global_conditions = global_conditions
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1)
		self.fig2, self.ax2 = plt.subplots(1, 1)
		
		self.manual_init()
		
		self.plot_data()
		
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
	
	def get_condition(self, c:str):
		
		if c in self.conditions:
			return self.conditions[c]
		elif c in self.global_conditions:
			return self.global_conditions[c]
		else:
			return None
	
	def plot_data(self):
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
		
		
		self.ax1.plot(requested_Idc_mA[mask], Idc_mA[mask], linestyle=':', marker='o', markersize=2, color=(0.6, 0, 0.7))
		self.ax2.plot(requested_Idc_mA[mask], self.extra_z[mask], linestyle=':', marker='o', markersize=2, color=(0.45, 0.05, 0.1))
		
		self.ax1.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax1.set_xlabel("Requested DC Bias (mA)")
		self.ax1.set_ylabel("Measured DC Bias (mA)")
		self.ax1.grid(True)
		if self.get_condition('fix_scale'):
			self.ax1.set_ylim(self.ylims1)
		
		self.ax2.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax2.set_xlabel("Requested DC Bias (mA)")
		self.ax2.set_ylabel("Additional Impedance (Ohms)")
		self.ax2.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax2.set_ylim(self.ylims2)
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class HarmGenPlotWidget(QtWidgets.QWidget):
	
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
		
		self.plot_data()
		
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
	
	def plot_data(self):
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
		
		self.fig1.canvas.draw_idle()

class BiasDomainTabWidget(QtWidgets.QTabWidget):
	
	def __init__(self, global_conditions:dict, main_window):
		super().__init__()
		
		self.gcond = global_conditions
		self.main_window = main_window
		
		#------------ Harmonics widget
		
		self.hgwidget = HarmGenPlotWidget(freq_rf_GHz, power_rf_dBm, requested_Idc_mA, global_conditions=self.gcond)
		self.main_window.gcond_subscribers.append(self.hgwidget)
		
		# Add to tabs object and handle list
		ntp = TabPlot(self.hgwidget.fig1, self.hgwidget.toolbar)
		self.addTab(self.hgwidget, "Harmonic Generation")
		
		#------------ CE widget
		
		self.cebdwidget = CE23BiasDomainPlotWidget(global_conditions=self.gcond)
		self.main_window.gcond_subscribers.append(self.cebdwidget)
		
		# Add to tabs object and handle list

		self.addTab(self.cebdwidget, "Efficiency")
		
		#------------ Harmonics widget
		
		self.ivwidget = IVPlotWidget(global_conditions=self.gcond)
		self.main_window.gcond_subscribers.append(self.ivwidget)
		
		# Add to tabs object and handle list
		self.addTab(self.ivwidget, "Bias Current")
		
class HGA1Window(QtWidgets.QMainWindow):

	def __init__(self, log, freqs, powers, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Save local variables
		self.log = log
		
		# Master Data
		self.freq_list = freqs
		self.pwr_list = powers
		
		self.unique_freq_list = np.unique(freqs)
		self.unique_pwr_list = np.unique(powers)
		
		# Initialize global conditions
		self.gcond = {'sel_freq_GHz': self.freq_list[len(self.freq_list)//2], 'sel_power_dBm': self.pwr_list[len(self.pwr_list)//2]}
		self.gcond_subscribers = []
		
		# Basic setup
		self.setWindowTitle("HGA1 Window")
		self.grid = QtWidgets.QGridLayout() # Create the primary layout
		self.add_menu()
		
		# Create tab widget
		self.tab_widget = QtWidgets.QTabWidget()
		self.tab_widget.currentChanged.connect(self._tab_changed)
		self.make_tabs() # Make tabs
		
		# Make sliders
		self.slider_box = QtWidgets.QWidget()
		self.populate_slider_box()
		
		# Place each widget
		self.grid.addWidget(self.tab_widget, 0, 0)
		self.grid.addWidget(self.slider_box, 0, 1)
		
		# Set the central widget
		central_widget = QtWidgets.QWidget()
		central_widget.setLayout(self.grid)
		self.setCentralWidget(central_widget)
		
		self.show()
	
	def set_gcond(self, key, value):
		
		self.gcond[key] = value
		
		for sub in self.gcond_subscribers:
			sub.global_conditions[key] = value
	
	def plot_all(self):
		
		for sub in self.gcond_subscribers:
			sub.plot_data()
	
	def update_freq(self, x):
		try:
			new_freq = self.unique_freq_list[x]
			self.set_gcond('sel_freq_GHz', new_freq)
			self.plot_all()
		except Exception as e:
			log.warning(f"Index out of bounds! ({e})")
		
		self.freq_slider_vallabel.setText(f"{new_freq} GHz")
	
	def _tab_changed(self, x):
		
		print(self.tab_widget.currentWidget())
		print(type(self.tab_widget.currentWidget()))
	
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
	
	def make_tabs(self):
		
		self.bias_domain_widget = BiasDomainTabWidget(self.gcond, self)
		self.tab_widget.addTab(self.bias_domain_widget, "Bias Domain")
		
		# #------------ Harmonics widget
		
		# self.hgwidget = HarmGenPlotWidget(freq_rf_GHz, power_rf_dBm, requested_Idc_mA, global_conditions=self.gcond)
		# self.gcond_subscribers.append(self.hgwidget)
		
		# # Add to tabs object and handle list
		# ntp = TabPlot(self.hgwidget.fig1, self.hgwidget.toolbar)
		# self.tab_handles.append(ntp)
		# self.tab_widget.addTab(self.hgwidget, "Harmonic Generation")
		
		# #------------ CE widget
		
		# self.cebdwidget = CE23BiasDomainPlotWidget(global_conditions=self.gcond)
		# self.gcond_subscribers.append(self.cebdwidget)
		
		# # Add to tabs object and handle list
		# ntp1 = TabPlot(self.cebdwidget.fig1, self.cebdwidget.toolbar1)
		# ntp2 = TabPlot(self.cebdwidget.fig2, self.cebdwidget.toolbar2) #TODO I don't need these
		# self.tab_handles.append(ntp1)
		# self.tab_handles.append(ntp2)
		# self.tab_widget.addTab(self.cebdwidget, "Efficiency")
		
		# #------------ Harmonics widget
		
		# self.ivwidget = IVPlotWidget(global_conditions=self.gcond)
		# self.gcond_subscribers.append(self.ivwidget)
		
		# # Add to tabs object and handle list
		# ntp1 = TabPlot(self.ivwidget.fig1, self.ivwidget.toolbar1)
		# ntp2 = TabPlot(self.ivwidget.fig2, self.ivwidget.toolbar2) #TODO I don't need these
		# self.tab_handles.append(ntp1)
		# self.tab_handles.append(ntp2)
		# self.tab_widget.addTab(self.ivwidget, "Bias Current")
	
	def add_menu(self):
		''' Adds menus to the window'''
		
		self.bar = self.menuBar()
		
		# File Menu --------------------------------------
		
		self.file_menu = self.bar.addMenu("File")
		self.file_menu.triggered[QAction].connect(self._process_file_menu)
		
		self.save_graph_act = QAction("Save Graph", self)
		self.save_graph_act.setShortcut("Ctrl+Shift+G")
		self.file_menu.addAction(self.save_graph_act)
		
		# Graph Menu --------------------------------------
		
		self.graph_menu = self.bar.addMenu("Graph")
		self.graph_menu.triggered[QAction].connect(self._process_graph_menu)
		
		self.fix_scales_act = QAction("Fix Scales", self, checkable=True)
		self.fix_scales_act.setShortcut("Ctrl+F")
		self.fix_scales_act.setChecked(True)
		self.set_gcond('fix_scale', self.fix_scales_act.isChecked())
		self.graph_menu.addAction(self.fix_scales_act)
		
	def _process_file_menu(self, q):
		
		if q.text() == "Save Graph":
			self.log.warning("TODO: Implement save graph")
	
	def _process_graph_menu(self, q):
		
		if q.text() == "Fix Scales":
			self.set_gcond('fix_scale', self.fix_scales_act.isChecked())

app = QtWidgets.QApplication(sys.argv)

w = HGA1Window(log, freq_rf_GHz, power_rf_dBm)
app.exec()
