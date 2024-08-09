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

# TODO:
# 1. Init graph properly (says void and stuff)
# 2. Add labels to values on freq and power sliders
# 3. Add save log, save graph, zoom controls, etc.
# 4. Frequency-domain plots!
# 5. Datatips
# 6. Buttons to zoom to max efficiency

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
filename = "RP22B_MP3_t1_1Aug2024_R4C4T1_r1.hdf"

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
	
	def __init__(self, conditions:dict, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = conditions
		self.global_conditions = global_conditions
		
		# Calculate total power and CE
		
		
		self.total_power = rf1W + rf2W + rf3W
		self.ce2 = rf2W/self.total_power*100
		self.ce3 = rf3W/self.total_power*100
		
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
		
		self.ax1.set_title(f"f-fund = {f} GHz, f-harm2 = {2*f} GHz, p = {p} dBm")
		self.ax1.set_xlabel("Requested DC Bias (mA)")
		self.ax1.set_ylabel("2nd Harm. Conversion Efficiency (%)")
		self.ax1.grid(True)
		
		self.ax2.set_title(f"f-fund = {f} GHz, f-harm3 = {3*f} GHz, p = {p} dBm")
		self.ax2.set_xlabel("Requested DC Bias (mA)")
		self.ax2.set_ylabel("3rd Harm. Conversion Efficiency (%)")
		self.ax2.grid(True)
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class IVPlotWidget(QtWidgets.QWidget):
	
	def __init__(self, conditions:dict, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = conditions
		self.global_conditions = global_conditions
		
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
		
		self.ax2.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax2.set_xlabel("Requested DC Bias (mA)")
		self.ax2.set_ylabel("Additional Impedance (Ohms)")
		self.ax2.grid(True)
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class HarmGenPlotWidget(QtWidgets.QWidget):
	
	def __init__(self, f_GHz:list, p_dBm:list, ridc:list, conditions:dict, global_conditions:dict={}):
		super().__init__()
		
		# Conditions dictionaries
		self.conditions = conditions
		self.global_conditions = global_conditions
		
		# Save primary data
		self.freq_rf_GHz = f_GHz
		self.power_rf_dBm = p_dBm
		self.requested_Idc_mA = ridc
		
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
		self.ax1.set_ylim([-100, -20])
		
		self.fig1.canvas.draw_idle()
		
class HGA1Window(QtWidgets.QMainWindow):

	def __init__(self, log, freqs, powers, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Save local variables
		self.log = log
		self.tab_handles = []
		
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
		self.hgwidget = None
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
	
	def update_pwr(self, x):
		try:
			new_pwr = self.unique_pwr_list[x]
			self.set_gcond('sel_power_dBm', new_pwr)
			self.plot_all()
		except Exception as e:
			log.warning(f"Index out of bounds! ({e})")
		
		self.pwr_slider_vallabel.setText(f"{new_pwr} dBm")
	
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
		
		ng.addWidget(self.freq_slider_hdrlabel, 0, 0)
		ng.addWidget(self.freq_slider, 1, 0, alignment=Qt.AlignmentFlag.AlignHCenter)
		ng.addWidget(self.freq_slider_vallabel, 2, 0)
		
		ng.addWidget(self.pwr_slider_hdrlabel, 0, 1)
		ng.addWidget(self.pwr_slider, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
		ng.addWidget(self.pwr_slider_vallabel, 2, 1)
		self.slider_box.setLayout(ng)
	
	def make_tabs(self):
		
		#------------ Harmonics widget
		
		self.hgwidget = HarmGenPlotWidget(freq_rf_GHz, power_rf_dBm, requested_Idc_mA, conditions={}, global_conditions=self.gcond)
		self.gcond_subscribers.append(self.hgwidget)
		
		# Add to tabs object and handle list
		ntp = TabPlot(self.hgwidget.fig1, self.hgwidget.toolbar)
		self.tab_handles.append(ntp)
		self.tab_widget.addTab(self.hgwidget, "Harmonic Generation")
		
		#------------ CE widget
		
		self.cebdwidget = CE23BiasDomainPlotWidget(conditions={}, global_conditions=self.gcond)
		self.gcond_subscribers.append(self.cebdwidget)
		
		# Add to tabs object and handle list
		ntp1 = TabPlot(self.cebdwidget.fig1, self.cebdwidget.toolbar1)
		ntp2 = TabPlot(self.cebdwidget.fig2, self.cebdwidget.toolbar2) #TODO I don't need these
		self.tab_handles.append(ntp1)
		self.tab_handles.append(ntp2)
		self.tab_widget.addTab(self.cebdwidget, "Efficiency")
		
		#------------ Harmonics widget
		
		self.ivwidget = IVPlotWidget(conditions={}, global_conditions=self.gcond)
		self.gcond_subscribers.append(self.ivwidget)
		
		# Add to tabs object and handle list
		ntp1 = TabPlot(self.ivwidget.fig1, self.ivwidget.toolbar1)
		ntp2 = TabPlot(self.ivwidget.fig2, self.ivwidget.toolbar2) #TODO I don't need these
		self.tab_handles.append(ntp1)
		self.tab_handles.append(ntp2)
		self.tab_widget.addTab(self.ivwidget, "Bias Current")
	
	# def make_normal_tab(self):
	# 	''' Makes the tab for harmonic generation'''
		
	# 	# Prepare tab
	# 	ntab = QtWidgets.QWidget()
	# 	ntlayout = QtWidgets.QGridLayout()
	# 	ntab.setLayout(ntlayout)
		
	# 	# Prepare new figure and toolbar
	# 	fig_left = plt.figure()
	# 	figc_left = FigureCanvas(fig_left)
	# 	tbar_left = NavigationToolbar2QT(figc_left, self)
		
	# 	# Prepare new figure and toolbar
	# 	fig_right = plt.figure()
	# 	figc_right = FigureCanvas(fig_right)
	# 	tbar_right = NavigationToolbar2QT(figc_right, self)
		
	# 	# Add to layout
	# 	ntlayout.addWidget(tbar_left, 0, 0)
	# 	ntlayout.addWidget(figc_left, 1, 0)
	# 	ntlayout.addWidget(tbar_right, 0, 1)
	# 	ntlayout.addWidget(figc_right, 1, 1)
		
	# 	# Add to tabs object and handle list
	# 	ntpl = TabPlot(fig_left, tbar_left)
	# 	ntpr = TabPlot(fig_right, tbar_right)
	# 	self.tab_handles.append(ntpl)
	# 	self.tab_handles.append(ntpr)
	# 	self.tab_widget.addTab(ntab, "Normal")
	
	def add_menu(self):
		''' Adds menus to the window'''
		
		self.bar = self.menuBar()
		
		self.file_menu = self.bar.addMenu("File")
		self.file_menu.triggered[QAction].connect(self._process_file_menu)
		
		self.save_graph_act = QAction("Save Graph", self)
		self.save_graph_act.setShortcut("Ctrl+Shift+G")
		self.file_menu.addAction(self.save_graph_act)
		
	def _process_file_menu(self, q):
		
		if q.text() == "Save Graph":
			self.log.warning("TODO: Implement save graph")

app = QtWidgets.QApplication(sys.argv)
app.setStyle('Oxygen')
w = HGA1Window(log, freq_rf_GHz, power_rf_dBm)
app.exec()
