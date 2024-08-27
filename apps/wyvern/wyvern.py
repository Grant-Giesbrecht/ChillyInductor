import sys
import matplotlib
matplotlib.use('qtagg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from pylogfile.base import *

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction, QActionGroup, QDoubleValidator, QIcon, QFontDatabase, QFont
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QWidget, QTabWidget, QLabel, QGridLayout, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QMainWindow, QSlider, QPushButton

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
import argparse

log = LogPile()

# TODO: Important
#
# * Tests show if only the presently displayed graph is rendered, when the sliders are adjusted, the update speed is far better (no kidding).
# * Tests also show pre-allocating a mask isn't super useful. It's mostly the matplotlib rendering that slows it down.

# TODO:
# 1. Init graph properly (says void and stuff)
# 2. Add labels to values on freq and power sliders
# 3. Add save log, save graph, zoom controls, etc.
# 5. Datatips
# 6. Buttons to zoom to max efficiency
# 7. Data panel at bottom showing file stats:
#		- collection date, operator notes, file size, num points, open log in lumberjack button

# 8. Better way to load any data file

# 9. Make filename label copyable
# 10. Fix axes applies to X too
# 11. Fix axes only fits X-percentile.

# TODO: Graphs to add
# 1. Applied voltage, measured current, target current, estimated additional impedance.
# 2. Temperature vs time
#

parser = argparse.ArgumentParser()
# parser.add_argument('-h', '--help')
parser.add_argument('-s', '--subtle', help="Run without naming.", action='store_true')
cli_args = parser.parse_args()
# print(cli_args)
# print(cli_args.subtle)

def get_font(font_ttf_path):
	
	font_id = QFontDatabase.addApplicationFont(font_ttf_path)
	if font_id == -1:
		print(f"Failed to read font")
		return None
	
	return QFontDatabase.applicationFontFamilies(font_id)[0]

# chicago_ff = get_font("./assets/Chicago.ttf")
# chicago_12 = QFont(chicago_ff, 12)


class MasterData:
	''' Class to represent all the data analyzed by the application'''
	
	def __init__(self):
		self.clear_all()
		self.import_hdf()
		self.import_sparam()
		
	def clear_all(self):
		
		# Names of files loaded
		self.current_sweep_file = ""
		self.current_sparam_file = ""
		
		# Main sweep data - from file
		self.power_rf_dBm = []
		self.waveform_f_Hz = []
		self.waveform_s_dBm = []
		self.waveform_rbw_Hz = []
		self.MFLI_V_offset_V = []
		self.requested_Idc_mA = []
		self.raw_meas_Vdc_V = []
		self.Idc_mA = []
		self.detect_normal = []
		self.temperature_K = []
		
		# Main sweep data - derived
		self.rf1 = []
		self.rf2 = []
		self.rf3 = []
		self.rf1W = []
		self.rf2W = []
		self.rf3W = []
		self.unique_bias = []
		self.unique_pwr = []
		self.unique_freqs = []
		
		# S-Parameter arrays
		self.S_freq_GHz = []
		self.S11 = []
		self.S21 = []
		self.S12 = []
		self.S22 = []
		self.S11_dB = []
		self.S21_dB = []
		self.S12_dB = []
		self.S22_dB = []
	
	def import_sparam(self):
		''' Imports S-parameter data into the master data object'''
		
		sp_datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm', "VNA Traces"])
		if sp_datapath is None:
			print(f"{Fore.RED}Failed to find s-parameter data location{Style.RESET_ALL}")
			sys.exit()
		else:
			print(f"{Fore.GREEN}Located s-parameter data directory at: {Fore.LIGHTBLACK_EX}{sp_datapath}{Style.RESET_ALL}")
		
		sp_filename = "Sparam_31July2024_-30dBm_R4C4T1_Wide.csv"
		
		sp_analysis_file = os.path.join(sp_datapath, sp_filename)#"Sparam_31July2024_-30dBm_R4C4T1.csv")
		
		try:
			sparam_data = read_rohde_schwarz_csv(sp_analysis_file)
		except Exception as e:
			print(f"Failed to read S-parameter CSV file. {e}")
			sys.exit()

		self.S11 = sparam_data.S11_real + complex(0, 1)*sparam_data.S11_imag
		self.S21 = sparam_data.S21_real + complex(0, 1)*sparam_data.S21_imag
		self.S11_dB = lin_to_dB(np.abs(self.S11))
		self.S21_dB = lin_to_dB(np.abs(self.S21))
		self.S_freq_GHz = sparam_data.freq_Hz/1e9
		
		self.current_sparam_file = sp_analysis_file
		
	def import_hdf(self):
		''' Imports sweep data into the master data object'''
		
		# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm'])
		datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 2 43mm'])
		# datapath = '/Volumes/M5 PERSONAL/data_transfer'
		if datapath is None:
			print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
			sys.exit()
		else:
			print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

		# filename = "RP22B_MP3_t1_31July2024_R4C4T1_r1_autosave.hdf"
		# filename = "RP22B_MP3_t1_1Aug2024_R4C4T1_r1.hdf"
		# filename = "RP22B_MP3_t2_8Aug2024_R4C4T1_r1.hdf"
		# filename = "RP22B_MP3a_t3_19Aug2024_R4C4T2_r1.hdf"
		filename = "RP22B_MP3a_t2_20Aug2024_R4C4T2_r1.hdf"
		
		analysis_file = os.path.join(datapath, filename)

		

		##--------------------------------------------
		# Read HDF5 File


		print("Loading file contents into memory")
		# log.info("Loading file contents into memory")

		t_hdfr_0 = time.time()
		with h5py.File(analysis_file, 'r') as fh:
			
			# Read primary dataset
			GROUP = 'dataset'
			self.freq_rf_GHz = fh[GROUP]['freq_rf_GHz'][()]
			self.power_rf_dBm = fh[GROUP]['power_rf_dBm'][()]
			
			self.waveform_f_Hz = fh[GROUP]['waveform_f_Hz'][()]
			self.waveform_s_dBm = fh[GROUP]['waveform_s_dBm'][()]
			self.waveform_rbw_Hz = fh[GROUP]['waveform_rbw_Hz'][()]
			
			self.MFLI_V_offset_V = fh[GROUP]['MFLI_V_offset_V'][()]
			self.requested_Idc_mA = fh[GROUP]['requested_Idc_mA'][()]
			self.raw_meas_Vdc_V = fh[GROUP]['raw_meas_Vdc_V'][()]
			self.Idc_mA = fh[GROUP]['Idc_mA'][()]
			self.detect_normal = fh[GROUP]['detect_normal'][()]
			
			self.temperature_K = fh[GROUP]['temperature_K'][()]

		##--------------------------------------------
		# Generate Mixing Products lists

		self.rf1 = spectrum_peak_list(self.waveform_f_Hz, self.waveform_s_dBm, self.freq_rf_GHz*1e9)
		self.rf2 = spectrum_peak_list(self.waveform_f_Hz, self.waveform_s_dBm, self.freq_rf_GHz*2e9)
		self.rf3 = spectrum_peak_list(self.waveform_f_Hz, self.waveform_s_dBm, self.freq_rf_GHz*3e9)

		self.rf1W = dBm2W(self.rf1)
		self.rf2W = dBm2W(self.rf2)
		self.rf3W = dBm2W(self.rf3)
		
		##-------------------------------------------
		# Calculate conversion efficiencies
		
		self.total_power = self.rf1W + self.rf2W + self.rf3W
		self.ce2 = self.rf2W/self.total_power*100
		self.ce3 = self.rf3W/self.total_power*100
		
		##-------------------------------------------
		# Generate lists of unique conditions

		self.unique_bias = np.unique(self.requested_Idc_mA)
		self.unique_pwr = np.unique(self.power_rf_dBm)
		self.unique_freqs = np.unique(self.freq_rf_GHz)
		
		self.current_sweep_file = analysis_file
		
		##------------------------------------------
		# Generate Z-scores
		
		self.zs_ce2 = calc_zscore(self.ce2)
		self.zs_ce3 = calc_zscore(self.ce3)
		
		self.zs_rf1 = calc_zscore(self.rf1)
		self.zs_rf2 = calc_zscore(self.rf2)
		self.zs_rf3 = calc_zscore(self.rf3)


##--------------------------------------------
# Create GUI

def get_graph_lims(data:list, step=None):
	
	umin = np.min(data)
	umax = np.max(data)
	
	return [np.floor(umin/step)*step, np.ceil(umax/step)*step]

def calc_zscore(data:list):
	data = np.array(data)
	mu = np.mean(data)
	stdev = np.std(data)
	return (data - mu)/stdev
	

class OutlierControlWidget(QWidget):
	
	def __init__(self):
		super().__init__()
		
		self.enable_cb = QCheckBox("Remove Outliers")
		
		self.zscore_label = QLabel("Z-Score < ")
		self.zscore_edit = QLineEdit()
		self.zscore_edit.setValidator(QDoubleValidator())
		
		self.bottom_spacer = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
		
		self.grid = QGridLayout()
		self.grid.addWidget(self.enable_cb, 0, 0, 1, 2, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
		self.grid.addWidget(self.zscore_label, 1, 0, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
		self.grid.addWidget(self.zscore_edit, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
		self.grid.addItem(self.bottom_spacer, 2, 0)
		self.setLayout(self.grid)
	
	
class ZScorePlotWindow(QMainWindow):
	
	def __init__(self, x_data, y_zscore, legend_labels, x_label):
		super().__init__()
		
		self.x_data = x_data
		self.y_zscore = y_zscore
		self.legend_labels = legend_labels
		self.x_label = x_label
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1)
		
		# Create widgets
		self.fig1cvs = FigureCanvas(self.fig1)
		self.toolbar1 = NavigationToolbar2QT(self.fig1cvs, self)
		
		self.render_plot()
		
		# Add widgets to parent-widget and set layout
		self.grid = QtWidgets.QGridLayout()
		self.grid.addWidget(self.toolbar1, 0, 0)
		self.grid.addWidget(self.fig1cvs, 1, 0)
		
		central_widget = QtWidgets.QWidget()
		central_widget.setLayout(self.grid)
		self.setCentralWidget(central_widget)
	
	def render_plot(self):
		
		self.ax1.cla()
		
		for (x, y, leglab) in zip(self.x_data, self.y_zscore, self.legend_labels):
			self.ax1.plot(x, y, label=leglab, linestyle=':', marker='X')
			
		self.ax1.set_ylabel("Z-Score")
		self.ax1.legend()
		self.ax1.grid(True)
		self.ax1.set_xlabel(self.x_label)
		
		# if type(self.y_zscore[0]) == list or type(self.y_zscore[0]) == np.ndarray: # 2D list
			
		# 	for (x, y, leglab) in zip(self.x_data, self.y_zscore, self.legend_labels):
		# 		self.ax1.plot(x, y, label=leglab)
			
		# 	self.ax1.set_ylabel("Z-Score")
		# 	self.ax1.legend()
		# 	self.ax1.grid(True)
		# 	self.ax1.set_xlabel(self.x_label)
		# else:
		# 	self.ax1.plot(self.x_data, self.y_zscore)
		# 	self.ax1.set_xlabel(self.x_label)
		# 	self.ax1.set_ylabel(self.legend_labels)
		# 	self.ax1.grid(True)
		
		self.fig1.canvas.draw_idle()

class TabPlotWidget(QWidget):
	
	def __init__(self, global_conditions:dict, log:LogPile, mdata:MasterData):
		super().__init__()
		
		self.log = log
		self.mdata = mdata
		self.gcond = global_conditions
		self.conditions = {}
		
		self._is_active = False # Indicates if the current tab is displayed
		self.plot_is_current = False # Does the plot need to be re-rendered?
		
		self.zscore_data = []
		self.zscore_labels = []
		self.zscore_x_data = []
		self.zscore_x_label = ""
		
	def init_zscore_data(self, y_data:list, legend_labels:list, x_data:list=[], x_label:str="Datapoint Index"):
		''' y_data, x_data are lists of lists. legend_label is a list of strings. Each list row corresponds to one trace.
		 Only one x-label provided. '''
		
		self.zscore_data = y_data
		self.zscore_labels = legend_labels
		self.zscore_x_data = x_data
		self.zscore_x_label = x_label
	
	def is_active(self):
		return self._is_active
	
	def set_active(self, b:bool):
		self._is_active = b
		self.plot_data()
	
	def plot_data(self):
		''' If the plot needs to be updated (active and out of date) re-renders the graph and displays it. '''
		
		if self.is_active() and (not self.plot_is_current):
			self.render_plot()
	
	def calc_mask(self):
		return None
	
	def get_condition(self, c:str):
		
		if c in self.conditions:
			return self.conditions[c]
		elif c in self.gcond:
			return self.gcond[c]
		else:
			return None
	
	def plot_zscore_if_active(self):
		''' Generates a Z-Score breakout window if window is active and if z-score window is possible. Requires:
		* calc_mask must be overridden
		* init_zscore_data must have been called.'''
		
		# Return if not active
		if not self.is_active():
			return
		
		# Return if no z-score data provided
		if len(self.zscore_data) == 0 or len(self.zscore_labels) == 0:
			self.log.info("Active plot is not showing Z-score because data was not provided.")
			return
		
		# Return if no mask
		mask = self.calc_mask()
		if mask is None:
			self.log.info("Active plot is not showing Z-score because calc_mask() was not overridden.")
			return
		
		# Create masked y-data
		y_data = []
		for zsd in self.zscore_data:
			y_data.append(zsd[mask])
		
		# Create default X values if not provided
		x_data = []
		if len(self.zscore_x_data) == 0:
			for yd in y_data:
				x_data.append(list(range(0, len(zsd))))
		else:
			for xd in self.zscore_x_data:
				x_data.append(xd[mask])
			
		self.zscore_dialog = ZScorePlotWindow(x_data, y_data, self.zscore_labels, self.zscore_x_label)
		self.zscore_dialog.show()
	
	def update_plot(self):
		''' Marks the plot to be updated eventually. '''
		
		# Indicate need to replot
		self.plot_is_current = False
		
		self.plot_data()
		
	@abstractmethod
	def render_plot(self):
		pass

class HarmGenFreqDomainPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict, log:LogPile, mdata:MasterData):
		super().__init__(global_conditions, log, mdata)
		
		# Conditions dictionaries
		self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01}
		
		self.manual_init()
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1)
		
		# Estimate system Z
		expected_Z = self.mdata.MFLI_V_offset_V[1]/(self.mdata.requested_Idc_mA[1]/1e3) #TODO: Do something more general than index 1
		system_Z = self.mdata.MFLI_V_offset_V/(mdata.Idc_mA/1e3)
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
		self.init_zscore_data([self.mdata.zs_rf1, self.mdata.zs_rf2, self.mdata.zs_rf3], ['Fundamental', '2nd Harmonic', '3rd Harmonic'], [self.mdata.freq_rf_GHz, self.mdata.freq_rf_GHz, self.mdata.freq_rf_GHz], "Frequency (GHz)")
		
		self.ylims1 = get_graph_lims(np.concatenate((self.mdata.rf1, self.mdata.rf2, self.mdata.rf3)), step=10)
		self.xlims1 = get_graph_lims(self.mdata.freq_rf_GHz, step=0.5)
		
	def calc_mask(self):
		b = self.get_condition('sel_bias_mA')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask_bias = (self.mdata.requested_Idc_mA == b)
		mask_pwr = (self.mdata.power_rf_dBm == p)
		return (mask_bias & mask_pwr)
	
	def render_plot(self):
		use_fund = self.get_condition('freqxaxis_isfund')
		b = self.get_condition('sel_bias_mA')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask = self.calc_mask()
		
		# Plot results
		self.ax1.cla()
		
		# Check correct number of points
		mask_len = np.sum(mask)
		# if len(self.mdata.unique_freqs) != mask_len:
		# 	log.warning(f"Cannot display data: Mismatched number of points (bias = {b} mA, pwr = {p} dBm, mask: {mask_len}, freq: {len(self.mdata.unique_freqs)})")
		# 	self.fig1.canvas.draw_idle()
		# 	return
		
		if use_fund:
			self.ax1.plot(self.mdata.freq_rf_GHz[mask], self.mdata.rf1[mask], linestyle=':', marker='o', markersize=4, color=(0, 0.7, 0))
			self.ax1.plot(self.mdata.freq_rf_GHz[mask], self.mdata.rf2[mask], linestyle=':', marker='o', markersize=4, color=(0, 0, 0.7))
			self.ax1.plot(self.mdata.freq_rf_GHz[mask], self.mdata.rf3[mask], linestyle=':', marker='o', markersize=4, color=(0.7, 0, 0))
			self.ax1.set_xlabel("Fundamental Frequency (GHz)")
		else:
			self.ax1.plot(self.mdata.freq_rf_GHz[mask], self.mdata.rf1[mask], linestyle=':', marker='o', markersize=4, color=(0, 0.7, 0))
			self.ax1.plot(self.mdata.freq_rf_GHz[mask]*2, self.mdata.rf2[mask], linestyle=':', marker='o', markersize=4, color=(0, 0, 0.7))
			self.ax1.plot(self.mdata.freq_rf_GHz[mask]*3, self.mdata.rf3[mask], linestyle=':', marker='o', markersize=4, color=(0.7, 0, 0))
			self.ax1.set_xlabel("Tone Frequency (GHz)")
			
		self.ax1.set_title(f"Bias = {b} mA, p = {p} dBm")
		self.ax1.set_ylabel("Power (dBm)")
		self.ax1.legend(["Fundamental", "2nd Harm.", "3rd Harm."])
		self.ax1.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax1.set_ylim(self.ylims1)
			if use_fund:
				self.ax1.set_xlim(self.xlims1)
			else:
				self.ax1.set_xlim((self.xlims1[0], self.xlims1[1]*3))
		self.fig1.tight_layout()
		
		self.fig1.canvas.draw_idle()

class CE23FreqDomainPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict, log:LogPile, mdata:MasterData):
		super().__init__(global_conditions, log, mdata)
		
		# Conditions dictionaries
		self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01}
		
		self.manual_init()
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1)
		self.fig2, self.ax2 = plt.subplots(1, 1)
		
		# Estimate system Z
		expected_Z = self.mdata.MFLI_V_offset_V[1]/(self.mdata.requested_Idc_mA[1]/1e3) #TODO: Do something more general than index 1
		system_Z = self.mdata.MFLI_V_offset_V/(self.mdata.Idc_mA/1e3)
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
		
		self.init_zscore_data([self.mdata.zs_ce2, self.mdata.zs_ce3], ["2f0 Conversion Efficiency", "3f0 Conversion Efficiency"], [self.mdata.requested_Idc_mA, self.mdata.requested_Idc_mA], "Bias Current (mA)")
		
		self.ylims1 = get_graph_lims(self.mdata.ce2, 5)
		self.ylims2 = get_graph_lims(self.mdata.ce3, 0.5)
		
		self.xlimsX = get_graph_lims(self.mdata.freq_rf_GHz, 0.5)
	
	def calc_mask(self):
		b = self.get_condition('sel_bias_mA')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask_bias = (self.mdata.requested_Idc_mA == b)
		mask_pwr = (self.mdata.power_rf_dBm == p)
		return (mask_bias & mask_pwr)
	
	def render_plot(self):
		b = self.get_condition('sel_bias_mA')
		p = self.get_condition('sel_power_dBm')
		use_fund = self.get_condition('freqxaxis_isfund')
		
		# Filter relevant data
		mask_bias = (self.mdata.requested_Idc_mA == b)
		mask_pwr = (self.mdata.power_rf_dBm == p)
		mask = (mask_bias & mask_pwr)
		
		# Plot results
		self.ax1.cla()
		self.ax2.cla()
		
		if use_fund:
			self.ax1.plot(self.mdata.freq_rf_GHz[mask], self.mdata.ce2[mask], linestyle=':', marker='o', markersize=4, color=(0.6, 0, 0.7))
			self.ax2.plot(self.mdata.freq_rf_GHz[mask], self.mdata.ce3[mask], linestyle=':', marker='o', markersize=4, color=(0.45, 0.05, 0.1))
			self.ax1.set_xlabel("Fundamental Frequency (GHz)")
			self.ax2.set_xlabel("Fundamental Frequency (GHz)")
		else:
			self.ax1.plot(self.mdata.freq_rf_GHz[mask]*2, self.mdata.ce2[mask], linestyle=':', marker='o', markersize=4, color=(0.6, 0, 0.7))
			self.ax2.plot(self.mdata.freq_rf_GHz[mask]*3, self.mdata.ce3[mask], linestyle=':', marker='o', markersize=4, color=(0.45, 0.05, 0.1))
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
			
			if use_fund:
				self.ax1.set_xlim(self.xlimsX)
				self.ax2.set_xlim(self.xlimsX)
			else:
				self.ax1.set_xlim((self.xlimsX[0]*2, self.xlimsX[1]*2))
				self.ax2.set_xlim((self.xlimsX[0]*3, self.xlimsX[1]*3))
				
		self.fig1.tight_layout()
		self.fig2.tight_layout()
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class CE23BiasDomainPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict, log:LogPile, mdata:MasterData):
		super().__init__(global_conditions, log, mdata)
		
		# Conditions dictionaries
		self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01}
		
		self.manual_init()
		
		# Create figure
		self.fig1, self.ax1 = plt.subplots(1, 1)
		self.fig2, self.ax2 = plt.subplots(1, 1)
		
		# Estimate system Z
		expected_Z = self.mdata.MFLI_V_offset_V[1]/(self.mdata.requested_Idc_mA[1]/1e3) #TODO: Do something more general than index 1
		system_Z = self.mdata.MFLI_V_offset_V/(self.mdata.Idc_mA/1e3)
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
		
		self.init_zscore_data([self.mdata.zs_ce2, self.mdata.zs_ce3], ["2f0 Conversion Efficiency", "3f0 Conversion Efficiency"], [self.mdata.requested_Idc_mA, self.mdata.requested_Idc_mA], "Bias Current (mA)")
		
		self.ylims1 = get_graph_lims(self.mdata.ce2, 5)
		self.ylims2 = get_graph_lims(self.mdata.ce3, 0.5)
		
		self.xlimsX = get_graph_lims(self.mdata.requested_Idc_mA, 0.25)
	
	def calc_mask(self):
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask_freq = (self.mdata.freq_rf_GHz == f)
		mask_pwr = (self.mdata.power_rf_dBm == p)
		return (mask_freq & mask_pwr)
	
	def render_plot(self):
		
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask = self.calc_mask()
		
		# Plot results
		self.ax1.cla()
		self.ax2.cla()
		
		self.ax1.plot(self.mdata.requested_Idc_mA[mask], self.mdata.ce2[mask], linestyle=':', marker='o', markersize=4, color=(0.6, 0, 0.7))
		self.ax2.plot(self.mdata.requested_Idc_mA[mask], self.mdata.ce3[mask], linestyle=':', marker='o', markersize=4, color=(0.45, 0.05, 0.1))
		
		self.ax1.set_title(f"f-fund = {f} GHz, f-harm2 = {rd(2*f)} GHz, p = {p} dBm")
		self.ax1.set_xlabel("Requested DC Bias (mA)")
		self.ax1.set_ylabel("2nd Harm. Conversion Efficiency (%)")
		self.ax1.grid(True)
		
		self.ax2.set_title(f"f-fund = {f} GHz, f-harm3 = {rd(3*f)} GHz, p = {p} dBm")
		self.ax2.set_xlabel("Requested DC Bias (mA)")
		self.ax2.set_ylabel("3rd Harm. Conversion Efficiency (%)")
		self.ax2.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax1.set_ylim(self.ylims1)
			self.ax1.set_xlim(self.xlimsX)
			
			self.ax2.set_ylim(self.ylims2)
			self.ax2.set_xlim(self.xlimsX)
		
		self.fig1.tight_layout()
		self.fig2.tight_layout()
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class IVPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict, log:LogPile, mdata:MasterData):
		super().__init__(global_conditions, log, mdata)
		
		# Conditions dictionaries
		self.conditions = self.conditions = {'rounding_step1': 0.1, 'rounding_step2': 0.01, 'rounding_step_x1b':0.05}
		
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
		expected_Z = self.mdata.MFLI_V_offset_V[1]/(self.mdata.requested_Idc_mA[1]/1e3) #TODO: Do something more general than index 1
		system_Z = self.mdata.MFLI_V_offset_V/(self.mdata.Idc_mA/1e3)
		self.extra_z = system_Z - expected_Z
		
		self.zs_extra_z = calc_zscore(self.extra_z)
		self.zs_meas_Idc = calc_zscore(self.mdata.Idc_mA)
		
		self.init_zscore_data( [self.zs_extra_z, self.zs_meas_Idc], ['Extra Impedance', 'Measured Idc'], [self.mdata.requested_Idc_mA, self.mdata.requested_Idc_mA], 'Requested DC Bias (mA)' )
		
		self.ylim1 = get_graph_lims(self.mdata.Idc_mA, 0.25)
		self.ylim2 = get_graph_lims(self.extra_z, 50)
		self.xlimT = get_graph_lims(self.mdata.requested_Idc_mA, 0.25)
		self.xlim1b = get_graph_lims(self.mdata.MFLI_V_offset_V, 0.1)
		self.xlim2b = self.ylim1
		
	
	def calc_mask(self):
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask_freq = (self.mdata.freq_rf_GHz == f)
		mask_pwr = (self.mdata.power_rf_dBm == p)
		return (mask_freq & mask_pwr)
	
	def render_plot(self):
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask = self.calc_mask()
		
		# Plot results
		self.ax1t.cla()
		self.ax1b.cla()
		self.ax2t.cla()
		self.ax2b.cla()
		
		# Check correct number of points
		mask_len = np.sum(mask)
		# if len(self.mdata.unique_bias) != mask_len:
		# 	log.warning(f"Cannot display data: Mismatched number of points (freq = {f} GHz, pwr = {p} dBm, mask: {mask_len}, bias: {len(self.mdata.unique_bias)})")
		# 	self.fig1.canvas.draw_idle()
		# 	returnz
		
		minval = np.min([0, np.min(self.mdata.requested_Idc_mA[mask]), np.min(self.mdata.Idc_mA[mask]) ])
		maxval = np.max([0, np.max(self.mdata.requested_Idc_mA[mask]), np.max(self.mdata.Idc_mA[mask]) ])
		
		self.ax1t.plot(self.mdata.requested_Idc_mA[mask], self.mdata.Idc_mA[mask], linestyle=':', marker='o', markersize=4, color=(0.6, 0, 0.7), label="Measured")
		self.ax1t.plot([minval, maxval], [minval, maxval], linestyle='-', color=(0.8, 0, 0), linewidth=0.5, label="1:1 ratio")
		self.ax1b.plot(self.mdata.MFLI_V_offset_V[mask], self.mdata.Idc_mA[mask], linestyle=':', marker='s', markersize=4, color=(0.2, 0, 0.8))
		
		self.ax2t.plot(self.mdata.requested_Idc_mA[mask], self.extra_z[mask], linestyle=':', marker='o', markersize=4, color=(0.45, 0.5, 0.1))
		self.ax2b.plot(self.mdata.Idc_mA[mask], self.extra_z[mask], linestyle=':', marker='o', markersize=4, color=(0, 0.5, 0.8))
		
		self.ax1t.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax1t.legend()
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
			self.ax1t.set_ylim(self.ylim1)
			self.ax1b.set_ylim(self.ylim1)
			
			self.ax2t.set_ylim(self.ylim2)
			self.ax2b.set_ylim(self.ylim2)
			
			self.ax1t.set_xlim(self.xlimT)
			self.ax2t.set_xlim(self.xlimT)
			self.ax1b.set_xlim(self.xlim1b)
			self.ax2b.set_xlim(self.xlim2b)
		
		self.fig1.tight_layout()
		self.fig2.tight_layout()
		
		self.fig1.canvas.draw_idle()
		self.fig2.canvas.draw_idle()

class SParamSPDPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict, log:LogPile, mdata:MasterData):
		super().__init__(global_conditions, log, mdata)
		
		# Conditions dictionaries
		self.conditions = {'rounding_step':10}
		
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
		# umax = np.max([np.max(self.mdata.rf1), np.max(self.mdata.rf2), np.max(self.mdata.rf3)])
		# umin = np.min([np.min(self.mdata.rf1), np.min(self.mdata.rf2), np.min(self.mdata.rf3)])
		
		# rstep = self.get_condition('rounding_step')
		# if rstep is None:
		# 	rstep = 10
		
		# self.ylims = [np.floor(umin/rstep)*rstep, np.ceil(umax/rstep)*rstep]
	
	def render_plot(self):
		# f = self.get_condition('sel_freq_GHz')
		# p = self.get_condition('sel_power_dBm')
		
		# # Filter relevant data
		# mask_freq = (self.mdata.freq_rf_GHz == f)
		# mask_pwr = (self.mdata.power_rf_dBm == p)
		# mask = (mask_freq & mask_pwr)
		
		# Plot results
		self.ax1.cla()
		
		# Check correct number of points
		# mask_len = np.sum(mask)
		# if len(self.self.mdata.unique_bias) != mask_len:
		# 	log.warning(f"Cannot display data: Mismatched number of points (freq = {f} GHz, pwr = {p} dBm, mask: {mask_len}, bias: {len(self.self.mdata.unique_bias)})")
		# 	self.fig1.canvas.draw_idle()
		# 	return
		
		
		self.ax1.plot(self.mdata.S_freq_GHz, self.mdata.S11_dB, linestyle=':', marker='o', markersize=1, color=(0.7, 0, 0))
		self.ax1.plot(self.mdata.S_freq_GHz, self.mdata.S21_dB, linestyle=':', marker='o', markersize=1, color=(0, 0.7, 0))
		
		if self.get_condition('sparam_show_sum'):
			self.ax1.plot(self.mdata.S_freq_GHz, lin_to_dB(np.abs(self.mdata.S11+self.mdata.S21)), linestyle=':', marker='.', markersize=1, color=(0.7, 0.7, 0))
			self.ax1.legend(["S11", "S21", "S11+S21"])
		else:
			self.ax1.legend(["S11", "S21"])	
		# self.ax1.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax1.set_xlabel("Frequency (GHz)")
		self.ax1.set_ylabel("Power (dBm)")
		
		self.ax1.grid(True)
		
		# if self.get_condition('fix_scale'):
		# 	self.ax1.set_ylim(self.ylims)
		
		self.fig1.tight_layout()
		
		self.fig1.canvas.draw_idle()


class HarmGenBiasDomainPlotWidget(TabPlotWidget):
	
	def __init__(self, global_conditions:dict, log:LogPile, mdata:MasterData):
		super().__init__(global_conditions, log, mdata)
		
		# Conditions dictionaries
		self.conditions = {'rounding_step':10}
		
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
	
	def calc_mask(self):
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask_freq = (self.mdata.freq_rf_GHz == f)
		mask_pwr = (self.mdata.power_rf_dBm == p)
		return (mask_freq & mask_pwr)
	
	def manual_init(self):
		
		self.init_zscore_data([self.mdata.zs_rf1, self.mdata.zs_rf2, self.mdata.zs_rf3], ['Fundamental', '2nd Harmonic', '3rd Harmonic'], [self.mdata.Idc_mA, self.mdata.Idc_mA, self.mdata.Idc_mA], "Bias Current (mA)")
		
		self.ylims1 = get_graph_lims(np.concatenate((self.mdata.rf1, self.mdata.rf2, self.mdata.rf3)), step=10)
		self.xlims1 = get_graph_lims(self.mdata.Idc_mA, step=0.25)
			
	def render_plot(self):
		f = self.get_condition('sel_freq_GHz')
		p = self.get_condition('sel_power_dBm')
		
		# Filter relevant data
		mask = self.calc_mask()
		
		# Plot results
		self.ax1.cla()
		
		self.ax1.plot(self.mdata.Idc_mA[mask], self.mdata.rf1[mask], linestyle=':', marker='o', markersize=4, color=(0, 0.7, 0))
		self.ax1.plot(self.mdata.Idc_mA[mask], self.mdata.rf2[mask], linestyle=':', marker='o', markersize=4, color=(0, 0, 0.7))
		self.ax1.plot(self.mdata.Idc_mA[mask], self.mdata.rf3[mask], linestyle=':', marker='o', markersize=4, color=(0.7, 0, 0))
		self.ax1.set_title(f"f = {f} GHz, p = {p} dBm")
		self.ax1.set_xlabel("DC Bias (mA)")
		self.ax1.set_ylabel("Power (dBm)")
		self.ax1.legend(["Fundamental", "2nd Harm.", "3rd Harm."])
		self.ax1.grid(True)
		
		if self.get_condition('fix_scale'):
			self.ax1.set_ylim(self.ylims1)
			self.ax1.set_xlim(self.xlims1)
		
		self.fig1.tight_layout()
		
		self.fig1.canvas.draw_idle()

class BiasDomainTabWidget(QTabWidget):
	
	def __init__(self, global_conditions:dict, main_window):
		super().__init__()
		
		self.main_window = main_window
		self.object_list = []
		self._is_active = False
		
		#------------ Harmonics widget
		
		self.object_list.append(HarmGenBiasDomainPlotWidget(global_conditions, self.main_window.log, self.main_window.mdata))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "Harmonic Generation")
		
		#------------ CE widget
		
		self.object_list.append(CE23BiasDomainPlotWidget(global_conditions, self.main_window.log, self.main_window.mdata))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "Efficiency")
		
		#------------ Harmonics widget
		
		self.object_list.append(IVPlotWidget(global_conditions, self.main_window.log, self.main_window.mdata))
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
		
		self.object_list.append(HarmGenFreqDomainPlotWidget(self.gcond, self.main_window.log, self.main_window.mdata))
		self.main_window.gcond_subscribers.append(self.object_list[-1])
		self.addTab(self.object_list[-1], "Harmonic Generation")
		
		#------------ CE widget
		
		self.object_list.append(CE23FreqDomainPlotWidget(self.gcond, self.main_window.log, self.main_window.mdata))
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
		
		self.object_list.append(SParamSPDPlotWidget(self.gcond, self.main_window.log, self.main_window.mdata))
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

	def __init__(self, log, mdata, app, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Save local variables
		self.log = log
		self.app = app
		self.mdata = mdata
		
		# Initialize global conditions
		self.gcond = {'sel_freq_GHz': self.mdata.unique_freqs[len(self.mdata.unique_freqs)//2], 'sel_power_dBm': self.mdata.unique_pwr[len(self.mdata.unique_pwr)//2], 'sel_bias_mA': self.mdata.unique_bias[len(self.mdata.unique_bias)//2], 'fix_scale':False, 'freqxaxis_isfund':False, "remove_outliers":True, "remove_outliers_ce2_zscore":10}
		
		self.gcond_subscribers = []
		
		# Basic setup
		if cli_args.subtle:
			self.setWindowTitle("Cryogenic Data Analyzer")
		else:
			self.setWindowTitle("Wyvern Cryogenic Data Analyzer")
		self.grid = QtWidgets.QGridLayout() # Create the primary layout
		self.add_menu()
		
		# Make a controls widget
		self.control_widget = OutlierControlWidget()
		
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
		self.active_file_label.setText(f"Active Main Sweep File: {self.mdata.current_sweep_file}")
		self.active_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
		
		# Active s-param file label
		self.active_spfile_label = QLabel()
		self.active_spfile_label.setText(f"Active S-Parameter File: {self.mdata.current_sparam_file}")
		self.active_spfile_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
		
		# Place each widget
		self.grid.addWidget(self.control_widget, 0, 0)
		self.grid.addWidget(self.tab_widget, 0, 1)
		self.grid.addWidget(self.slider_box, 0, 2)
		self.grid.addWidget(self.active_file_label, 1, 1, 1, 2)
		self.grid.addWidget(self.active_spfile_label, 2, 1, 1, 2)
		
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
			sub.gcond[key] = value
	
	def plot_all(self):
		
		for sub in self.gcond_subscribers:
			sub.update_plot()
	
	def plot_active_zscore(self):
		
		for sub in self.gcond_subscribers:
			sub.plot_zscore_if_active()
	
	def update_freq(self, x):
		try:
			new_freq = self.mdata.unique_freqs[x]
			self.set_gcond('sel_freq_GHz', new_freq)
			self.plot_all()
		except Exception as e:
			log.warning(f"Index out of bounds! ({e})")
			return
		
		self.freq_slider_vallabel.setText(f"{new_freq} GHz")
	
	def update_active_tab(self):
		
		# Set all objects to inactive
		for obj in self.tab_widget_widgets:
			obj.set_active(False)
		
		self.tab_widget_widgets[self.tab_widget.currentIndex()].set_active(True)
	
	def update_pwr(self, x):
		try:
			new_pwr = self.mdata.unique_pwr[x]
			self.set_gcond('sel_power_dBm', new_pwr)
			self.plot_all()
		except Exception as e:
			log.warning(f"Index out of bounds! ({e})")
		
		self.pwr_slider_vallabel.setText(f"{new_pwr} dBm")
		
	def update_bias(self, x):
		try:
			new_b = self.mdata.unique_bias[x]
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
		
		self.freq_slider = QSlider(Qt.Orientation.Vertical)
		self.freq_slider.valueChanged.connect(self.update_freq)
		self.freq_slider.setSingleStep(1)
		self.freq_slider.setMinimum(0)
		self.freq_slider.setMaximum(len(np.unique(self.mdata.unique_freqs))-1)
		self.freq_slider.setTickInterval(1)
		self.freq_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksLeft)
		self.freq_slider.setSliderPosition(0)
		
		self.pwr_slider_hdrlabel = QtWidgets.QLabel()
		self.pwr_slider_hdrlabel.setText("Power (dBm)")
		
		self.pwr_slider_vallabel = QtWidgets.QLabel()
		self.pwr_slider_vallabel.setText("VOID (dBm)")
		
		self.pwr_slider = QSlider(Qt.Orientation.Vertical)
		self.pwr_slider.valueChanged.connect(self.update_pwr)
		self.pwr_slider.setSingleStep(1)
		self.pwr_slider.setMinimum(0)
		self.pwr_slider.setMaximum(len(np.unique(self.mdata.unique_pwr))-1)
		self.pwr_slider.setTickInterval(1)
		self.pwr_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksLeft)
		self.pwr_slider.setSliderPosition(0)
		
		self.bias_slider_hdrlabel = QtWidgets.QLabel()
		self.bias_slider_hdrlabel.setText("Bias (mA)")
		
		self.bias_slider_vallabel = QtWidgets.QLabel()
		self.bias_slider_vallabel.setText("VOID (mA)")
		
		self.bias_slider = QSlider(Qt.Orientation.Vertical)
		self.bias_slider.valueChanged.connect(self.update_bias)
		self.bias_slider.setSingleStep(1)
		self.bias_slider.setMinimum(0)
		self.bias_slider.setMaximum(len(self.mdata.unique_bias)-1)
		self.bias_slider.setTickInterval(1)
		self.bias_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksLeft)
		self.bias_slider.setSliderPosition(0)
		
		# bottomBtn = QPushButton(icon=QIcon("./assets/max_ce2.png"), parent=self)
		# bottomBtn.setFixedSize(100, 40)
		# bottomBtn.setIconSize(QSize(100, 40))
		
		bottomBtn = QPushButton("Max CE2", parent=self)
		bottomBtn.setFixedSize(100, 40)
		# bottomBtn.setIconSize(QSize(100, 40))
		
		ng.addWidget(self.freq_slider_hdrlabel, 0, 0)
		ng.addWidget(self.freq_slider, 1, 0, alignment=Qt.AlignmentFlag.AlignHCenter)
		ng.addWidget(self.freq_slider_vallabel, 2, 0)
		
		ng.addWidget(self.pwr_slider_hdrlabel, 0, 1)
		ng.addWidget(self.pwr_slider, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
		ng.addWidget(self.pwr_slider_vallabel, 2, 1)
		
		ng.addWidget(self.bias_slider_hdrlabel, 0, 2)
		ng.addWidget(self.bias_slider, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
		ng.addWidget(self.bias_slider_vallabel, 2, 2)
		
		ng.addWidget(bottomBtn, 3, 0, 1, 3)
		
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
		
			# END Graph Menu: Freq-axis sub menu -------------
		
		self.zscore_act = QAction("Show Active Z-Score", self)
		self.zscore_act.setShortcut("Shift+Z")
		self.graph_menu.addAction(self.zscore_act)
		
		# S-Parameter Menu --------------------------------------
		
		self.sparam_menu = self.bar.addMenu("S-Params")
		self.sparam_menu.triggered[QAction].connect(self._process_sparam_menu)
		
		self.sparam_showsum_act = QAction("Show Sum", self, checkable=True)
		self.sparam_showsum_act.setShortcut("Shift+S")
		self.sparam_showsum_act.setChecked(False)
		self.set_gcond('sparam_show_sum', self.sparam_showsum_act.isChecked())
		self.sparam_menu.addAction(self.sparam_showsum_act)
		
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
		elif q.text() == "Show Active Z-Score":
			self.plot_active_zscore()
	
	def _process_sparam_menu(self, q):
		
		if q.text() == "Show Sum":
			self.set_gcond('sparam_show_sum', self.sparam_showsum_act.isChecked())
			self.plot_all()



master_data = MasterData()
app = QtWidgets.QApplication(sys.argv)

chicago_ff = get_font("./assets/Chicago.ttf")
menlo_ff = get_font("./assets/Menlo-Regular.ttf")
app.setStyleSheet(f"""
QWidget {{
	font-family: '{menlo_ff}';
}}""")

w = HGA1Window(log, master_data, app)
app.exec()
