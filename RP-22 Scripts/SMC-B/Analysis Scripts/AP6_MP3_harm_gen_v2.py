import sys
import matplotlib
matplotlib.use('qtagg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from pylogfile.base import *

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction

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
		
		## Do things that should eventually be moved?? --------------------------
		
		self.conditions = {'sel_freq_GHz': self.freq_list[len(self.freq_list)//2], 'sel_power_dBm': self.pwr_list[len(self.pwr_list)//2]}
		
		# Down-sample freq list
		if len(self.freq_list) > 15:
			self.freq_list_downsampled = downsample_labels(self.freq_list, 11)
		else:
			self.freq_list_downsampled = self.freq_list
		
		# Down-sample power list
		if len(self.pwr_list) > 15:
			self.pwr_list_downsampled = downsample_labels(self.pwr_list, 11)
		else:
			self.pwr_list_downsampled = self.pwr_list
		
		# Frequency Slider
		ax_freq = self.fig1.add_axes([0.84, 0.1, 0.03, 0.8])
		slider_freq = Slider(ax_freq, 'Freq (GHz)', np.min(self.freq_list), np.max(self.freq_list), initcolor='none', valstep=self.freq_list, color='green', orientation='vertical', valinit=self.conditions['sel_freq_GHz'])
		slider_freq.on_changed(self.update_freq)
		ax_freq.add_artist(ax_freq.yaxis)
		ax_freq.set_yticks(self.freq_list, labels=self.freq_list_downsampled)
		
		# Power Slider
		ax_pwr = self.fig1.add_axes([0.93, 0.1, 0.03, 0.8])
		slider_pwr = Slider(ax_pwr, 'Power (dBm)', np.min(self.pwr_list), np.max(self.pwr_list), initcolor='none', valstep=self.pwr_list, color='red', orientation='vertical', valinit=self.conditions['sel_power_dBm'])
		slider_pwr.on_changed(self.update_pwr)
		ax_pwr.add_artist(ax_pwr.yaxis)
		ax_pwr.set_yticks(self.pwr_list, labels=self.pwr_list_downsampled)
		
		## End do things that should eventually be moved?? --------------------------
		
		self.plot_data()
		
		# Create widgets
		self.fig1c = FigureCanvas(self.fig1)
		self.toolbar = NavigationToolbar2QT(self.fig1c, self)
		
		# Add widgets to parent-widget and set layout
		self.grid = QtWidgets.QGridLayout()
		self.grid.addWidget(self.toolbar, 0, 0)
		self.grid.addWidget(self.fig1c, 1, 0)
		self.setLayout(self.grid)
		
	def update_pwr(self, x):
		self.conditions['sel_power_dBm'] = x
		self.plot_data()
	
	def update_freq(self, x):
		self.conditions['sel_freq_GHz'] = x
		self.plot_data()
	
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
			log.warning(f"Cannot display data: Mismatched number of points (mask: {mask_len}, bias: {len(self.req_bias_list)})")
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
		
		print(f"Drawing")
		
class HGA1Window(QtWidgets.QMainWindow):

	def __init__(self, log, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Save local variables
		self.log = log
		self.tab_handles = []
		
		# Basic setup
		self.setWindowTitle("HGA1 Window")
		self.grid = QtWidgets.QGridLayout() # Create the primary layout
		self.add_menu()
		
		# Create tab widget
		self.tab_widget = QtWidgets.QTabWidget()
		self.grid.addWidget(self.tab_widget, 0, 0)
		self.make_tabs() # Make tabs
		
		
		
		# Set the central widget
		central_widget = QtWidgets.QWidget()
		central_widget.setLayout(self.grid)
		self.setCentralWidget(central_widget)
		
		self.show()
	
	def make_tabs(self):
		
		self.make_harmgen_tab()
		self.make_normal_tab()
	
	def make_harmgen_tab(self):
		''' Makes the tab for harmonic generation'''
		
		hgwidget = HarmGenPlotWidget(freq_rf_GHz, power_rf_dBm, requested_Idc_mA, {})
		
		# Add to tabs object and handle list
		ntp = TabPlot(hgwidget.fig1, hgwidget.toolbar)
		self.tab_handles.append(ntp)
		self.tab_widget.addTab(hgwidget, "Harm. Gen.")
	
	def make_normal_tab(self):
		''' Makes the tab for harmonic generation'''
		
		# Prepare tab
		ntab = QtWidgets.QWidget()
		ntlayout = QtWidgets.QGridLayout()
		ntab.setLayout(ntlayout)
		
		# Prepare new figure and toolbar
		fig_left = plt.figure()
		figc_left = FigureCanvas(fig_left)
		tbar_left = NavigationToolbar2QT(figc_left, self)
		
		# Prepare new figure and toolbar
		fig_right = plt.figure()
		figc_right = FigureCanvas(fig_right)
		tbar_right = NavigationToolbar2QT(figc_right, self)
		
		# Add to layout
		ntlayout.addWidget(tbar_left, 0, 0)
		ntlayout.addWidget(figc_left, 1, 0)
		ntlayout.addWidget(tbar_right, 0, 1)
		ntlayout.addWidget(figc_right, 1, 1)
		
		# Add to tabs object and handle list
		ntpl = TabPlot(fig_left, tbar_left)
		ntpr = TabPlot(fig_right, tbar_right)
		self.tab_handles.append(ntpl)
		self.tab_handles.append(ntpr)
		self.tab_widget.addTab(ntab, "Normal")
	
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
w = HGA1Window(log)
app.exec()
