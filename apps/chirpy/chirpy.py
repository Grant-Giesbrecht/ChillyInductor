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

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detail', help="Show log details.", action='store_true')
parser.add_argument('-m', '--macos', help="Use macOS filesystem..", action='store_true')
parser.add_argument('--julymac', help="Use macOS filesystem..", action='store_true')
parser.add_argument('-x', '--skipfit', help="Skips autofit of data.", action='store_true')
parser.add_argument('--coarse', help="Coarse fit steps.", action='store_true')
parser.add_argument('--drive', help="Specify the drive letter from which to load data.", choices=['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'], type=str.upper)
parser.add_argument('--loglevel', help="Set the logging display level.", choices=['LOWDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str.upper)
args = parser.parse_args()

# Initialize log
log = plf.LogPile()
if args.loglevel is not None:
	print(f"\tSetting log level to {args.loglevel}")
	log.set_terminal_level(args.loglevel)
else:
	log.set_terminal_level("DEBUG")
log.str_format.show_detail = args.detail

#=================== Science Code ==========================

def linear_sine (x, ampl, omega, phi, m, offset):
	''' Fitting function '''
	return np.sin(omega*x-phi)*(ampl + m*(x-x[0])) + offset

#==================== Define control parameters =======================

AMPLITUDE_CTRL = "amplitude"
FREQUENCY_CTRL = "freq"
PHI_CTRL = "phi"
SLOPE_CTRL = "slope"
OFFSET_CTRL = "offset"

FIT_EXPLORE_IDX_CTRL = "fit-explore-idx"
ZCROSS_EXPLORE_NAVG_CTRL = "zcross-n-avg"

##==================== Create custom classes for Black-Hole ======================

def find_zero_crossings(x, y):
	''' Finds zero crossings of x, then uses y-data to interpolate between the points.'''
	
	signs = np.sign(y)
	sign_changes = np.diff(signs)
	zc_indices = np.where(sign_changes != 0)[0]
	
	# Trim end points - weird things can occur if the waveform starts or ends at zero
	zc_indices = zc_indices[1:-1]
	
	# Interpolate each zero-crossing
	cross_times = []
	for zci in zc_indices:
		dx = x[zci+1] - x[zci]
		dy = y[zci+1] - y[zci]
		frac = np.abs(y[zci]/dy)
		cross_times.append(x[zci]+dx*frac)
	
	return cross_times

def get_colormap_colors(colormap_name, n):
	"""
	Returns 'n' colors as tuples that are evenly spaced in the specified colormap.
	
	Parameters:
	colormap_name (str): The name of the colormap.
	n (int): The number of colors to return.
	
	Returns:
	list: A list of 'n' colors as tuples.
	"""
	cmap = plt.get_cmap(colormap_name)
	colors = [cmap(i / (n - 1)) for i in range(n)]
	return colors

def reversed_but_actually_tho(orig_list):
	return [x for x in reversed(orig_list)]

def find_closest_index(lst, X):
	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
	return closest_index

def is_interstitial(regions, time):
	''' Checks if a time point is in-between trim regions.
	
	Requires that the regions are in chronological order.
	
	Returns:
		In a region: -1
		Outside of all regions: -2
		Between regions: region index of next region.
	
	'''
	
	# Check if in region
	for reg in regions:
		# Check to see if point is in region
		if (time <= reg[1]) and (time >= reg[0]):
			return -1
	
	# Check to see if in between regions
	for idx in range(len(regions)-1):
		if (time > regions[idx][1]) and (time < regions[idx+1][0]):
			return idx+1
	
	# Outside all bounds
	return -2

class FitResult():
	
	def __init__(self, window, times, param, param_err):
		
		self.window = copy.deepcopy(window)
		self.times = copy.deepcopy(times)
		self.param = copy.deepcopy(param)
		self.param_errs = copy.deepcopy(param_err)

class ChirpDataset(bh.BHDataset):
	
	def __init__(self, log:plf.LogPile, source_info:bh.BHDataSource, do_fit:bool=True, step_points=10):
		super().__init__(log, source_info)
		
		self.log.info(f"Creating new BHDataset object from file: >{source_info.file_fullpath}<.")
		
		self.time_ns = []
		self.volt_mV = []
		
		self.fit_results = []
		
		self.manual_fit_freqs = [] # This saves frequencies from manual fits
		
		# TODO: Some kind of error handling
		self.master_df = pd.read_csv(source_info.file_fullpath, skiprows=4, encoding='utf-8')
		
		# Raw data
		self.time_ns_full = list(self.master_df['Time']*1e9)
		self.volt_mV_full = list(self.master_df['Ampl']*1e3)
		
		self.time_ns_trim = []
		self.volt_mV_trim = []
		
		self.err_freqs = []
		
		self.fit_freqs = []
		self.fit_times = []
		self.fit_ampls = []
		self.fit_phis = []
		self.fit_slopes = []
		self.fit_offsets = []
		self.window_rms = []
		
		# Zero-crossing analysis results
		# -> Key is N_avg value
		self.zerocross_times = {}
		self.zerocross_freqs = {} 
		
		self.ampl_bounds_low = []
		self.ampl_bounds_hi = []
		self.slope_bounds_low = []
		self.slope_bounds_hi = []
		
		# self.fit_bounds = []
		
		# Define options
		self.time_reversal = False
		self.trim_time = True
		self.window_step_points = step_points
		# self.window_size_ns = 10
		self.window_size_ns = 50
		
		self.windowed_freq_analysis_linear_guided()
	
	def get_zero_cross(self, N_avg:int):
		''' Returns the list of N_avg times and frequencies in a tuple. Calculates
		values if not previously calculated.'''
		
		# Calculate parameters if required
		if N_avg not in self.zerocross_freqs:
			self.log.info("Running zero-crossings analysis for N_avg=>{N_avg}<.")
			self.calculate_zero_cross(N_avg)
		
		# Return values
		return (self.zerocross_times[N_avg], self.zerocross_freqs[N_avg])
		
	
	def calculate_zero_cross(self, N_avg:int):
		
		# Find zero crossings
		tzc = find_zero_crossings(self.time_ns_trim, self.volt_mV_trim)
		
		# Select every other zero crossing to see full periods and become insensitive to y-offsets.
		tzc_fullperiod = tzc[::2*N_avg]
		periods = np.diff(tzc_fullperiod)
				
		# Calcualte frequencies
		self.zerocross_freqs[N_avg] = (1/periods)*N_avg
		
		# Calculate times
		self.zerocross_times[N_avg] = tzc_fullperiod[:-1] + periods/2
		
	
	def windowed_freq_analysis_linear_guided(self):
		
		GUIDED_SINE_FIT = 2
		pi = 3.1415926535
		merge = True
		include_3rd = False

		fit_method = GUIDED_SINE_FIT

		# Window fit options
		window_size_ns = self.window_size_ns
		window_step_points = self.window_step_points

		# Show fits
		show_example_fits = False
		num_example_fits = 10

		# Reverse
		time_reversal = True
		reset_start_positions = True
		
		dframe = self.master_df
		pulse_range_ns = self.source_info.options['trim_regions']
		if pulse_range_ns == []:
			self.log.debug(f"No trim regions provided. Skipping trim action.")
			self.trim_time = False
			
		# pulse_range_ns=[(-410, -330), (-300, -225), (-190, -120)]
		
		def print_range(index, name, bounds, param):
			lb=bounds[0][index]
			ub=bounds[1][index]
			val=param[index]
			print(plf.markdown(f"    {name}: >:q{lb}< \< >{val}< \< >:q{ub}<"))
		
		self.log.debug(f"Beginning windowed fit on data id:>{self.unique_id}<.")
		t0 = time.time()
		
		if self.time_reversal:
			time_ns_full = np.array(reversed_but_actually_tho(self.time_ns_full))*-1
			ampl_mV_full = reversed_but_actually_tho(self.volt_mV_full)
		else:
			time_ns_full = self.time_ns_full
			ampl_mV_full = self.volt_mV_full
		
		# Trim timeseries
		if self.trim_time:
			time_ns = []
			ampl_mV = []

			for prange in pulse_range_ns:
				idx_start = find_closest_index(time_ns_full, prange[0])
				idx_end = find_closest_index(time_ns_full, prange[1])
				time_ns = np.concatenate([time_ns, time_ns_full[idx_start:idx_end+1]])
				ampl_mV = np.concatenate([ampl_mV, ampl_mV_full[idx_start:idx_end+1]])
			
		else:
			time_ns = time_ns_full
			ampl_mV = ampl_mV_full
		
		self.time_ns_trim = time_ns
		self.volt_mV_trim = ampl_mV
		
		# Get period and widnow size
		example_fit_sample_period = round((len(time_ns)/window_step_points-1)/num_example_fits)
		window_step_ns = (time_ns[1]-time_ns[0]) * self.window_step_points
		
		#===================== Perform Hilbert transform ==========================
		# Goal is to provide tighter bounds on amplitude and slope
		
		# Perform hilbert transform and calculate derivative parameters
		a_signal = hilbert(ampl_mV)
		ampl_env = np.abs(a_signal)
		
		#===================== Perform windowed gaussian fit ==========================
		num_fail = 0
		
		# Initial guess
		freq = 4.825
		freq = 0.1
		param = [50, 2*3.14159*freq, 0, 0, 0]
		
		# Set bounds
		lower = [10, 2*pi*0.08, -pi*1.5, -10, -5]
		upper = [220, 2*pi*0.32, pi*1.5, 10, 5]
		bounds = [lower, upper]
		
		# Initialize window
		window = [time_ns[0], window_size_ns+time_ns[0]]
		
		# Prepare output arrays
		fit_times = []
		fit_omegas = []
		fit_covs = []
		window_rms = []
		
		fit_ampls = []
		fit_phis = []
		fit_ms = []
		fit_offs = []
		ampl_bounds_hi = []
		ampl_bounds_low = []
		slope_bounds_low = []
		slope_bounds_hi = []
		
		# Loop over data
		count = 0
		while True:
			
			count += 1
			
			# Find start and end index
			idx0 = find_closest_index(time_ns, window[0])
			idxf = find_closest_index(time_ns, window[1])
			
			# Get fit region
			time_ns_fit = np.array(time_ns[idx0:idxf])
			ampl_mV_fit = np.array(ampl_mV[idx0:idxf])
			envl_mV_fit = np.array(ampl_env[idx0:idxf])
			
			# Check length
			if len(envl_mV_fit) < 1:
				print(f"{Fore.RED}Trimming time failed.{Style.RESET_ALL}")
				w0 = window[0]
				w1 = window[1]
				print(f"{Fore.LIGHTBLACK_EX}    Window: {w0} to {w1}{Style.RESET_ALL}")
				print(f"{Fore.LIGHTBLACK_EX}    Index: {idx0} to {idxf}{Style.RESET_ALL}")
				
			
			# Adjust bounds
			blow = np.min(np.abs(envl_mV_fit))-2
			if blow < 0:
				blow = 0
			bhi = np.max(np.abs(envl_mV_fit))+2
			bounds[0][0] = blow # Adjust lower bounds
			bounds[1][0] = bhi # Adjust upper bounds
			
			# Adjust slope
			est_slope_abs = (np.max(envl_mV_fit)-np.min(envl_mV_fit))/(time_ns_fit[-1]-time_ns_fit[0])
			est_delta = np.mean(envl_mV_fit[-5:]) - np.mean(envl_mV_fit[:5])
			slope_sign = (est_delta)/np.abs(est_delta)
			est_slope = est_slope_abs * slope_sign
			if slope_sign > 0:
				slope_bounds = [est_slope * 0.25 - 0.5, est_slope * 1.5 + 0.5]
			else:
				slope_bounds = [est_slope * 1.5 - 0.5, est_slope * 0.25 + 0.5]
			
			bounds[0][3] = np.min(slope_bounds)
			bounds[1][3] = np.max(slope_bounds)
			
			# Set initial guess
			param[0] = np.mean([np.max(envl_mV_fit), np.min(envl_mV_fit)])
			param[3] = est_slope
			
			# Perform fit
			try:
				param, param_cov = curve_fit(linear_sine, time_ns_fit, ampl_mV_fit, p0=param, bounds=bounds)
				
				# if show_example_fits and np.mod(count, example_fit_sample_period) == 0:
					
				# 	fit_vals = linear_sine(time_ns_fit, param[0], param[1], param[2], param[3], param[4])
				# 	print(f"Showing fit sample: (count={count})")
				# 	print_range(0, "Amplitude", bounds, param)
				# 	print_range(1, "Omega", bounds, param)
				# 	print_range(2, "Phi", bounds, param)
				# 	print_range(3, "Slope", bounds, param)
				# 	print_range(4, "Offset", bounds, param)
					
				# 	fig1 = plt.figure(1)
				# 	gs = fig1.add_gridspec(2, 1)
				# 	ax1a = fig1.add_subplot(gs[0, 0])
				# 	ax1b = fig1.add_subplot(gs[1, 0])
					
				# 	ax1a.plot(time_ns, ampl_mV)
				# 	ax1a.grid(True)
					
				# 	rect = patches.Rectangle((window[0], np.min(ampl_mV)-5), window[1]-window[0], np.max(ampl_mV)-np.min(ampl_mV)+10, color=(0, 0.8, 0), alpha=0.15)
					
				# 	ax1b.plot(time_ns_fit, ampl_mV_fit, linestyle='--', marker='.', color=(0.6, 0, 0), alpha=0.5, label="Measured data")
				# 	ax1b.plot(time_ns_fit, envl_mV_fit, linestyle='--', marker='.', color=(1, 0.5, 0.05), alpha=0.5, label="Envelope")
				# 	ax1b.plot(time_ns_fit, fit_vals, linestyle='--', marker='.', color=(0, 0, 0.6), alpha=0.5, label="Fit")
				# 	ax1a.add_patch(rect)
				# 	ax1b.set_title("Fit Sections")
				# 	ax1b.grid(True)
				# 	ax1b.legend()
				# 	plt.tight_layout()
				# 	plt.show()
				
			except Exception as e:
				
				try:
					self.log.warning(f"Failed to converge: (>:q{e}<). Attempting phase shift.")
					
					local_bounds = copy.deepcopy(bounds)
									
					param[2] -= pi
					if param[2] < -pi:
						param[2] += 2*pi
					
					local_bounds[0][2] = param[2]-.5
					local_bounds[1][2] = param[2]+.5
					
					param, param_cov = curve_fit(linear_sine, time_ns_fit, ampl_mV_fit, p0=param, bounds=local_bounds)
					
				except Exception as e:
					num_fail += 1
					self.log.error(f"Failed to converge: (>:q{e}<)")
					
					# fit_vals = linear_sine(time_ns_fit, param[0], param[1], param[2], param[3], param[4])
					# print_range(0, "Amplitude", bounds, param)
					# print_range(1, "Omega", bounds, param)
					# print_range(2, "Phi", bounds, param)
					# print_range(3, "Slope", bounds, param)
					# print_range(4, "Offset", bounds, param)
					
					# if num_fail <= 3:
					# 	fig1 = plt.figure(1)
					# 	gs = fig1.add_gridspec(2, 1)
					# 	ax1a = fig1.add_subplot(gs[0, 0])
					# 	ax1b = fig1.add_subplot(gs[1, 0])
						
					# 	ax1a.plot(time_ns, ampl_mV)
					# 	ax1a.grid(True)
						
					# 	ax1b.plot(time_ns_fit, ampl_mV_fit, linestyle='--', marker='.', color=(0.6, 0, 0), alpha=0.5, label="Measured data")
					# 	ax1b.plot(time_ns_fit, envl_mV_fit, linestyle='--', marker='.', color=(1, 0.5, 0.05), alpha=0.5, label="Envelope")
					# 	ax1b.plot(time_ns_fit, fit_vals, linestyle='--', marker='.', color=(0, 0, 0.6), alpha=0.5, label="Fit")
					# 	ax1b.set_title("Fit Sections")
					# 	ax1b.grid(True)
					# 	ax1b.legend()
					# 	plt.tight_layout()
					# 	plt.show()
					
					# Update window
					window[0] += window_step_ns #window_size_ns*window_step_fraction
					window[1] += window_step_ns #window_size_ns*window_step_fraction
					
					continue
				
			param_err = np.sqrt(np.diag(param_cov))
			
			self.fit_results.append(FitResult(window, time_ns_fit, param, param_err))
			
			# Save data
			fit_times.append((time_ns_fit[0]+time_ns_fit[-1])/2)
			fit_omegas.append(param[1])
			fit_covs.append(param_err[1])
			fit_ampls.append(param[0])
			fit_phis.append(param[2])
			fit_ms.append(param[3])
			fit_offs.append(param[4])
			self.window_rms.append(np.sqrt(np.mean(ampl_mV_fit**2)))
			
			# Record bounds
			ampl_bounds_low.append(bounds[0][0])
			ampl_bounds_hi.append(bounds[1][0])
			slope_bounds_low.append(bounds[0][3])
			slope_bounds_hi.append(bounds[1][3])
			
			# Update window
			window[0] += window_step_ns #window_size_ns*window_step_fraction
			window[1] += window_step_ns #window_size_ns*window_step_fraction
			
			# Break condition
			if window[1] > time_ns[-1]:
				break
			
			# Else check for window in-between pulses
			if self.trim_time:
				next_region = is_interstitial(pulse_range_ns, window[1])
				if next_region != -1: # Not in a region
					
					if next_region == -2: # THis should have been caught above
						self.log.error(f"Outside all bounds!")
					else:
						self.log.debug(f"Jumping to region idx >{next_region}<")
						window = [pulse_range_ns[next_region][0], pulse_range_ns[next_region][0]+window_size_ns]
		
		# convert from angular frequency
		fit_freqs = np.array(fit_omegas)/2/np.pi
		fit_err = np.array(fit_covs)/2/np.pi
		
		self.log.info(f"Fit (data id:>{self.unique_id}<) completed in >:a{time.time()-t0}< s")
		
		self.fit_freqs = fit_freqs
		self.fit_times = fit_times
		self.fit_ampls = fit_ampls
		self.fit_phis = fit_phis
		self.fit_slopes = fit_ms
		self.fit_offsets = fit_offs
		
		self.ampl_bounds_low = ampl_bounds_low
		self.ampl_bounds_hi = ampl_bounds_hi
		self.slope_bounds_low = slope_bounds_low
		self.slope_bounds_hi = slope_bounds_hi
		
		# return {'fit_freqs': fit_freqs, 'fit_err':fit_err, 'fit_times':fit_times, 'raw_times':time_ns, 'raw_ampl':ampl_mV, 'fit_offs':fit_offs, 'fit_ampls':fit_ampls, 'fit_phis':fit_phis, 'fit_ms':fit_ms, 'bounds':bounds, 'ampl_bounds_low':ampl_bounds_low, 'ampl_bounds_hi':ampl_bounds_hi, 'slope_bounds_hi':slope_bounds_hi, 'slope_bounds_low':slope_bounds_low}



# class AutoFitViewerWidget(QWidget):
	
# 	def __init__(self, main_window, slider_panel):
# 		super().__init__()
		
# 		self.slider_panel = slider_panel
		
# 		self.control_parameters = ["frequency"]
		
# 		self.main_window = main_window
# 		self.transfer_state_button = QPushButton("Apply to sliders")
# 		self.transfer_state_button.setFixedSize(100, 25)
# 		self.transfer_state_button.clicked.connect(self._apply_fit)
	
# 	def _apply_fit(self):
		
# 		# Get the active dataset
# 		dm = self.main_window.data_manager
# 		ds = dm.get_active()
		
# 		ds.fit_ampls[]
		
# 		# Apply to sliders
# 		for cp in self.control_parameters:
			
# 			ds.get_para
		
class ZeroCrossWidget(QWidget):
	''' This inherits from QWidget not BHListenerWidget because it will contain
	both listener and controller widgets. '''
	
	def __init__(self, main_window):
		super().__init__()
		
		self.main_window = main_window
		self.listener_widgets = [] # Used to set the active state for all listeners in widget
		
		# Create plot widget
		self.plot_widget = bhw.BHMultiPlotWidget(main_window, grid_dim=[1, 1], plot_locations=[[0, 0]], custom_render_func=self.render_zcross, include_settings_button=True)
		
		# Add parameters for adjustable bounds
		self.plot_widget.configure_integrated_bounds(ax=0, xlim=None, ylim=[4.7, 4.9])
		
		# Add to control subscriber and local listeners
		self.main_window.add_control_subscriber(self.plot_widget)
		self.listener_widgets.append(self.plot_widget)
		
		# Create slider
		self.n_avg_slider = bhw.BHSliderWidget(main_window, ZCROSS_EXPLORE_NAVG_CTRL, header_label="Averaging\nCount", min=1, max=10, step=1, initial_val=3)
		self.main_window.add_dataset_subscriber(self.n_avg_slider)
		
		# self.fit_cb = QCheckBox("Show Fit")
		# self.fit_cb.setChecked(True)
		# self.fit_cb.stateChanged.connect(self.render_zcross)
		
		# Apply widgets
		self.grid = QGridLayout()
		self.grid.addWidget(self.plot_widget, 0, 0)
		self.grid.addWidget(self.n_avg_slider, 0, 1)
		self.setLayout(self.grid)
	
	@staticmethod
	def render_zcross(pw):
		''' Callback for plot update. '''
		
		NUM_PLOTS = 1
		ds = pw.data_manager.get_active()
		
		# Abort if no dataset exists
		if ds is None:
			return
		
		# Get averaging number
		N_avg = pw.control_requested.get_param(ZCROSS_EXPLORE_NAVG_CTRL)
		(t_zc, f_zc) = ds.get_zero_cross(N_avg)
		
		# Clear old data
		for i in range(NUM_PLOTS):
			pw.axes[i].cla()
		
		# Replot
		pw.axes[0].plot(t_zc, f_zc, linestyle=':', marker='o', color=(0, 0.65, 0), label='Zero-Crossing')
		pw.axes[0].set_ylabel("Frequency (GHz)")
		pw.axes[0].set_title("Zero-Crossing Analysis")
		pw.axes[0].legend()
		
		pw.axes[0].plot(ds.fit_times, ds.fit_freqs, linestyle=':', marker='.', color=(0.65, 0, 0.35), label='Fit')
		pw.axes[0].legend()
		
		# Apply universal settings
		for i in range(NUM_PLOTS):
			pw.axes[i].grid(True)
			pw.axes[i].set_xlabel("Time (ns)")
	
	def set_active(self, b:bool):
		
		for lw in self.listener_widgets:
			lw.set_active(b)


class FitExplorerWidget(QWidget):
	''' This inherits from QWidget not BHListenerWidget because it will contain
	both listener and controller widgets. '''
	
	def __init__(self, main_window):
		super().__init__()
		
		self.main_window = main_window
		self.listener_widgets = [] # Used to set the active state for all listeners in widget
		
		# Create plot widget
		self.plot_widget = bhw.BHMultiPlotWidget(main_window, grid_dim=[3, 2], plot_locations=[[slice(0,2), slice(0, 2)], [2, 0], [2, 1]], custom_render_func=self.render_manual_fit, include_settings_button=True)
		
		# Add parameters for adjustable bounds
		self.plot_widget.configure_integrated_bounds(ax=0, xlim=None, ylim=[-75, 75])
		
		# Add to control subscriber and local listeners
		self.main_window.add_control_subscriber(self.plot_widget)
		self.listener_widgets.append(self.plot_widget)
		
		# Create slider
		self.fit_idx_slider = bhw.BHSliderWidget(main_window, FIT_EXPLORE_IDX_CTRL, header_label="Fit Number", min=0, max=1, step=1, dataset_changed_callback=self.dataset_changed) #TODO: Update slider when 
		self.main_window.add_dataset_subscriber(self.fit_idx_slider)
		
		# Create transfer button
		self.transfer_state_button = QPushButton("Apply to sliders", parent=self)
		self.transfer_state_button.setFixedSize(100, 25)
		self.transfer_state_button.clicked.connect(self._apply_fit)
		
		# Create print button
		self.print_button = QPushButton("Print fit", parent=self)
		self.print_button.setFixedSize(100, 25)
		self.print_button.clicked.connect(self._print_fit)
		
		# Apply widgets
		self.grid = QGridLayout()
		self.grid.addWidget(self.plot_widget, 0, 0, 3, 1)
		self.grid.addWidget(self.fit_idx_slider, 0, 1)
		self.grid.addWidget(self.transfer_state_button, 1, 1)
		self.grid.addWidget(self.print_button, 2, 1)
		self.setLayout(self.grid)
	
	def _apply_fit(self):
		
		# Get current dataset
		ds = self.main_window.data_manager.get_active()
		if ds is None:
			return
		
		# Get selected index
		idx = self.fit_idx_slider.get_slider_position()
		
		# Get values for each parameter
		ampl_val = ds.fit_ampls[idx]
		freq_val = ds.fit_freqs[idx]
		phi_val = ds.fit_phis[idx]
		slope_val = ds.fit_slopes[idx]
		offset_val = ds.fit_offsets[idx]
		
		
		# Apply to sliders
		slider_dict = self.main_window.slider_group_widget.sliders
		for sldr_key in slider_dict:
			
			if sldr_key == AMPLITUDE_CTRL:
				slider_dict[sldr_key].set_slider_position(ampl_val)
			elif sldr_key == FREQUENCY_CTRL:
				slider_dict[sldr_key].set_slider_position(freq_val*1e3)
			elif sldr_key == PHI_CTRL:
				slider_dict[sldr_key].set_slider_position(phi_val*180/np.pi)
			elif sldr_key == SLOPE_CTRL:
				slider_dict[sldr_key].set_slider_position(slope_val)
			elif sldr_key == OFFSET_CTRL:
				slider_dict[sldr_key].set_slider_position(offset_val)
	
	def _print_fit(self):
		
		# Get current dataset
		ds = self.main_window.data_manager.get_active()
		if ds is None:
			return
		
		# Get selected index
		idx = self.fit_idx_slider.get_slider_position()
		
		# Get values for each parameter
		ampl_val = ds.fit_ampls[idx]
		freq_val = ds.fit_freqs[idx]
		phi_val = ds.fit_phis[idx]
		slope_val = ds.fit_slopes[idx]
		offset_val = ds.fit_offsets[idx]
		fit_time = ds.fit_times[idx]
		
		print(plf.markdown(f"Fit Parameters: dataset=>{ds.source_info.file_name}<, ID=>:q{ds.source_info.unique_id}<"))
		print(plf.markdown(f"    Fit Index = {idx}"))
		print(plf.markdown(f"    Fit Time = {fit_time} (ns)"))
		print(plf.markdown(f">:q        Amplitude = >{ampl_val}>:q (mV)<"))
		print(plf.markdown(f">:q        Frequency = >{freq_val}>:q (GHz)<"))
		print(plf.markdown(f">:q        Phi       = >{phi_val*180/np.pi}>:q (deg)<"))
		print(plf.markdown(f">:q        Slope     = >{slope_val}>:q (mV/ns)<"))
		print(plf.markdown(f">:q        Offset    = >{offset_val}>:q (mV)<"))
		
	
	@staticmethod
	def dataset_changed(wid):
		
		# Get current dataset
		ds = wid.data_manager.get_active()
		
		# Change slider maximum to ds length-1
		wid.set_maximum(len(ds.fit_times)-1)
	
	@staticmethod
	def render_manual_fit(pw):
		''' Callback for plot update. '''
		
		NUM_PLOTS = 3
		ds = pw.data_manager.get_active()
		
		# Abort if no dataset exists
		if ds is None:
			return
		
		# Abort if fit has not been run
		if len(ds.fit_ampls) < 1:
			return
		
		# Get window size and fit result
		fit_idx = pw.control_requested.get_param(FIT_EXPLORE_IDX_CTRL)
		window = ds.fit_results[fit_idx].window
		fit_y = linear_sine(ds.fit_results[fit_idx].times, *ds.fit_results[fit_idx].param)
		
		# Calculate sine
		ampl = pw.control_requested.get_param(AMPLITUDE_CTRL)
		freq = pw.control_requested.get_param(FREQUENCY_CTRL)
		phi = pw.control_requested.get_param(PHI_CTRL)*np.pi/180
		slope = pw.control_requested.get_param(SLOPE_CTRL)
		offset = pw.control_requested.get_param(OFFSET_CTRL)
		
		omega = freq/1e3*2*np.pi
		y = linear_sine(ds.fit_results[fit_idx].times, ampl, omega, phi, slope, offset)
		
		# Clear old data
		for i in range(NUM_PLOTS):
			pw.axes[i].cla()
		
		# Replot
		pw.axes[0].plot(ds.fit_results[fit_idx].times, y, linestyle=':', marker='x', color=(0, 0.65, 0), label='Fit')
		pw.axes[0].set_ylabel("Amplitude (mV)")
		pw.axes[0].set_title("Fit Section")
		
		pw.axes[0].plot(ds.fit_results[fit_idx].times, fit_y, linestyle=':', marker='.', color=(0.75, 0, 0), label='Measured Data')
		pw.axes[0].grid(True)
		pw.axes[0].legend()
		
		pw.axes[1].plot(ds.fit_times, ds.fit_freqs, linestyle=':', marker='.', color=(0.65, 0, 0.35))
		pw.axes[1].set_ylabel("Frequency (GHz)")
		pw.axes[1].set_title("Auto-fit Frequency")
		pw.axes[1].grid(True)
		
		rect = patches.Rectangle((window[0], np.min(ds.volt_mV_full)-5), window[1]-window[0], np.max(ds.volt_mV_full)-np.min(ds.volt_mV_full)+10, color=(0, 0.8, 0), alpha=0.15)
		
		pw.axes[2].plot(ds.time_ns_full, ds.volt_mV_full, linestyle=':', marker='.', color=(0, 0, 0.65))
		pw.axes[2].add_patch(rect)
		pw.axes[2].set_ylabel("Amplitude (mV)")
		pw.axes[2].set_title("Full Chirp")
		pw.axes[2].grid(True)
		
		# Apply universal settings
		for i in range(NUM_PLOTS):
			pw.axes[i].grid(True)
			pw.axes[i].set_xlabel("Time (ns)")
	
	def set_active(self, b:bool):
		
		for lw in self.listener_widgets:
			lw.set_active(b)

class ChirpAnalyzerMainWindow(bh.BHMainWindow):
	
	def __init__(self, log, app, data_manager):
		super().__init__(log, app, data_manager, window_title="Chirp Analyzer")
		
		self.main_grid = QGridLayout()
		
		# Create select widget
		self.select_widget = bh.BHDatasetSelectBasicWidget(self, log)
		
		# Initialize control state
		self.control_requested.add_param(AMPLITUDE_CTRL, 50)
		self.control_requested.add_param(FREQUENCY_CTRL, 4850)
		self.control_requested.add_param(PHI_CTRL, 0)
		self.control_requested.add_param(SLOPE_CTRL, 0)
		self.control_requested.add_param(OFFSET_CTRL, 0)
		self.control_requested.add_param(FIT_EXPLORE_IDX_CTRL, 0)
		self.control_requested.add_param(ZCROSS_EXPLORE_NAVG_CTRL, 1)
		
		# Create fit viewer
		self.auto_fit_plot =  bhw.BHMultiPlotWidget(self, grid_dim=[4, 2], plot_locations=[[slice(0,2), 0], [slice(2,4), 0], [0, 1], [1, 1], [2, 1], [3, 1]], custom_render_func=render_auto_fit)
		self.add_control_subscriber(self.auto_fit_plot)
		
		# Create rms viewer
		self.rms_plot =  bhw.BHMultiPlotWidget(self, grid_dim=[2, 1], plot_locations=[[0, 0], [1, 0]], custom_render_func=render_rms)
		self.add_control_subscriber(self.rms_plot)
		
		# Manual sine window
		# self.manual_plot = bhw.BHPlotWidget(self, custom_render_func=render_sine)
		# self.add_control_subscriber(self.manual_plot)
		
		self.fit_explorer = FitExplorerWidget(self)
		
		self.zero_cross = ZeroCrossWidget(self)
		
		
		
		# self.manual_plot = bhw.BHMultiPlotWidget(self, grid_dim=[2, 2], plot_locations=[[0, slice(0, 2)], [1, 0], [1, 1]], custom_render_func=render_manual_fit)
		# self.add_control_subscriber(self.manual_plot)
		
		# Make Tab widget
		self.tab_widget = bh.BHTabWidget(self)
		self.tab_widget.addTab(self.auto_fit_plot, "Fit Viewer")
		self.tab_widget.addTab(self.fit_explorer, "Manual Fit Explorer")
		self.tab_widget.addTab(self.rms_plot, "RMS Viewer")
		self.tab_widget.addTab(self.zero_cross, "Zero-Crossings Viewer")
		self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		
		#TODO: Create a controller
		self.ampl_slider = bhw.BHSliderWidget(self, param=AMPLITUDE_CTRL, header_label="Amplitude", min=0, max=200, unit_label="mV", step=0.5)
		self.freq_slider = bhw.BHSliderWidget(self, param=FREQUENCY_CTRL, header_label="Frequency", min=4700, max=4900, unit_label="MHz", step=1)
		self.phi_slider = bhw.BHSliderWidget(self, param=PHI_CTRL, header_label="Phi", min=-270, max=270, unit_label="deg", step=1)
		self.slope_slider = bhw.BHSliderWidget(self, param=SLOPE_CTRL, header_label="Slope", min=-2.5, max=2.5, unit_label="mV/ns", step=0.025)
		self.offset_slider = bhw.BHSliderWidget(self, param=OFFSET_CTRL, header_label="Offset", min=-5, max=5, unit_label="mV", step=0.05)
		
		self.slider_group_widget = bhw.BHSliderPanel(self)
		self.slider_group_widget.add_slider(self.ampl_slider)
		self.slider_group_widget.add_slider(self.freq_slider)
		self.slider_group_widget.add_slider(self.phi_slider)
		self.slider_group_widget.add_slider(self.slope_slider)
		self.slider_group_widget.add_slider(self.offset_slider)
		
		self.add_basic_menu_bar()
		
		# Position widgets
		self.main_grid.addWidget(self.tab_widget, 0, 0)
		self.main_grid.addWidget(self.slider_group_widget, 0, 1)
		# self.main_grid.addWidget(self.ampl_slider, 0, 1)
		# self.main_grid.addWidget(self.freq_slider, 0, 2)
		# self.main_grid.addWidget(self.phi_slider, 0, 3)
		# self.main_grid.addWidget(self.slope_slider, 0, 4)
		# self.main_grid.addWidget(self.offset_slider, 0, 5)
		self.main_grid.addWidget(self.select_widget, 1, 0)
		
		# Create central widget
		self.central_widget = QtWidgets.QWidget()
		self.central_widget.setLayout(self.main_grid)
		self.setCentralWidget(self.central_widget)
		
		self.show()

##==================== Create custom functions for Black-Hole ======================

def load_chirp_dataset(source, log):
	pts = 10
	if args.coarse:
		pts = 100
	return ChirpDataset(log, source, do_fit=(not args.skipfit), step_points=pts)

def render_rms(pw):
	
	NUM_PLOTS = 2
	
	# clear all axes
	for i in range(NUM_PLOTS):
		pw.axes[i].cla()
	
	# Get data from manager
	dataset = pw.data_manager.get_active()
	if dataset is None:
		return
	
	pw.axes[0].plot(dataset.fit_times, dataset.window_rms, linestyle=':', marker='.', color=(0, 0.65, 0))
	pw.axes[1].plot(dataset.time_ns_full, dataset.volt_mV_full, linestyle=':', marker='.', color=(0, 0, 0.7))
	
	pw.axes[0].set_ylabel("Amplitude (mV RMS)")
	pw.axes[1].set_ylabel("Amplitude (mV)")
	
	pw.axes[0].set_title("Window Fit RMS Values")
	pw.axes[1].set_title("Full Pulse")
	
	# Format all axes
	for i in range(NUM_PLOTS):
		pw.axes[i].set_xlabel("Time (ns)")
		pw.axes[i].grid(True)

def render_auto_fit(pw):
	
	NUM_PLOTS = 6
	
	# Clear all axes
	for i in range(NUM_PLOTS):
		pw.axes[i].cla()
	
	# Get data from manager
	dataset = pw.data_manager.get_active()
	if dataset is None:
		return
	
	pw.axes[0].plot(dataset.fit_times, dataset.fit_freqs, linestyle=':', marker='.', color=(0, 0.65, 0))
	pw.axes[1].plot(dataset.time_ns_full, dataset.volt_mV_full, linestyle=':', marker='.', color=(0, 0, 0.7))
	
	color2 = (0.75, 0, 0)
	pw.axes[2].plot(dataset.fit_times, dataset.fit_ampls, linestyle=':', marker='.', color=color2)
	pw.axes[3].plot(dataset.fit_times, dataset.fit_phis, linestyle=':', marker='.', color=color2)
	pw.axes[4].plot(dataset.fit_times, dataset.fit_slopes, linestyle=':', marker='.', color=color2)
	pw.axes[5].plot(dataset.fit_times, dataset.fit_offsets, linestyle=':', marker='.', color=color2)
	
	pw.axes[0].set_ylabel("Frequency (GHz)")
	pw.axes[1].set_ylabel("Amplitude (mV)")
	pw.axes[2].set_ylabel("Amplitude (mV)")
	pw.axes[3].set_ylabel("Offset (mV)")
	pw.axes[4].set_ylabel("Slope (mV/ns)")
	pw.axes[5].set_ylabel("Phase Shift (rad)")
	
	
	# Format all axes
	for i in range(NUM_PLOTS):
		pw.axes[i].set_xlabel("Time (ns)")
		pw.axes[i].grid(True)
	

#================= Basic PyQt App creation things =========================

# Create app object
app = QtWidgets.QApplication(sys.argv)
app.setStyle(f"Fusion")
# app.setWindowIcon

# Create Data Manager
data_manager = bh.BHDatasetManager(log, load_function=load_chirp_dataset)
if args.julymac:
	if not data_manager.load_configuration("chirpy_conf_macOS_july.json"):
		exit()
elif args.macos:
	if not data_manager.load_configuration("chirpy_conf_macOS.json"):
		exit()
else:
	drive_letter = "D:\\"
	if args.drive is not None:
		drive_letter = f"{args.drive}:\\"
	if not data_manager.load_configuration("chirpy_conf.json", user_abbrevs={"$DRIVE_LETTER$":drive_letter}):
		exit()

window = ChirpAnalyzerMainWindow(log, app, data_manager)

app.exec()