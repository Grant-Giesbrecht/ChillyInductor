from colorama import Fore, Style
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

import logging
import getopt, sys
from dataclasses import dataclass
import math

LOG_LEVEL = logging.WARNING

#-----------------------------------------------------------
# Parse arguments
argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "debug", "info", "warning", "error", "critical"])
except getopt.GetoptError as err:
	print("--help for help")
	sys.exit(2)
for opt, aarg in opts:
	if opt in ("-h", "--help"):
		print(f"{Fore.RED}Just kidding I haven't made any help text yet. ~~OOPS~~{Style.RESET_ALL}")
		sys.exit()
	elif opt == "--debug":
		LOG_LEVEL = logging.DEBUG
	elif opt == "--info":
		LOG_LEVEL = logging.INFO
	else:
		assert False, "unhandled option"
	# ...
#-----------------------------------------------------------

tabchar = "    "
prime_color = Fore.YELLOW
standard_color = Fore.LIGHTBLACK_EX
quiet_color = Fore.WHITE
logging.basicConfig(format=f'{prime_color}%(levelname)s:{standard_color} %(message)s{quiet_color} | %(asctime)s{Style.RESET_ALL}', level=LOG_LEVEL)

PI = 3.1415926535

@dataclass
class LKSolution:
	Lk = None
	
	Vp = None # Not actually used
	betaL = None # Electrical length of chip in radians
	P0 = None
	theta = None
	Iac = None
	Ibias = None
	Zin = None # Impedance looking into chip (from source side)
	
	L_ = None # From Lk
	Zchip = None # From L_, characteristic impedance of chip
	
	Iac_result_td = None # Use Iac to find solution and calculate Iac again, this is the result in time domain
	Iac_result_spec = None # Iac result as spectrum, shows fundamental, 2harm, and 3harm as touple (idx 0 = fund, ..., 2 = 3rd harm)
	rmse = None # |Iac_result - Iac|

def rd(x:float, num_decimals:int=2):
	return f"{round(x*10**num_decimals)/(10**num_decimals)}"

def find_nearest(array,value):
	""" Finds closest value.
	
	Thanks to StackExchange:
	https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
	"""
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
		return idx-1
	else:
		return idx

def xfmr(Z0, ZL, betaL):
	""" Calculates the input impedance looking into a transformer with characteristic 
	impedance Z0, terminated in load ZL, and electrical length betaL (radians)"""
	return Z0 * (ZL + 1j*Z0*np.tan(betaL))/(Z0 + 1j*ZL*np.tan(betaL))

class LKSystem:
	""" This class represents a solution to the nonlinear chip system, give a set of input conditions (things
	like actual chip length, input power, etc)."""
	
	def __init__(self, Pgen_dBm:float, C_:float, l_phys:float, freq:float, q:float, L0:float):
		""" Initialize system with given conditions """
		
		# Simulations options
		self.use_interp = False
		
		self.Pgen = (10**(Pgen_dBm/10))/1000
		self.C_ = C_
		self.l_phys = l_phys
		self.freq = freq
		self.q = q
		self.L0 = L0
		
		self.Zcable = 50 # Z0 of cable leading into chip
		self.Rsrc = 50 # R of generator
		self.Xsrc = 0 # X of generator
		
		# Time domain options
		self.num_periods = None
		self.max_harm = None
		self.min_points_per_wave = None
		self.t = []
		
		# Solve for Generator voltage from power
		self.Vgen  = np.sqrt(self.Pgen*200)
		
		# Create solution object
		self.soln = LKSolution() # Current solution data
		
		self.configure_time_domain(1000, 3, 30)
		
	def configure_time_domain(self, num_periods:float, max_harm:int, min_points_per_wave:int=10):
		""" Configures the time domain settings
		 
		  num_periods: Minimum number of periods to simulate
		  max_harm: Number of harmoinics to simulate
		  min_points_per_wave: Minimum number of time points per wavelength (at all frequencies)
		  
		"""
		
		# Save options
		self.num_periods = num_periods
		self.max_harm = max_harm
		self.min_points_per_wave = min_points_per_wave
		
		# Calculate max 
		t_max = num_periods/self.freq
		num_points = min_points_per_wave*max_harm*num_periods+1
		
		# Create t array
		self.t = np.linspace(0, t_max, num_points)
		
		logging.info(f"Configured time domain with {len(self.t)} points.")
	
	def solve_inductance(self, Iac:float, Idc:float):
		""" Given the input current values, finds the inductance of the chip
		and updates the solution with its value."""
		
		# Solve for inductance (Lk)
		self.soln.Lk = self.L0 + self.L0/self.q**2 * ( Idc**2 + 2*Idc*Iac*np.sin(self.freq*2*PI*self.t) + Iac**2/2 - Iac**2/2*np.cos(2*self.freq*2*PI*self.t) ) 
		
		self.soln.L_ = self.soln.Lk/self.l_phys
		
		# Lk_DC = self.L0 + self.L0/self.q**2 * ( Idc**2 + Iac**2/2) 
		# Lk_fund = self.L0 + self.L0/self.q**2 * ( 2*Idc*Iac ) 
		# Lk_2H = self.L0 + self.L0/self.q**2 * ( Idc**2 + 2*Idc*Iac*np.sin(self.freq*2*PI) ) 
				
	def check_solution(self, Iac:float, Idc:float, show_plot=False):
		""" Using the current best guess, find the solution error """
		
		# Update Iac in solution
		self.soln.Iac = Iac
		self.soln.Ibias = Idc
		
		# Update inductance estimate
		self.solve_inductance(Iac, Idc)
		
		# Find Z0 of chip
		self.soln.Vp = 1/np.sqrt(self.C_ * self.soln.L_)
		self.soln.Zchip = np.sqrt(self.soln.L_ / self.C_)
		
		# Find electrical length of chip (from phase velocity)
		self.soln.betaL = 2*PI*self.l_phys*self.freq*np.sqrt(self.C_ * self.soln.L_)
		
		# Find input impedance to chip
		self.soln.Zin = xfmr(self.soln.Zchip, self.Zcable, self.soln.betaL)
		
		# Get real and imag components of Zin
		Rin = np.real(self.soln.Zin)
		Xin = np.imag(self.soln.Zin)
		
		# Find angle of Zin
		self.theta = np.angle(self.soln.Zin)
		
		# Calcualte transmitted power (to chip)
		self.soln.P0 = np.abs(self.Vgen)**2 * Rin * 0.5 / ((Rin + self.Rsrc)**2 + (Xin + self.Xsrc)**2)
		
		# Find resulting Iac and error
		self.soln.Iac_result_td = np.sqrt(2*self.soln.P0/np.cos(self.theta)/self.soln.Zin)
		err_list = np.abs(self.soln.Iac_result_td - self.soln.Iac)
		self.soln.rmse = np.sqrt(np.mean(err_list**2))
		
		# Save to logger
		logging.info(f"Solution error: {round(self.soln.rmse*1000)/1000}")
		
		if show_plot:
			plot_Ts = 5
			idx_end = find_nearest(self.t, plot_Ts/self.freq)
			
			# plt.plot(self.t*1e9, self.soln.Iac_result_td*1e3)
			# plt.xlabel("Time (ns)")
			# plt.ylabel("AC Current (mA)")
			# plt.title("Solution Iteration Time Domain Plot")
			# plt.show()
			
			plt.plot(self.t[:idx_end]*1e9, self.soln.Lk[:idx_end]*1e9)
			plt.xlabel("Time (ns)")
			plt.ylabel("Lk (nH)")
			plt.title("Solution Iteration Time Domain Plot")
			plt.grid()
			plt.show()
		
	def get_spec_components(self, show_plot=False):
		""" Uses the current solution and calculates the spectral components"""
		
		y = self.soln.Iac_result_td
		num_pts = len(self.soln.Iac_result_td)
		dt = self.t[1]-self.t[0]
		
		# Run FFT
		spec_raw = fft(y)[:num_pts//2]
		
		# Fix magnitude to compensate for number of points
		spec = 2.0/num_pts*np.abs(spec_raw)
		
		# Get corresponding x axis (frequencies)
		spec_freqs = fftfreq(num_pts, dt)[:num_pts//2]
		
		if self.use_interp:
			logging.warning("interp feature has not been implemented. See fft_demo.py to see how.")
		
		# Find index of peak - Fundamental
		idx = find_nearest(spec_freqs, self.freq)
		Iac_fund = np.max([ spec[idx-1], spec[idx], spec[idx+1] ])
		
		# Find index of peak - Fundamental
		idx = find_nearest(spec_freqs, self.freq*2)
		Iac_2H = np.max([ spec[idx-1], spec[idx], spec[idx+1] ])
		
		# Find index of peak - Fundamental
		idx = find_nearest(spec_freqs, self.freq*3)
		Iac_3H = np.max([ spec[idx-1], spec[idx], spec[idx+1] ])
		
		# Save to solution set
		self.Iac_result_spec = (Iac_fund, Iac_2H, Iac_3H)
		
		logging.info(f"Calcualted Iac spectral components: fund={rd(Iac_fund*1e6, 3)}, 2H={rd(Iac_2H*1e6, 3)}, 3H={rd(Iac_3H*1e6, 3)} uA")
		
		if show_plot:
			
			plt.plot(spec_freqs/1e9, spec*1e3)
			plt.xlabel("Frequency (GHz)")
			plt.ylabel("AC Current (mA)")
			plt.title("Solution Iteration Spectrum")
			
			plt.show()
		
		