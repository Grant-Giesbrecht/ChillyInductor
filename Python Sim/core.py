from colorama import Fore, Style
import numpy as np
from scipy.fft import fft, fftfreq

from base import *

import copy
import logging
import getopt, sys
from dataclasses import dataclass
import math

addLoggingLevel('MAIN', logging.INFO + 5)

LOG_LEVEL = logging.WARNING

#-----------------------------------------------------------
# Parse arguments
argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "debug", "info", "main", "warning", "error", "critical"])
except getopt.GetoptError as err:
	print("--help for help")
	sys.exit(2)
for opt, aarg in opts:
	if opt in ("-h", "--help"):
		print(f"{Fore.RED}Just kidding I haven't made any help text yet. ~~OOPS~~{Style.RESET_ALL}")
		sys.exit()
	elif opt == "--debug":
		LOG_LEVEL = logging.DEBUG
	elif opt == "--main":
		LOG_LEVEL = logging.MAIN
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
cspecial = Fore.GREEN # COlor used to highlight content inside logging messages
logging.basicConfig(format=f'{prime_color}%(levelname)s:{standard_color} %(message)s{quiet_color} | %(asctime)s{Style.RESET_ALL}', level=LOG_LEVEL)

PI = 3.1415926535

# Starting Iac guess options
GUESS_ZERO_REFLECTION = 1
GUESS_USE_LAST = 2

@dataclass
class Simopt:
	""" Contains simulation options"""
	
	# Simulation options
	use_interp = False # This option allows FFT interpolation, but has NOT been implemented
	
	# Convergence options
	max_iter = 1000 # Max iterations for convergence
	tol_pcnt = 1 # Tolerance in percent between Iac guesses
	guess_update_coef = 0.5 # Fraction by which to compromise between guess and result Iac (0=remain at guess, 1=use result; 0.5 recommended)
	ceof_shrink_factor = 0.2 # Fraction by which to modify guess_update_coef when sign reverses (good starting point: 0.2)
	
	# How to pick initial Iac guess
	start_guess_method = GUESS_ZERO_REFLECTION

@dataclass
class LKSolution:
	""" Contains data to represent a solution to the LKsystem problem"""
	
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
	
	Iac_result_rms = None # Magnitude of Iac
	Iac_result_td = None # Use Iac to find solution and calculate Iac again, this is the result in time domain
	Iac_result_spec = None # Iac result as spectrum, shows fundamental, 2harm, and 3harm as touple (idx 0 = fund, ..., 2 = 3rd harm)
	rmse = None # |Iac_result - Iac|
	
	convergence_failure = None # Set as True if fails to converge
	num_iter = None # Number of iterations completed

	spec = None # Spectrum data [AC Current amplitude in Amps]
	spec_freqs = None # Spectrum frequencies in Hz

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
		self.opt = Simopt()

		# System Settings
		self.Pgen = (10**(Pgen_dBm/10))/1000
		self.C_ = C_
		self.l_phys = l_phys
		self.freq = freq
		self.q = q
		self.L0 = L0
		self.Zcable = 50 # Z0 of cable leading into chip
		self.Rsrc = 50 # R of generator
		self.Xsrc = 0 # X of generator
		self.Vgen  = np.sqrt(self.Pgen*200) # Solve for Generator voltage from power
		
		# Time domain options
		self.num_periods = None
		self.max_harm = None
		self.min_points_per_wave = None
		self.t = []

		# Create solution object
		self.soln = LKSolution() # Current solution data
		
		self.solution = [] # List of solution data
		self.bias_points = [] # List of bias values corresponding to solution data
		
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
		
		logging.info(f"Configured time domain with {cspecial}{len(self.t)}{standard_color} points.")
		
	def crunch(self, Iac:float, Idc:float, show_plot_td=False, show_plot_spec=False):
		""" Using the provided Iac guess, find the reuslting solution, and the error
		between the solution and the initial guess.
		
		Converts the resulting Iac time domain data into spectral components, saving the
		fundamental through 3rd hamonic as a touple, with idx=0 assigned to fundamental.
		
		"""
		
		# Update Iac in solution
		self.soln.Iac = Iac
		self.soln.Ibias = Idc
		
		# Solve for inductance (Lk)
		self.soln.Lk = self.L0 + self.L0/self.q**2 * ( Idc**2 + 2*Idc*Iac*np.sin(self.freq*2*PI*self.t) + Iac**2/2 - Iac**2/2*np.cos(2*self.freq*2*PI*self.t) ) 
		
		self.soln.L_ = self.soln.Lk/self.l_phys
		
		# # Update inductance estimate
		# self.solve_inductance(Iac, Idc)
		
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
		self.soln.Iac_result_rms = np.sqrt(2*self.soln.P0/np.cos(self.theta)/self.soln.Zin)
		self.soln.Iac_result_td = np.sqrt(2)*self.soln.Iac_result_rms*np.sin(2*PI*self.freq*self.t) #TODO: Need a factor of sqrt(2)
		err_list = np.abs(self.soln.Iac_result_rms - self.soln.Iac) # Error in signal *amplitude* at each time point
		self.soln.rmse = np.sqrt(np.mean(err_list**2))
		
		# Save to logger
		logging.debug(f"Solution error: {round(self.soln.rmse*1000)/1000}")
		
		if show_plot_td:
			plot_Ts = 5
			idx_end = find_nearest(self.t, plot_Ts/self.freq)
			
			plt.plot(self.t*1e9, self.soln.Iac_result_td*1e3)
			plt.xlabel("Time (ns)")
			plt.ylabel("AC Current (mA)")
			plt.title("Solution Iteration Time Domain Plot")
			plt.show()
			
			# plt.plot(self.t[:idx_end]*1e9, self.soln.Lk[:idx_end]*1e9)
			# plt.xlabel("Time (ns)")
			# plt.ylabel("Lk (nH)")
			# plt.title("Solution Iteration Time Domain Plot")
			# plt.grid()
			# plt.show()
		
		############# Previously called get_spec_components() ###########
		
		y = self.soln.Iac_result_td
		num_pts = len(self.soln.Iac_result_td)
		dt = self.t[1]-self.t[0]
		
		# Run FFT
		spec_raw = fft(y)[:num_pts//2]
		
		# Fix magnitude to compensate for number of points
		spec = 2.0/num_pts*np.abs(spec_raw)
		self.soln.spec = spec
		
		# Get corresponding x axis (frequencies)
		spec_freqs = fftfreq(num_pts, dt)[:num_pts//2]
		self.soln.spec_freqs = spec_freqs
		
		if self.opt.use_interp:
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
		self.soln.Iac_result_spec = (Iac_fund, Iac_2H, Iac_3H)
		
		logging.debug(f"Calcualted Iac spectral components: fund={rd(Iac_fund*1e6, 3)}, 2H={rd(Iac_2H*1e6, 3)}, 3H={rd(Iac_3H*1e6, 3)} uA")
		
		if show_plot_spec:
			
			plt.plot(spec_freqs/1e9, spec*1e3)
			plt.xlabel("Frequency (GHz)")
			plt.ylabel("AC Current (mA)")
			plt.title("Solution Iteration Spectrum")
			
			plt.show()
		
	def plot_solution(self, s:LKSolution=None):
		
		# Pick last solution if none provided
		if s is None:
			s = self.soln
		
		# calculate end index
		plot_Ts = 5
		idx_end = find_nearest(self.t, plot_Ts/self.freq)
		
		# Create time domain figure
		plt.figure(1)
		plt.plot(self.t[:idx_end]*1e9, np.real(s.Iac_result_td[:idx_end])*1e3, '-b')
		plt.plot(self.t[:idx_end]*1e9, np.abs(s.Iac_result_td[:idx_end])*1e3, '-r')
		plt.plot(self.t[:idx_end]*1e9, np.sqrt(2)*np.abs(s.Iac_result_rms[:idx_end])*1e3, '-g')
		plt.xlabel("Time (ns)")
		plt.ylabel("AC Current (mA)")
		plt.title(f"Time Domain Data, Idc = {rd(s.Ibias*1e3)} mA")
		plt.legend(["TD Real", "TD Abs.", "|Amplitude|"])
		plt.grid()
		
		# Create spectrum figure
		plt.figure(2)
		plt.plot(s.spec_freqs/1e9, s.spec*1e3)
		plt.xlabel("Frequency (GHz)")
		plt.ylabel("AC Current (mA)")
		plt.title(f"Current Spectrum, Idc = {rd(s.Ibias*1e3)} mA")
		plt.xlim((0, self.freq*5/1e9))
		plt.grid()
		
		plt.show()
	
	def solve(self, Ibias_vals:list, show_plot_on_conv=False):
		""" Takes a list of bias values, plugs them in for Idc, and solves for
		the AC current s.t. error is within tolerance. """
		
		logging.info(f"Beginning iterative solve for {cspecial}{len(Ibias_vals)}{standard_color} bias points.")
		
		Iac_crude_guess = np.abs(self.Vgen)**2 / 50 # Crude guess for AC current (Assume 50 ohm looking into chip)
		
		# Scan over each bias value
		for Idc in Ibias_vals:
			
			last_sign = None # Sign of last change
			self.soln.num_iter = 0	# Reset iteration counter
			guess_coef = self.opt.guess_update_coef # Reset guess_coef
			
			if self.opt.start_guess_method == GUESS_ZERO_REFLECTION:
				Iac_guess = Iac_crude_guess
			elif self.opt.start_guess_method == GUESS_USE_LAST:
				if len(self.solution) > 0:
					Iac_guess = self.solution[-1].Iac
				else:
					Iac_guess = Iac_crude_guess
			
			# Loop until converge
			while True:
				
				# Crunch the numbers of this guess value
				self.crunch(Iac_guess, Idc)
				self.soln.num_iter += 1
				
				# Calculate signed error
				error = self.soln.Iac_result_spec[0] - Iac_guess
				error_pcnt = (np.max([self.soln.Iac_result_spec[0], Iac_guess])/np.min([self.soln.Iac_result_spec[0], Iac_guess])-1)*100
				
				# Check for convergence
				if error_pcnt < self.opt.tol_pcnt:
					
					# Add to logger
					logging.info(f"Datapoint ({cspecial}Idc={rd(Idc*1e3)} mA{standard_color}),({cspecial}Iac={rd(Iac_guess*1e3, 3)} mA{standard_color}) converged with {cspecial}error={rd(error_pcnt, 3)}%{standard_color} after {cspecial}{self.soln.num_iter}{standard_color} iterations ")
					
					# Create deep
					new_soln = copy.deepcopy(self.soln)
					new_soln.convergence_failure = True
					
					# Add solution to list
					self.solution.append(new_soln)
					
					# Add bias point to list
					self.bias_points.append(Idc)
					
					# Plot result if requested
					if show_plot_on_conv:
						self.plot_solution()
					
					# Exit convergence loop
					break
					
				# Check for exceed max iterations
				elif self.soln.num_iter >= self.opt.max_iter:
					
					# Add to logger
					logging.warning(f"Failed to converge for point ({cspecial}Idc={rd(Idc*1e3)} mA{standard_color}).")
					
					# Exit convergence loop
					break
				
				# Else update guess
				else:
					
					# Error is positive
					if error > 0:
						
						# First iteration - set sign
						if last_sign is None:
							last_sign = np.sign(error)
						# Last change was in different direction
						elif last_sign < 1:
							guess_coef *= self.opt.ceof_shrink_factor # Change update size
							last_sign = -1 # Update change direction
							logging.debug(f"Error sign changed. Changing guess update coefficient to {cspecial}{guess_coef}{standard_color}")
						
						# Update guess
						Iac_guess = Iac_guess + error * guess_coef
					
					# Else error is negative
					else:
						
						# First iteration - set sign
						if last_sign is None:
							last_sign = np.sign(error)
						# Last change was in different direction
						elif last_sign > 1:
							guess_coef *= self.opt.ceof_shrink_factor # Change update size
							last_sign = 1 # Update change direction
							logging.debug(f"Error sign changed. Changing guess update coefficient to {cspecial}{guess_coef}{standard_color}")
							
						# Update guess
						Iac_guess = Iac_guess + error * guess_coef
					
					logging.debug(f"Last guess produced {cspecial}error={error}{standard_color}. Updated guess to {cspecial}Iac={Iac_guess}{standard_color}. [{cspecial}iter={self.soln.num_iter}{standard_color}]")