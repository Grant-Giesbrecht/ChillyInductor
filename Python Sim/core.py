from colorama import Fore, Style
from base import *

import logging
import getopt, sys
from Simulator_ABCD import *
from Simulator_P0 import *


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

logging.basicConfig(format=f'{prime_color}%(levelname)s:{standard_color} %(message)s{quiet_color} | %(asctime)s{Style.RESET_ALL}', level=LOG_LEVEL)

def simcode_to_str(sim_id:int):
	""" Accepts a sim code and returns the simulator's name"""
	
	if sim_id == SIMULATOR_ABCD:
		return LKSimABCD.NAME
	elif sim_id == SIMULATOR_P0:
		return LKSimP0.NAME
	
	return "?"



class Simopt:
	
	# Simulation class to use
	simulator = SIMULATOR_ABCD

class LKSystem:
	
	def __init__(self, Pgen_dBm:float, C_:float, l_phys:float, freq:float, q:float, L0:float, max_harm:int=6, ZL=50, Zg=50):
		
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
		self.ZL = ZL # Impedance of load
		self.Zg = Zg # Impedance of generator
		self.Vgen  = np.sqrt(self.Pgen*200) # Solve for Generator voltage from power
		self.max_harm = max_harm # Harmonic number to go up to in spectral domain (plus DC)
		self.system_loss = None # Tuple containing system loss at each harmonic (linear scale, not dB)
		self.Itickle = None # Amplitude (A) of tickle signal (Set to none to exclude tickle)
		self.freq_tickle = None # Amplitude (A) of tickle signal (Set to none to exclude tickle)
		self.harms = np.array(range(self.max_harm+1)) # List of harmonic numbers to include in spectral analysis
		
		# Time domain options
		self.num_periods = None
		self.num_periods_tickle = None # If tickle is included, this may not be none, in which case this will set the t_max time (greatly extending simulation time)
		self.max_harm_td = None
		self.min_points_per_wave = None
		self.t = []
		
		# Create solution object
		self.soln = LKSolution() # Current solution data
		
		self.solution = [] # List of solution data
		self.bias_points = [] # List of bias values corresponding to solution data
		
		self.sim_abcd = LKSimABCD(self)
		self.sim_p0 = LKSimP0(self)
		self.simulators = [self.sim_abcd, self.sim_p0]
		
		self.configure_time_domain(1000, 3, 30)
	
	def configure_tickle(self, Itickle:float, freq_tickle:float, num_periods:float=20):
		""" This function configures the tickle variables, enabling a tickle signal to be
		included in the simulation. Current is the amplitude in amps, and freq is in Hz."""
		
		# Configure the tickle for each simulator
		
		for sim in self.simulators:
			sim.configure_tickle(Itickle, freq_tickle, num_periods)
		
	def configure_time_domain(self, num_periods:float, max_harm_td:int, min_points_per_wave:int=10):
		""" Configures the time domain settings
		 
		  num_periods: Minimum number of periods to simulate
		  max_harm_td: Number of harmoinics to simulate
		  min_points_per_wave: Minimum number of time points per wavelength (at all frequencies)
		  
		"""
		
		# Configure time domain for each simulator
		
		for sim in self.simulators:
			sim.configure_time_domain(num_periods, max_harm_td, min_points_per_wave)
		
	def configure_loss(self, file:str=None, sparam_data:dict=None):
		""" Reads a pkl file with a dictionary containing variables 'freq_Hz' and 'S21_dB'
		and calculates the loss at each of the simulated harmonics. Can provide the dictionary
		without specifying a filename by using the sparam_data argument.
		"""
		
		# Configure loss for each simulator
		
		for sim in self.simulators:
			sim.configure_loss(file, sparam_data)
	
	def setopt(self, param:str, value):
		"""Changes a parameter for every simulator."""
		
		# List all simulators
		simulators_ = [s for s in self.simulators]
		simulators_.append(self)
		
		
		# Iterate over all simulators
		for sim in simulators_:
			
			# Skip simulator if parameter is not present
			if not hasattr(sim.opt, param):
				continue
			
			# Set value
			setattr(sim.opt, param, value)
	
	def validate_simulator_code(self, sim_id:int):
		""" Verifies that the code provided indicates a recognized simulator"""
		
		if sim_id == SIMULATOR_ABCD:
			return True
		
		if sim_id == SIMULATOR_P0:
			return True
		
		return False
	
	def select_simulator(self, sim_id:int):
		""" Specifies which simulator(s) to use by default. """
		
		# Check that valid simulator was provided
		if not self.validate_simulator_code(sim_id):
			logging.warning("Unrecognized simulator ID provided.")
			return
		
		# Update simulator
		self.opt.simulator = sim_id
	
	def solve(self, Ibias_vals:list, simulator:int=None):
		""" Solve the system at the bias values specified by Ibias_vals.
		
		If 'simulator' is specified, this simulator will be used instead of whatever
		was specified as the default by 'select_simulator()', however it will not modify
		the default/future function calls.
		
		"""
		# Auto-select simulator
		use_simulator = self.opt.simulator
		if simulator is not None:
			
			# Verify that a valid simulator was provided
			if not self.validate_simulator_code(simulator):
				logging.warning("Unrecognized simulator ID provided. Using default: {}")
			else:
				use_simulator = simulator
		
		# Check which simulator is selected
		if self.opt.simulator == SIMULATOR_ABCD:
			
			# Specify simulator
			sim = self.sim_abcd
			logging.info(f"Beginning solve with simulator: {Fore.LIGHTBLUE_EX}{sim.NAME}{Style.RESET_ALL}")
			
			# Run simulation
			sim.solve(Ibias_vals)
		elif self.opt.simulator == SIMULATOR_P0:
			
			# Specify simulator
			sim = self.sim_p0
			logging.info(f"Beginning solve with simulator: {Fore.LIGHTBLUE_EX}{sim.NAME}{Style.RESET_ALL}")
			
			# Run simulation
			sim.solve(Ibias_vals)
		else:
			logging.warning("Unrecognized simulator selected.")
	
	def get_solution(self, parameter:str=None, simulator:int=None):
		""" Returns solution data """
		
		# Auto-select simulator
		use_simulator = self.opt.simulator
		if simulator is not None:
			
			# Verify that a valid simulator was provided
			if not self.validate_simulator_code(simulator):
				logging.warning(f"Unrecognized simulator ID provided. Using default: {simcode_to_str(use_simulator)}")
			else:
				use_simulator = simulator
		
		# Get full dataset
		if use_simulator == SIMULATOR_ABCD:
			soln = self.sim_abcd.get_solution()
		elif use_simulator == SIMULATOR_P0:
			soln = self.sim_p0.get_solution()
		else:
			logging.error("Failed to recognize simulator")
			return None
		
		# If no parameter was requested, return the full dataset
		if parameter is None:
			return soln
		
		# Return extracted parameter
		return soln_extract(soln, parameter)
		
		