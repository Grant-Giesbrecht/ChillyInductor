from colorama import Fore, Style
from base import *

import logging
import getopt, sys
from Simulator_ABCD import *


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

@dataclass
class LKSolution:
	
	# Name of simulator that generated solution
	source_simulator = None
	
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
		
		self.configure_time_domain(1000, 3, 30)
	
	def configure_tickle(self, Itickle:float, freq_tickle:float, num_periods:float=20):
		""" This function configures the tickle variables, enabling a tickle signal to be
		included in the simulation. Current is the amplitude in amps, and freq is in Hz."""
		
		# Configure the tickle for each simulator
		
		self.sim_abcd.configure_tickle(Itickle, freq_tickle, num_periods)
		
	def configure_time_domain(self, num_periods:float, max_harm_td:int, min_points_per_wave:int=10):
		""" Configures the time domain settings
		 
		  num_periods: Minimum number of periods to simulate
		  max_harm_td: Number of harmoinics to simulate
		  min_points_per_wave: Minimum number of time points per wavelength (at all frequencies)
		  
		"""
		
		# Configure time domain for each simulator
		
		self.sim_abcd.configure_time_domain(num_periods, max_harm_td, min_points_per_wave)
		
	def configure_loss(self, file:str=None, sparam_data:dict=None):
		""" Reads a pkl file with a dictionary containing variables 'freq_Hz' and 'S21_dB'
		and calculates the loss at each of the simulated harmonics. Can provide the dictionary
		without specifying a filename by using the sparam_data argument.
		"""
		
		# Configure loss for each simulator
		
		self.sim_abcd.configure_loss(file, sparam_data)
	
	def setopt(self, param:str, value):
		"""Changes a parameter for every simulator."""
		
		# List all simulators
		simulators = [self, self.sim_abcd]
		
		# Iterate over all simulators
		for sim in simulators:
			
			# Skip simulator if parameter is not present
			if not hasattr(sim.opt, param):
				continue
			
			# Set value
			setattr(sim.opt, param, value)
	
	def select_simulator(self, sim_id:int):
		""" Specifies which simulator(s) to use."""
		
		# Check that ID is recognized
		if sim_id != SIMULATOR_ABCD:
			logging.warning("Unrecognized simulator ID provided.")
			return
		
		# Update simulator
		self.opt.simulator = sim_id
	
	def solve(self, Ibias_vals:list):
		
		# Check which simulator is selected
		if self.opt.simulator == SIMULATOR_ABCD:
			
			# Specify simulator
			sim = self.sim_abcd
			logging.info(f"Beginning solve with simulator: {Fore.LIGHTBLUE_EX}{sim.NAME}{Style.RESET_ALL}")
			
			# Run simulation
			sim.solve(Ibias_vals)
		else:
			logging.warning("Unrecognized simulator selected.")