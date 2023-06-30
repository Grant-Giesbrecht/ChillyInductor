import matplotlib.pyplot as plt
import logging 
import math
from colorama import Fore, Style
import numpy as np
from dataclasses import dataclass

tabchar = "    "
prime_color = Fore.YELLOW
standard_color = Fore.WHITE
quiet_color = Fore.LIGHTBLACK_EX
cspecial = Fore.GREEN # COlor used to highlight content inside logging messages

PI = 3.1415926535

# Starting Iac guess options
GUESS_ZERO_REFLECTION = 1
GUESS_USE_LAST = 2

@dataclass
class LKSolution:
	""" General solution class. Each simulator can use its own solution object 
	to save data specific about its solving process, but all classes need to 
	translate their results to this format for compatability and comparison."""
	
	# Scalar/configuration variables
	Iac_g = None
	Ibias_c = None
	Vgen_c = None
	
	# Spectrum Data
	Ig_w = None
	Ig_wf = None
	
	IL_w = None
	IL_wf = None
	VL_w = None
	VL_wf = None
	
	freq_w = None
	freq_wf = None
	
	# Convergence data
	convergence_failure = None # Set as True if fails to converge
	num_iter = None # Number of iterations completed
	Iac_guess_history = None # List of all Iac guesses
	guess_coef_history = None # List of all guess_coefficeints
	error_history = None # List of all error values during converge
	error_history_pcnt = None # List of all error values during converge
	
	# Name of simulator that generated solution
	source_simulator = None

def soln_extract(solution:list, param:str, conv_only:bool=True, element:int=None):
	""" Takes a solution list and extracts the specified parameter as 
	a numpy array """
	
	# Handle empty set
	if len(solution) < 1:
		return []
	
	print(f"{Fore.LIGHTMAGENTA_EX}element = {element}{Style.RESET_ALL}")
	
	# Ensure attribute is present
	if not hasattr(solution[0], param):
		logging.warning("Requested attribute is not present")
		return None
	
	# Extract the relevant parameter for each solution point
	list_data = []
	for x in solution:
		
		# Skip non-converged points
		if conv_only and (x.convergence_failure):
			continue
		
		# Get data
		new_data = getattr(x, param)
		if (isinstance(new_data, list) or isinstance(new_data, np.ndarray)) and (element is not None):
			
			# Check bounds, else modify new_data
			if len(new_data) <= element:
				logging.warning("Failed to observe 'element' parameter; out of bounds.")
			else:
				new_data = new_data[element]
		else:
			print(f"\tis instance: {isinstance(new_data, list)}, type = {type(new_data)}")
		
		print(f"\tappending: {Fore.LIGHTYELLOW_EX}{new_data}{Style.RESET_ALL}")
		
		# Add data
		list_data.append(new_data)
	
	return np.array(list_data)


class CMap:
	
	def __init__(self, cmap_name:str, N:int=None, data:list=None):
		
		self.cmap_name = cmap_name
		
		if N is not None:
			self.N = N
		
		if data is not None:
			self.N = len(data)	
		
		self.cm = plt.get_cmap(cmap_name)
		self.cm = self.cm.resampled(self.N)
		
	def __call__(self, idx:int):
		
		return self.cm(int(idx))

def addLoggingLevel(levelName, levelNum, methodName=None):
	
	"""
	Comprehensively adds a new logging level to the `logging` module and the
	currently configured logging class.

	`levelName` becomes an attribute of the `logging` module with the value
	`levelNum`. `methodName` becomes a convenience method for both `logging`
	itself and the class returned by `logging.getLoggerClass()` (usually just
	`logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
	used.

	To avoid accidental clobberings of existing attributes, this method will
	raise an `AttributeError` if the level name is already an attribute of the
	`logging` module or if the method name is already present 

	Example
	-------
	>>> addLoggingLevel('TRACE', logging.DEBUG - 5)
	>>> logging.getLogger(__name__).setLevel("TRACE")
	>>> logging.getLogger(__name__).trace('that worked')
	>>> logging.trace('so did this')
	>>> logging.TRACE
	5
	
	SOURCE:
	https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945#35804945
	
	"""
	if not methodName:
		methodName = levelName.lower()

	if hasattr(logging, levelName):
		raise AttributeError('{} already defined in logging module'.format(levelName))
	if hasattr(logging, methodName):
		raise AttributeError('{} already defined in logging module'.format(methodName))
	if hasattr(logging.getLoggerClass(), methodName):
		raise AttributeError('{} already defined in logger class'.format(methodName))

	# This method was inspired by the answers to Stack Overflow post
	# http://stackoverflow.com/q/2183233/2988730, especially
	# http://stackoverflow.com/a/13638084/2988730
	def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(levelNum):
			self._log(levelNum, message, args, **kwargs)
	def logToRoot(message, *args, **kwargs):
		logging.log(levelNum, message, *args, **kwargs)

	logging.addLevelName(levelNum, levelName)
	setattr(logging, levelName, levelNum)
	setattr(logging.getLoggerClass(), methodName, logForLevel)
	setattr(logging, methodName, logToRoot)
	
def rd(x:float, num_decimals:int=2):
	
	if x is None:
		return "NaN"
	
	return f"{round(x*10**num_decimals)/(10**num_decimals)}"

def rdl(L:list, num_decimals:int=2):
	
	S = "["
	
	# Scan over list
	for item in L:
		S = S + rd(item, num_decimals) + ", "
	
	S = S[:-2] + "]"
	return S
	
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


