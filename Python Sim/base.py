import matplotlib.pyplot as plt
import logging 
import math
from colorama import Fore, Style
import numpy as np

tabchar = "    "
prime_color = Fore.YELLOW
standard_color = Fore.WHITE
quiet_color = Fore.LIGHTBLACK_EX
cspecial = Fore.GREEN # COlor used to highlight content inside logging messages

PI = 3.1415926535

# Starting Iac guess options
GUESS_ZERO_REFLECTION = 1
GUESS_USE_LAST = 2

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


